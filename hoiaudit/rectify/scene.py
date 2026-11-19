"""Per-dataset adapters: one raw sequence -> everything the rectifier needs.

Scene fields
  seq      metrics.Sequence from hoiaudit.sources (exact penetration before/after uses it)
  hands    side -> dict(betas (10,), rot (T,3), pose (T,45) axis-angle, trans (T,3) metres, valid (T,), flat bool)
           Pose is the full 45-d axis-angle used with MANO(flat_hand_mean=flat); DexYCB's 45 PCA coefficients are
           converted once (exact: its PCA basis is full rank), so outputs share one convention.
  objects  list of (sdf grid dict, L (T,4,4)): L maps world (mm) -> that mesh's frame. Union = min over objects.
  cams     list of dict(name, K (3,3), W2C (T,4,4) world metres -> camera metres, size (W,H))
  obj      the object whose pose may be corrected (None = objects fixed):
           base (T,4,4)  world (mm) -> object frame; a correction D (T,4,4, object frame, mm) makes it inv(D) @ base
           movable       indices into `objects` that move with it (ARCTIC: both parts; DexYCB: the grasped object)
           pts (T,n,3)   sample points on the object, object frame (mm), for its camera / smoothness terms
           tracked (T,)  frames where the object pose exists
           move(D)       -> metrics.Sequence with the corrected object pose (for exact before/after metrics)
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .. import metrics
from ..metrics import inv_T
from ..sources import common
from . import sdfgrid


@dataclass
class Scene:
    dataset: str
    name: str
    seq: metrics.Sequence
    hands: dict
    objects: list
    cams: list
    obj: dict | None = None


def _check_parity(scene):
    """MANO params reproduce the source's GT meshes (guards against convention drift between the two loaders)."""
    for side, h in scene.hands.items():
        V_src, valid = scene.seq.hands[side]
        v = np.where(valid & h["valid"])[0][:50]
        if not len(v):
            continue
        m = common.mano(side, h["flat"])
        V = m(h["betas"], h["rot"][v], h["pose"][v], h["trans"][v]) * 1000.0
        err = np.abs(V - V_src[v]).max()
        assert err < 1e-3, f"{scene.name} {side}: MANO params differ from source mesh by {err:.4f} mm"


# ----------------------------------------------------------------------------- ARCTIC
def arctic(task, cache_dir):
    from ..sources import arctic as src
    seq = src.load(task)
    p = Path(task)
    root, sid, name = p.parents[2], p.parent.name, p.name[: -len(".mano.npy")]
    m = np.load(p, allow_pickle=True).item()
    T = len(seq.frame_idx)
    hands = {s: {"betas": np.asarray(m[s]["shape"], np.float64), "rot": np.asarray(m[s]["rot"], np.float64),
                 "pose": np.asarray(m[s]["pose"], np.float64), "trans": np.asarray(m[s]["trans"], np.float64),
                 "valid": seq.hands[s][1].copy(), "flat": False} for s in ("left", "right")}
    # articulated object: rigid bottom + top rotating about -z by the per-frame angle (metrics.Articulated)
    art = seq.parts
    tf = art.top[art.F]
    L_base = inv_T(seq.T_wo)
    objects = []
    for sel, is_top in ((~tf.any(1), False), (tf.all(1), True)):
        F = art.F[sel]
        used = np.unique(F)
        remap = np.full(len(art.V), -1)
        remap[used] = np.arange(len(used))
        g = sdfgrid.build(art.V[used], remap[F], cache_dir=cache_dir)
        if is_top:
            R = _rz(-seq.part_T)                    # same rotation metrics.Articulated.mesh_at applies
            Rt = np.zeros((T, 4, 4))
            Rt[:, :3, :3] = np.swapaxes(R, 1, 2)
            Rt[:, 3, 3] = 1
            objects.append((g, Rt @ L_base))
        else:
            objects.append((g, L_base))
    misc = json.load(open(root / "meta" / "misc.json"))[sid]
    e = np.load(str(p).replace(".mano.npy", ".egocam.dist.npy"), allow_pickle=True).item()
    W2C = np.zeros((T, 4, 4))
    W2C[:, :3, :3] = np.asarray(e["R_k_cam_np"], np.float64)
    W2C[:, :3, 3] = np.asarray(e["T_k_cam_np"], np.float64).reshape(-1, 3)
    W2C[:, 3, 3] = 1
    cams = [{"name": "ego", "K": np.asarray(e["intrinsics"], np.float64), "W2C": W2C, "size": tuple(misc["image_size"][0])}]
    for v in range(8):
        cams.append({"name": f"view{v + 1}", "K": np.asarray(misc["intris_mat"][v], np.float64),
                     "W2C": np.asarray(misc["world2cam"][v], np.float64)[None], "size": tuple(misc["image_size"][v + 1])})
    rng = np.random.default_rng(0)
    pick = rng.choice(len(art.V), min(400, len(art.V)), replace=False)
    pts = np.repeat(art.V[pick][None], T, 0)
    top = art.top[pick]
    Rf = _rz(-seq.part_T)                                     # top part at each frame's angle (as mesh_at)
    pts[:, top] = np.einsum("tij,tpj->tpi", Rf, pts[:, top])

    def move(D):
        import copy
        sq = copy.copy(seq)
        sq.T_wo = seq.T_wo @ D                                # world <- object, corrected in the object frame
        return sq

    obj = {"base": L_base, "movable": [0, 1], "pts": pts, "tracked": np.isfinite(seq.T_wo).all((1, 2)), "move": move}
    sc = Scene("arctic", f"{sid}/{name}", seq, hands, objects, cams, obj)
    _check_parity(sc)
    return sc


def _rz(a):
    from ..mano_np import rodrigues
    return rodrigues(np.stack([np.zeros_like(a), np.zeros_like(a), a], 1))


# ----------------------------------------------------------------------------- DexYCB
def dexycb(task, cache_dir):
    from ..sources import dexycb as src
    seq = src.load(task)
    task = Path(task)
    root = task.parent.parent
    meta = yaml.safe_load(open(task / "meta.yml"))
    pm = np.load(task / "pose.npz")["pose_m"][:, 0]
    side = meta["mano_sides"][0]
    m = common.mano(side, False)
    calib = root / "calibration"
    betas = np.array(yaml.safe_load(open(calib / f"mano_{meta['mano_calib'][0]}" / "mano.yml"))["betas"], np.float64)
    hands = {side: {"betas": betas, "rot": pm[:, :3].astype(np.float64), "pose": m.pca_to_pose(pm[:, 3:48]),
                    "trans": pm[:, 48:51].astype(np.float64), "valid": seq.hands[side][1].copy(), "flat": False}}
    objects = []
    for k, part in enumerate(seq.parts):
        objects.append((sdfgrid.build(part.V, part.F, cache_dir=cache_dir), seq.part_T[:, k]))
    ext = _yml(calib / f"extrinsics_{meta['extrinsics']}" / "extrinsics.yml")["extrinsics"]
    cams = []
    for s in meta["serials"]:
        C2W = np.eye(4)
        C2W[:3] = np.asarray(ext[s], np.float64).reshape(3, 4)   # camera -> master camera (= world), metres
        c = _yml(calib / "intrinsics" / f"{s}_640x480.yml")["color"]
        K = np.array([[c["fx"], 0, c["ppx"]], [0, c["fy"], c["ppy"]], [0, 0, 1]], np.float64)
        cams.append({"name": s, "K": K, "W2C": np.linalg.inv(C2W)[None], "size": (640, 480)})
    g = seq.grasp_k
    base = seq.part_T[:, g]
    tracked = np.abs(base[:, :3, 3]).max(-1) < 1e6           # untracked frames were moved far away by the source
    V = seq.parts[g].V
    pick = np.random.default_rng(0).choice(len(V), min(400, len(V)), replace=False)

    def move(D):
        import copy
        sq = copy.copy(seq)
        pt = seq.part_T.copy()
        pt[tracked, g] = inv_T(D[tracked]) @ seq.part_T[tracked, g]
        sq.part_T = pt
        return sq

    obj = {"base": base, "movable": [g], "pts": np.repeat(V[pick][None], len(base), 0), "tracked": tracked, "move": move}
    sc = Scene("dexycb", seq.name, seq, hands, objects, cams, obj)
    _check_parity(sc)
    return sc


class _TupleLoader(yaml.SafeLoader):
    """DexYCB calibration files tag sequences as !!python/tuple; read them as plain lists (no arbitrary objects)."""


_TupleLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda ld, n: ld.construct_sequence(n))


def _yml(path):
    return yaml.load(open(path), Loader=_TupleLoader)


ADAPTERS = {"arctic": arctic, "dexycb": dexycb}
