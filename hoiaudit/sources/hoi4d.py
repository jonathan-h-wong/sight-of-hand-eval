"""HOI4D (Liu et al. 2022). Egocentric RGB-D (Kinect), 15 fps, category-level objects with CAD models.

Local layout (scripts/prepare_data.sh hoi4d; labels only):
    ROOT/Hand_pose/handpose_{right,left}_hand/<path>/<frame>.pickle   poseCoeff(48), beta(10), trans(3) m
    ROOT/annotations/<path>/objpose/<frame>.json                       dataList/objects: center, rotation (euler), dims
    ROOT/CAD/rigid/<Category>/<NNN>.obj                                metres, canonical frame of the pose
<path> = ZY2021080000<cam>/H<human>/C<class>/N<instance>/S<room>/s<layout>/T<task>.

Conventions (leolyliu/HOI4D-Instructions README):
  hand   : manopth ManoLayer(use_pca=False, flat_hand_mean=True), theta = poseCoeff (3 global + 45), beta,
           verts + trans, camera frame, metres. Left hands use the stock MANO_LEFT.
  object : center (x, y, z) translation, rotation = scipy Rotation.from_euler('XYZ', (x, y, z)), camera frame.
           For rigid objects the CAD mesh (metres, watertight) is the canonical frame: world = R v + t.
           `dimensions` is the annotated box size and is not applied; `dim_ratio` (CAD bbox volume / annotated box
           volume, cube root) is reported per frame so a scale mismatch shows up in the audit.
  Only the seven rigid categories are loaded (Bottle, Bowl, Chair, Kettle, Knife, Mug, ToyCar). Articulated
  categories pose each part separately and are skipped (list_tasks filters them).
Checks (6 random rigid sequences, 2026-11-23): MANO joints reprojected with camera_params/<cam>/intrin.npy land on
the released kps2D at 3.9 px median with flat_hand_mean=True (36 px with False). Of the Euler orders XYZ / xyz / ZYX /
zyx, 'XYZ' puts the hand closest to the object surface (median |min_sd| 9 mm vs 23-28 mm) with the highest contact
rate; `dim_ratio` is 1.0-1.3. Contact frames still sit ~5-7 mm inside the object at the median: the labels penetrate.
Mask key: <path>; frame = int(<frame>), the frame index of align_rgb/image.mp4.
"""
import json
from pathlib import Path

import numpy as np

from ..metrics import Part, Sequence
from .common import T_from, load_mesh, mano

FPS = 15.0
CLASSES = ['', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle', 'Safe', 'Bowl', 'Bucket', 'Scissors', '',
           'Pliers', 'Kettle', 'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair']
RIGID = {"ToyCar", "Mug", "Bottle", "Bowl", "Kettle", "Knife", "Chair"}
_PARTS = {}


def _cls(path):
    return CLASSES[int(Path(path).parts[2][1:])]


def list_tasks(root):
    root = Path(root)
    seqs = set()
    for side in ("right", "left"):
        base = root / "Hand_pose" / f"handpose_{side}_hand"
        for d in base.glob("*/*/*/*/*/*/*"):
            seqs.add(str(d.relative_to(base)))
    return sorted(str(root / "seqs" / s) for s in seqs
                  if _cls(s) in RIGID and (root / "annotations" / s / "objpose").is_dir())


def _mesh_path(root, path):
    cat = _cls(path)
    n = int(Path(path).parts[3][1:])
    for name in (f"{n:03d}.obj", f"{n}.obj"):
        p = Path(root) / "CAD" / "rigid" / cat / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no CAD model for {cat} N{n}")


def _part(root, path):
    p = _mesh_path(root, path)
    if p not in _PARTS:
        V, F = load_mesh(p)
        _PARTS[p] = Part(V * 1000.0, F)
    return _PARTS[p]


def _objpose(f):
    from scipy.spatial.transform import Rotation
    d = json.loads(Path(f).read_text()) if Path(f).read_text().lstrip().startswith("{") else eval(Path(f).read_text())
    a = (d.get("dataList") or d.get("objects"))[0]
    c, r, s = a["center"], a["rotation"], a["dimensions"]
    t = np.array([c["x"], c["y"], c["z"]], np.float64)
    R = Rotation.from_euler("XYZ", [r["x"], r["y"], r["z"]]).as_matrix()
    dim = np.array([s["length"], s["width"], s["height"]], np.float64)
    return R, t, dim


def load(task, stride=1):
    import pickle
    task = Path(task)
    root = task.parents[7]
    path = str(task.relative_to(root / "seqs"))
    ann = root / "annotations" / path / "objpose"
    hand_files = {side: {int(p.stem): p for p in (root / "Hand_pose" / f"handpose_{side}_hand" / path).glob("*.pickle")}
                  for side in ("right", "left")}
    obj_files = {int(p.stem): p for p in ann.glob("*.json")}
    frames = sorted(set(obj_files) & (set(hand_files["right"]) | set(hand_files["left"])))[::stride]
    fr = np.array(frames, dtype=np.int64)
    n = len(fr)
    part = _part(root, path)
    T_wo = np.full((n, 4, 4), np.nan)
    dim_ratio = np.full(n, np.nan)
    ext = part.V.max(0) - part.V.min(0)
    for i, f in enumerate(fr):
        try:
            R, t, dim = _objpose(obj_files[f])
        except Exception:
            continue
        T_wo[i] = T_from(R[None], t[None] * 1000.0)[0]
        dim_ratio[i] = float((np.prod(ext) / max(1e-9, np.prod(dim * 1000.0))) ** (1 / 3))
    hands = {}
    for side in ("right", "left"):
        hf = hand_files[side]
        valid = np.array([f in hf for f in fr], dtype=bool) & np.isfinite(T_wo).all((1, 2))
        if not valid.any():
            continue
        V = np.full((n, 778, 3), np.nan)
        idx = np.where(valid)[0]
        ds = []
        for i in idx:
            with open(hf[fr[i]], "rb") as fh:
                ds.append(pickle.load(fh, encoding="latin1"))
        pose = np.stack([np.asarray(d["poseCoeff"], np.float64).ravel() for d in ds])
        beta = np.stack([np.asarray(d["beta"], np.float64).ravel() for d in ds])
        trans = np.stack([np.asarray(d["trans"], np.float64).ravel() for d in ds])
        m = mano(side, True)
        for b in np.unique(beta, axis=0):
            sel = np.all(beta == b, axis=1)
            V[idx[sel]] = m(b, pose[sel, :3], pose[sel, 3:48], trans[sel]) * 1000.0
        hands[side] = (V, valid)
    extra = {"category": np.full(n, _cls(path)), "dim_ratio": dim_ratio}
    return Sequence("hoi4d", path, FPS, stride, fr, T_wo, [part], None, hands, extra)
