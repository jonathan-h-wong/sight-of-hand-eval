"""HO-3D v3 (Hampali et al. 2020/2022). Single right hand on one YCB object, multi-camera optimization (HOnnotate).

Local layout (scripts/prepare_data.sh ho3d; only labels are extracted, no images):
    ROOT/{train,evaluation}/<seq>/meta/<frame>.pkl     python2 pickle (latin1)
    ROOT/{train,evaluation}.txt                          "<seq>/<frame>" per line
    ROOT/evaluation_verts/<seq>.npz                      frames, verts (N,778,3) m, from HO3D_V3_GT/evaluation_verts.json
    ROOT/models/<ycb_name>/textured_simple.obj           YCB meshes (metres; from raw/dexycb/models.tar.gz)

Conventions (shreyashampali/ho3d vis_HO3D.py):
  hand   : MANO right, fullpose = handPose (48 = 3 global + 45 articulation, absolute: the chumpy wrapper sets
           m.fullpose directly, i.e. no hands_mean -> flat_hand_mean=True), betas = handBeta, + handTrans.
           Train split only; frames whose handPose is None are untracked.
           Evaluation split: meta carries no hand pose; the released GT vertices (evaluation_verts.json, in the
           order of evaluation.txt) are in the challenge's OpenGL frame and are flipped back (y, z) * -1.
  object : verts = v @ Rodrigues(objRot).T + objTrans  (world = R v + t), both splits.
  Everything is in the camera frame (OpenCV axes for meta), metres; ~30 fps.
Mask key: <split>/<seq>; frame = int(<frame>) (the meta / rgb file stem).
"""
import pickle
from pathlib import Path

import numpy as np

from ..metrics import Part, Sequence
from .common import T_from, aa_to_R, load_mesh, mano

FPS = 30.0
GL = np.array([1.0, -1.0, -1.0])  # OpenGL <-> OpenCV axis flip
_PARTS = {}


def list_tasks(root):
    return sorted(str(p) for p in Path(root).glob("*/*") if (p / "meta").is_dir())


def _part(root, name):
    if name not in _PARTS:
        V, F = load_mesh(Path(root) / "models" / name / "textured_simple.obj")
        _PARTS[name] = Part(V * 1000.0, F)
    return _PARTS[name]


def _meta(p):
    with open(p, "rb") as f:
        return pickle.load(f, encoding="latin1")


def load(task, stride=1):
    task = Path(task)
    split, seq = task.parent.name, task.name
    root = task.parent.parent
    files = sorted(task.glob("meta/*.pkl"), key=lambda p: int(p.stem))[::stride]
    fr = np.array([int(p.stem) for p in files])
    metas = [_meta(p) for p in files]
    names = {m["objName"] for m in metas if m.get("objName")}
    assert len(names) == 1, (task, names)  # one object per HO3D sequence
    name = names.pop()
    n = len(fr)
    T_wo = np.full((n, 4, 4), np.nan)
    has_obj = np.array([m.get("objRot") is not None and m.get("objTrans") is not None for m in metas])
    if has_obj.any():
        rot = np.stack([np.asarray(metas[i]["objRot"], np.float64).ravel() for i in np.where(has_obj)[0]])
        tr = np.stack([np.asarray(metas[i]["objTrans"], np.float64).ravel() for i in np.where(has_obj)[0]])
        T_wo[has_obj] = T_from(aa_to_R(rot), tr * 1000.0)

    V = np.full((n, 778, 3), np.nan)
    if split == "train":
        valid = np.array([m.get("handPose") is not None for m in metas]) & has_obj
        if valid.any():
            idx = np.where(valid)[0]
            pose = np.stack([np.asarray(metas[i]["handPose"], np.float64).ravel() for i in idx])
            beta = np.stack([np.asarray(metas[i]["handBeta"], np.float64).ravel() for i in idx])
            trans = np.stack([np.asarray(metas[i]["handTrans"], np.float64).ravel() for i in idx])
            m = mano("right", True)
            for b in np.unique(beta, axis=0):  # betas are per subject; constant within a sequence in practice
                sel = np.all(beta == b, axis=1)
                V[idx[sel]] = m(b, pose[sel, :3], pose[sel, 3:48], trans[sel]) * 1000.0
    else:
        z = np.load(root / "evaluation_verts" / f"{seq}.npz")
        pos = {int(f): i for i, f in enumerate(z["frames"])}
        valid = np.array([f in pos for f in fr]) & has_obj
        for i in np.where(valid)[0]:
            V[i] = z["verts"][pos[fr[i]]] * GL * 1000.0
    extra = {"split": np.full(n, split), "object": np.full(n, name)}
    return Sequence("ho3d", f"{split}/{seq}", FPS, stride, fr, T_wo, [_part(root, name)], None,
                    {"right": (V, valid)}, extra)
