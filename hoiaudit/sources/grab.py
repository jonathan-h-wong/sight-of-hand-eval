"""GRAB (Taheri et al. 2020). Raw layout (after unzipping raw/grab/*.zip into ROOT):

    ROOT/grab/sN/<obj>_<intent>[_k].npz            (grab__sN.zip)
    ROOT/objects_info.npz                          (grab__objects_info.zip; GRAB contact meshes, metres)
    ROOT/{male,female}/sN_{r,l}hand.ply            (tools__subject_meshes__{male,female}.zip)

Conventions (GRAB/grab/save_grab_vertices.py, tools/objectmodel.py):
  hands  : smplx MANO, v_template = subject hand mesh, betas = 0, flat_hand_mean=True, 45-d `fullpose`
  object : verts = v_template @ R(global_orient) + transl   (row-vector form, i.e. world = R^T v + t)
  120 fps.
Mask key: sN/<seq>  (mirrors grab__sN.zip/sN/<seq>.npz). Every frame of both hands is evaluated.
"""
from pathlib import Path

import numpy as np

from ..metrics import Part, Sequence
from .common import T_from, aa_to_R, load_mesh, mano

_OBJ = {}
_VT = {}


def _obj_name(name):
    return {"cubemedium": "cubemiddle"}.get(name, name)


def list_tasks(root):
    root = Path(root)
    return sorted(str(p) for p in root.glob("s*/*.npz"))


def _vtemp(root, rel):
    # rel like 'tools/subject_meshes/male/s1_rhand.ply'
    p = Path(root) / Path(rel).parent.name / Path(rel).name
    if str(p) not in _VT:
        _VT[str(p)] = load_mesh(p)[0]
    return _VT[str(p)]


def load(task, stride=1):
    z = np.load(task, allow_pickle=True)
    sid = str(z["sbj_id"])
    seq = Path(task).stem
    obj = str(z["obj_name"])
    meta_root = Path(task).parent.parent
    if obj not in _OBJ:
        info = np.load(meta_root / "objects_info.npz", allow_pickle=True)
        d = info[_obj_name(obj)].item()
        _OBJ[obj] = [Part(np.asarray(d["verts"], np.float64) * 1000.0, np.asarray(d["faces"], np.int64))]
    N = int(z["n_frames"])
    fr = np.arange(0, N, stride)
    op = z["object"].item()["params"]
    # world = v @ R + t  =>  R_wo = R^T
    R = np.swapaxes(aa_to_R(op["global_orient"][fr]), 1, 2)
    T_wo = T_from(R, op["transl"][fr] * 1000.0)
    hands, extra = {}, {"object": np.full(len(fr), obj), "intent": np.full(len(fr), str(z["motion_intent"]))}
    contact = z["contact"].item()
    for side, key in (("right", "rhand"), ("left", "lhand")):
        h = z[key].item()
        p = h["params"]
        m = mano(side, True)
        vt = _vtemp(meta_root, h["vtemp"])
        saved = m.v_template
        m.v_template = vt
        try:
            V = m(np.zeros(10), p["global_orient"][fr], p["fullpose"][fr], p["transl"][fr]) * 1000.0
        finally:
            m.v_template = saved
        hands[side] = (V, np.isfinite(V).all((1, 2)))
    # GRAB's own per-frame object contact (any object vertex touched by the body), for validation
    extra["grab_obj_contact"] = (contact["object"][fr] > 0).any(1)
    return Sequence("grab", f"{sid}/{seq}", 120.0, stride, fr, T_wo, _OBJ[obj], None, hands, extra)
