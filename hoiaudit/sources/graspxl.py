"""GraspXL (Zhang et al. 2024), MANO subsets. Physics-simulated (RaiSim) grasps on Objaverse objects.

Local layout (see scripts/fetch_graspxl_sample.py):
    ROOT/recorded/<archive>/<size>/<obj_id>/<file>.npy     dict: right_hand{trans,rot,pose(45)}, <obj_id>{trans,rot[,angle]}
    ROOT/object_dataset/<size>/<obj_id>/<obj_id>.obj       metres

Conventions (GraspXL_visualization scripts/visualizer_mano.py, visualizer_sharpa_table_top.py):
  hand   : smplx MANO right, betas = 0, use_pca=False, verts = MANO(rot, pose) + trans
           flat_hand_mean=False for mano_dataset_N, flat_hand_mean=True for mano_tabletop
  object : verts = v @ R(-rot) + trans  ==  world = R(rot) v + t
Mask key: <archive>/<size>/<obj_id>/<file stem>.
"""
from pathlib import Path

import numpy as np

from ..metrics import Part, Sequence
from .common import T_from, aa_to_R, load_mesh, mano

FLAT = {"mano_tabletop": True}  # default False (mano_dataset_N)
FPS = 60.0  # not stored in the files; only affects the smoothness columns, not penetration
_PARTS = {}


def list_tasks(root):
    return sorted(str(p) for p in (Path(root) / "recorded").glob("*/*/*/*.npy"))


def load(task, stride=1, flat_override=None):
    p = Path(task)
    oid, size, arch = p.parent.name, p.parent.parent.name, p.parent.parent.parent.name
    root = p.parents[4]
    if oid not in _PARTS:
        V, F = load_mesh(root / "object_dataset" / size / oid / f"{oid}.obj")
        _PARTS[oid] = [Part(V * 1000.0, F)]
    d = np.load(p, allow_pickle=True).item()
    h = d["right_hand"]
    okeys = [k for k in d if k != "right_hand"]
    assert len(okeys) == 1, okeys
    o = d[okeys[0]]
    N = len(h["trans"])
    fr = np.arange(0, N, stride)
    T_wo = T_from(aa_to_R(o["rot"][fr]), np.asarray(o["trans"][fr], np.float64) * 1000.0)
    flat = FLAT.get(arch, False) if flat_override is None else flat_override
    m = mano("right", flat)
    V = m(np.zeros(10), np.asarray(h["rot"][fr], np.float64), np.asarray(h["pose"][fr], np.float64),
          np.asarray(h["trans"][fr], np.float64)) * 1000.0
    extra = {"archive": np.full(len(fr), arch), "size": np.full(len(fr), size), "object": np.full(len(fr), oid),
             "t_frac": fr / max(1, N - 1)}
    if "angle" in o:
        extra["obj_angle"] = np.asarray(o["angle"][fr], np.float64).ravel()
    return Sequence("graspxl", f"{arch}/{size}/{oid}/{p.stem}", FPS, stride, fr, T_wo, _PARTS[oid], None,
                    {"right": (V, np.isfinite(V).all((1, 2)))}, extra)
