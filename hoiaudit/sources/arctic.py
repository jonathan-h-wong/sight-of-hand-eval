"""ARCTIC (Fan et al. 2023). Bimanual manipulation of articulated objects, 30 fps, MoCap-fitted MANO.

Local layout (unzip raw/arctic/data/raw_seqs.zip and meta.zip into ROOT):
    ROOT/raw_seqs/sXX/<obj>_<action>_<n>.mano.npy     dict left/right: rot, pose(45), trans (m), shape, fitting_err
    ROOT/raw_seqs/sXX/<obj>_<action>_<n>.object.npy   (T, 7): hinge angle (rad), axis-angle rot, trans (mm)
    ROOT/meta/object_vtemplates/<obj>/mesh.obj, parts.json   (mm; parts.json = top-part vertex flag)

Conventions (= xray-hands analysis/run_penetration.py, the ARCTIC benchmark): smplx MANO, flat_hand_mean=False,
subject `shape`, stock MANO_LEFT. The object is articulated: the top part rotates by the hinge angle.
Mask key: sXX/<seq>. Same keys and rows as xray-hands results/arctic_frames.parquet and grounded_mask/.
"""
import json
from pathlib import Path

import numpy as np

from ..metrics import Articulated, Sequence
from .common import T_from, aa_to_R, load_mesh, mano

_PARTS = {}


def list_tasks(root):
    return sorted(str(p) for p in (Path(root) / "raw_seqs").glob("s*/*.mano.npy"))


def _parts(root, obj):
    if obj not in _PARTS:
        d = Path(root) / "meta" / "object_vtemplates" / obj
        V, F = load_mesh(d / "mesh.obj")
        top = np.array(json.load(open(d / "parts.json")), bool)
        _PARTS[obj] = Articulated(V, F, top)
    return _PARTS[obj]


def load(task, stride=1):
    p = Path(task)
    root = p.parents[2]
    sid, seq = p.parent.name, p.name[: -len(".mano.npy")]
    obj = seq.split("_")[0]
    m = np.load(p, allow_pickle=True).item()
    o = np.load(str(p).replace(".mano.npy", ".object.npy"), allow_pickle=True)
    fr = np.arange(0, len(o), stride)
    o = o[fr]
    T_wo = T_from(aa_to_R(o[:, 1:4]), o[:, 4:7].astype(np.float64))
    hands, extra = {}, {}
    for side in ("left", "right"):
        h = m[side]
        V = mano(side, False)(h["shape"], h["rot"][fr], h["pose"][fr], h["trans"][fr]) * 1000.0
        hands[side] = (V, np.isfinite(V).all((1, 2)))
        extra[f"{side}:fit_err"] = np.asarray(h["fitting_err"])[fr]
    extra["obj_angle_rad"] = o[:, 0]
    extra["object"] = np.full(len(fr), obj)
    return Sequence("arctic", f"{sid}/{seq}", 30.0, stride, fr, T_wo, _parts(root, obj), o[:, 0].astype(np.float64),
                    hands, extra)
