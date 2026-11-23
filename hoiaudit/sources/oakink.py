"""OakInk-Image v2 (Yang et al. 2022). Multi-view (4 cameras) video of one or two right hands with one object, 30 fps.

Needs the per-frame annotations `image/anno_v2_1.zip`, which are NOT in the Hugging Face release
(raw/oakink_v1/); they are granted through the Google Form linked from oakink.net (https://forms.gle/g6QEmmCeZYLGaVe29).
Upload the zip to raw/oakink_v1/image/anno_v2_1.zip and `scripts/prepare_data.sh DATA oakink` picks it up.

Local layout (OakInk docs/datasets.md):
    ROOT/image/anno/seq_all.json                          [[seq_id/ts, sbj_flag, frame, cam_id], ...]
    ROOT/image/anno/hand_v/<seq_id>__<ts>__<sbj>__<frame>__<cam>.pkl     (778,3) MANO verts, camera space, metres
    ROOT/image/anno/obj_transf/<same>.pkl                 (4,4) T_c_o, object canonical -> camera
    ROOT/image/obj/<obj_id>.{obj,ply}                     object models (obj_id = seq_id.split('_')[0])

Conventions (oakink/OakInk oikit/oi_image/oi_image.py): the hand is used as the released 778 vertices, no MANO
forward pass needed. The object is world = T_c_o v. Geometry is identical in every view, so each frame is evaluated
once, in the lowest-numbered camera that has it (the same camera for both people). sbj_flag 0/1 distinguishes the two people of handover sequences.
Mesh units are detected (an extent above 5 means millimetres) and reported in the `obj_unit_mm` column.
Mask key: <seq_id>/<ts>; hand = sbj<flag>; frame = the OakInk frame index.
"""
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..metrics import Part, Sequence
from .common import load_mesh

FPS = 30.0
_PARTS = {}
_INDEX = {}


def _index(root):
    root = str(root)
    if root not in _INDEX:
        rows = json.loads((Path(root) / "image" / "anno" / "seq_all.json").read_text())
        idx = defaultdict(lambda: defaultdict(set))  # seq -> frame -> {(sbj, cam)}
        for seq, sbj, frame, cam in rows:
            idx[seq][int(frame)].add((int(sbj), int(cam)))
        _INDEX[root] = idx
    return _INDEX[root]


def list_tasks(root):
    return sorted(str(Path(root) / "image" / "anno" / "seqs" / s) for s in _index(root))


def _part(root, oid):
    if oid not in _PARTS:
        for ext in ("obj", "ply"):
            p = Path(root) / "image" / "obj" / f"{oid}.{ext}"
            if p.exists():
                V, F = load_mesh(p)
                mm = float(np.ptp(V, 0).max()) > 5.0
                _PARTS[oid] = (Part(V if mm else V * 1000.0, F), mm)
                break
        else:
            raise FileNotFoundError(oid)
    return _PARTS[oid]


def _pkl(p):
    with open(p, "rb") as f:
        return np.asarray(pickle.load(f), np.float64)


def load(task, stride=1):
    task = Path(task)
    root = task.parents[4]
    seq = f"{task.parent.name}/{task.name}"
    oid = task.parent.name.split("_")[0]
    part, mm = _part(root, oid)
    anno = root / "image" / "anno"
    by_frame = _index(root)[seq]
    frames = sorted(by_frame)[::stride]
    fr = np.array(frames, np.int64)
    n = len(fr)
    seq_id, ts = seq.split("/")
    T_wo = np.full((n, 4, 4), np.nan)
    hands = {}
    # one camera per frame for every person, so hands and object share a frame
    cam = {f: min(c for _, c in by_frame[f]) for f in fr}
    for sbj in sorted({s for f in fr for s, _ in by_frame[f]}):
        V = np.full((n, 778, 3), np.nan)
        valid = np.zeros(n, bool)
        for i, f in enumerate(fr):
            if (sbj, cam[f]) not in by_frame[f]:
                continue
            key = f"{seq_id}__{ts}__{sbj}__{f}__{cam[f]}.pkl"
            V[i] = _pkl(anno / "hand_v" / key) * 1000.0
            if not np.isfinite(T_wo[i]).all():
                T = _pkl(anno / "obj_transf" / key)
                T[:3, 3] *= 1000.0
                T_wo[i] = T
            valid[i] = True
        hands[f"sbj{sbj}"] = (V, valid)
    extra = {"object": np.full(n, oid), "obj_unit_mm": np.full(n, mm)}
    return Sequence("oakink", seq, FPS, stride, fr, T_wo, [part], None, hands, extra)
