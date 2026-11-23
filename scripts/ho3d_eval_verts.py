"""Split HO3D v3's released evaluation GT (HO3D_V3_GT/evaluation_verts.json, one 778x3 list per line of
evaluation.txt, same camera frame as the meta, metres) into ROOT/evaluation_verts/<seq>.npz {frames, verts} for hoiaudit/sources/ho3d.py.

    python scripts/ho3d_eval_verts.py --root data/ho3d --gt data/ho3d_gt/evaluation_verts.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="dir holding evaluation.txt")
    ap.add_argument("--gt", type=Path, required=True)
    a = ap.parse_args()
    names = [l.strip() for l in (a.root / "evaluation.txt").read_text().splitlines() if l.strip()]
    verts = json.loads(a.gt.read_text())
    assert len(verts) == len(names), (len(verts), len(names))
    by = defaultdict(list)
    for name, v in zip(names, verts):
        seq, frame = name.split("/")
        by[seq].append((int(frame), v))
    out = a.root / "evaluation_verts"
    out.mkdir(parents=True, exist_ok=True)
    for seq, rows in by.items():
        rows.sort()
        np.savez(out / f"{seq}.npz", frames=np.array([r[0] for r in rows]),
                 verts=np.array([r[1] for r in rows], np.float32))
    print(f"{len(names)} frames, {len(by)} sequences -> {out}")


if __name__ == "__main__":
    main()
