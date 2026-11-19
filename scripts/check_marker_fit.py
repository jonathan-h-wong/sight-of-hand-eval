"""Marker fitting error of rectified ARCTIC GT: how far do the hands / object move away from the MoCap markers?

ARCTIC's GT was fitted to optical markers (mocap_npy: 14 per hand, 8 on the object; same world frame and frame
count as raw_seqs). This reports, per hand-frame, the mean unsigned distance from each hand marker to the hand
surface for the GT and for the rectified hand, and the same for the object markers against the (corrected)
object surface. The GT is the fit to these markers, so any change can only raise this error; the question is by how
much, relative to the GT's own ~2-3 mm.

    python scripts/check_marker_fit.py --rect out/rectify/arctic --root data/arctic \
        --mocap data/arctic_mocap/mocap_npy --mano-dir data/mano/mano_v1_2/models --out out/rectify/arctic/marker_fit.parquet
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hoiaudit.metrics import apply_T, inv_T  # noqa: E402
from hoiaudit.sources import arctic, common  # noqa: E402

HAND = ["IDX3", "IDX6", "MID0", "MID6", "PNK3", "PNK6", "RNG3", "RNG6", "THM3", "THM6", "IHAND", "OHAND", "IWR", "OWR"]


def dist(P, V, F):
    import igl
    return np.abs(igl.signed_distance(np.ascontiguousarray(P, np.float64), np.ascontiguousarray(V), F)[0])


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rect", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--mocap", type=Path, required=True)
    ap.add_argument("--mano-dir", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    common.MANO_DIR = Path(a.mano_dir)
    rows = []
    for npz in sorted((a.rect / "seqs").rglob("*.npz")):
        key = str(npz.relative_to(a.rect / "seqs"))[:-4]
        sid, name = key.split("/")
        seq = arctic.load(a.root / "raw_seqs" / sid / f"{name}.mano.npy")
        z = np.load(npz)
        mk = np.load(a.mocap / f"{sid}_{name}.npy", allow_pickle=True).item()
        lab = [str(x) for x in mk["subject"]["labels"]]
        T = len(seq.frame_idx)
        # object: markers into the object frame with the GT pose and with the corrected pose
        art = seq.parts
        Pm = mk["object"]["points"].astype(np.float64)                                  # (T,8,3) world mm
        T_ow0 = inv_T(seq.T_wo)
        if "obj_world_from_obj" in z.files:
            W = z["obj_world_from_obj"].copy()
            W[:, :3, 3] *= 1000.0
            T_ow1 = inv_T(W)
            moved = z["obj_moved"]
        else:
            T_ow1, moved = T_ow0, np.zeros(T, bool)
        od0, od1 = np.full(T, np.nan), np.full(T, np.nan)
        for t in range(T):
            if not np.isfinite(Pm[t]).all():
                continue
            Vt, Ft = art.mesh_at(seq.part_T[t])
            od0[t] = dist(apply_T(T_ow0[t:t + 1], Pm[t:t + 1])[0], Vt, Ft).mean()
            od1[t] = od0[t] if not moved[t] else dist(apply_T(T_ow1[t:t + 1], Pm[t:t + 1])[0], Vt, Ft).mean()
        for side, c in (("left", "L"), ("right", "R")):
            idx = [lab.index(c + h) for h in HAND]
            P = mk["subject"]["points"][:, idx].astype(np.float64)
            V0, valid = seq.hands[side]
            m = common.mano(side, False)
            frozen, unres = z[f"{side}_frozen"], z[f"{side}_unresolved"]
            rect = valid & ~frozen & ~unres
            ri = np.where(rect)[0]
            V1 = V0.copy()
            V1[ri] = m(z[f"{side}_betas"], z[f"{side}_rot"][ri], z[f"{side}_pose"][ri], z[f"{side}_trans"][ri]) * 1000.0
            for t in np.where(valid)[0]:
                if not np.isfinite(P[t]).all():
                    continue
                d0 = dist(P[t], V0[t], m.faces)
                d1 = d0 if not rect[t] else dist(P[t], V1[t], m.faces)
                rows.append({"seq": key, "hand": side, "frame": int(t),
                             "status": "rectified" if rect[t] else ("unresolved" if unres[t] else "frozen"),
                             "pen_before_mm": float(z[f"{side}_pen_before_mm"][t]),
                             "marker_err_gt_mm": float(d0.mean()), "marker_err_rect_mm": float(d1.mean()),
                             "marker_err_max_gt_mm": float(d0.max()), "marker_err_max_rect_mm": float(d1.max()),
                             "obj_moved": bool(moved[t]), "obj_marker_err_gt_mm": od0[t], "obj_marker_err_rect_mm": od1[t]})
        print(key, flush=True)
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out)
    r = df[df.status == "rectified"]
    q = lambda x: [round(float(v), 2) for v in np.nanpercentile(x, [50, 95])]  # noqa: E731
    s = {"hand_frames": len(df), "rectified": len(r),
         "hand_marker_err_mm_p50_p95": {"gt_all_frames": q(df.marker_err_gt_mm), "gt_rectified_frames": q(r.marker_err_gt_mm),
                                        "rect_rectified_frames": q(r.marker_err_rect_mm)},
         "hand_marker_err_increase_mm_p50_p95": q(r.marker_err_rect_mm - r.marker_err_gt_mm),
         "worst_marker_increase_mm_p50_p95": q(r.marker_err_max_rect_mm - r.marker_err_max_gt_mm),
         "object_moved_frames": int(df.drop_duplicates(["seq", "frame"]).obj_moved.sum())}
    o = df.drop_duplicates(["seq", "frame"])
    o = o[o.obj_moved]
    if len(o):
        s["obj_marker_err_mm_p50_p95"] = {"gt": q(o.obj_marker_err_gt_mm), "rect": q(o.obj_marker_err_rect_mm)}
    a.out.with_suffix(".json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
