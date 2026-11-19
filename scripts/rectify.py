"""Rectify GT hand poses: remove hand-object penetration > 2 mm, keep frames <= 2 mm bit-identical.

    python scripts/rectify.py --dataset arctic --root data/arctic --out out/rectify/arctic --device cuda
    python scripts/rectify.py --dataset dexycb --root data/dexycb --out out/rectify/dexycb --device cuda
    # several processes on one or more GPUs:   --shard 0/4 ... --shard 3/4, then once:  --summarize

Writes, per sequence (resumable: finished sequences are skipped):
    OUT/seqs/<seq>.npz       per hand: <side>_rot (T,3), <side>_pose (T,45 axis-angle, flat_hand_mean=False),
                             <side>_trans (T,3) m, <side>_betas, <side>_valid, <side>_frozen (GT <= 2 mm, untouched),
                             <side>_unresolved (GT > 2 mm that could not be fixed locally, untouched),
                             <side>_pen_before_mm, <side>_pen_after_mm; meta (json)
                             obj_world_from_obj (T,4,4, metres): corrected pose of the corrected object (ARCTIC:
                             the object base; DexYCB: the grasped object), obj_D (T,4,4) the correction in the
                             object frame (mm), obj_moved / obj_blocked (T,) where it moved / was held for a
                             grounded hand in contact
    OUT/frames/<seq>.parquet per hand-frame report (optimize.report)
and with --summarize: OUT/frames.parquet (all sequences) + OUT/summary.json.

Row t of every array is raw frame t of the sequence (same keys as the audit's mask.parquet). DexYCB poses are
written as the full 45-d axis-angle (its PCA-45 basis converts exactly), so both datasets share one convention:
MANO(flat_hand_mean=False)(betas, rot, pose) + trans, in the dataset's world frame (ARCTIC world / DexYCB master
camera), metres.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hoiaudit import metrics  # noqa: E402
from hoiaudit.rectify import optimize, scene  # noqa: E402
from hoiaudit.sources import common  # noqa: E402


def seq_key(dataset, task):
    """Sequence key without loading the sequence (same keys as the sources / audit mask)."""
    p = Path(str(task))
    if dataset == "arctic":
        return f"{p.parent.name}/{p.name[: -len('.mano.npy')]}"
    return f"{p.parent.name}/{p.name}"          # dexycb: <date>-subject-NN/<seq>


def summarize(out):
    import pandas as pd
    parts = sorted((out / "frames").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet(out / "frames.parquet")
    a = df[~df.frozen & ~df.unresolved]          # rectified frames

    def q(x):
        return {f"p{k}": round(float(np.nanpercentile(x, k)), 3) for k in (50, 95, 99)} | {"max": round(float(np.nanmax(x)), 3)}

    s = {"sequences": int(df.seq.nunique()), "hand_frames": len(df), "frozen_frac": round(float(df.frozen.mean()), 4),
         "rectified_frames": len(a), "unresolved_frames": int(df.unresolved.sum()),
         "unresolved_frac_of_penetrating": round(float(df.unresolved.sum() / max(1, (~df.frozen).sum())), 4),
         "freeze_mm": optimize.FREEZE_MM,
         "pen_before_mm_rectified": q(a.pen_before_mm) if len(a) else None,
         "pen_after_mm_rectified": q(a.pen_after_mm) if len(a) else None,
         "joint_shift_mean_mm_rectified": q(a.dj_mean_mm) if len(a) else None,
         "joint_shift_max_mm_rectified": q(a.dj_max_mm) if len(a) else None,
         "px_max_rectified": q(a.px_max) if len(a) else None,
         "contact_kept_frac_rectified": round(float(((a.min_sd_after_mm < 5) == (a.min_sd_before_mm < 5)).mean()), 4),
         "rel_hf_mm_all": {"before": q(df.rel_hf_before_mm.dropna()), "after": q(df.rel_hf_after_mm.dropna())},
         "weights": optimize.W, "target_mm": optimize.TARGET_MM}
    errs = sorted((out / "errors").glob("*.txt")) if (out / "errors").exists() else []
    s["errors"] = [e.read_text()[:2000] for e in errs]
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: v for k, v in s.items() if k != "errors"}, indent=1), f"\nerrors: {len(errs)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(scene.ADAPTERS), required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mano-dir", default=os.environ.get("MANO_DIR", "data/mano/mano_v1_2/models"))
    ap.add_argument("--sdf-cache", type=Path, default=None, help="default: OUT/sdf_cache")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1", help="i/n: this process handles every n-th sequence")
    ap.add_argument("--limit", type=int, default=0, help="only N sequences, spread over the list")
    ap.add_argument("--seq", action="append", default=[], help="only these sequence keys (repeatable)")
    ap.add_argument("--summarize", action="store_true", help="only merge finished sequences into frames/summary")
    ap.add_argument("--no-object", action="store_true", help="hands only: never correct the object pose")
    ap.add_argument("--weights", default="{}", help='JSON overrides of optimize.W, e.g. \'{"ocam": 2}\'')
    a = ap.parse_args()
    if a.summarize:
        return summarize(a.out)

    import importlib
    import torch
    common.MANO_DIR = Path(a.mano_dir)
    optimize.W.update(json.loads(a.weights))
    src = importlib.import_module(f"hoiaudit.sources.{a.dataset}")
    tasks = src.list_tasks(a.root)
    if a.limit:
        tasks = [tasks[i] for i in np.linspace(0, len(tasks) - 1, min(a.limit, len(tasks))).astype(int)]
    i, n = map(int, a.shard.split("/"))
    tasks = tasks[i::n]
    cache = a.sdf_cache or a.out / "sdf_cache"
    for d in ("seqs", "frames", "errors"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for j, task in enumerate(tasks):
        try:
            key = seq_key(a.dataset, task)
            npz = a.out / "seqs" / f"{key}.npz"
            if (a.seq and key not in a.seq) or npz.exists():
                continue
            sc = scene.ADAPTERS[a.dataset](task, cache)
            assert sc.name == key, (sc.name, key)
            res = optimize.rectify(sc, device=a.device, log=lambda s: print("   ", s, flush=True),
                                   move_object=not a.no_object)
            df = optimize.report(sc, res)
            arrays = {}
            for side in sc.hands:
                r = res[side]
                arrays |= {f"{side}_rot": r["params"]["rot"], f"{side}_pose": r["params"]["pose"],
                           f"{side}_trans": r["params"]["trans"], f"{side}_betas": r["betas"],
                           f"{side}_valid": r["valid"], f"{side}_frozen": r["frozen"],
                           f"{side}_unresolved": r["unresolved"],
                           f"{side}_pen_before_mm": r["pen_before_mm"], f"{side}_pen_after_mm": r["pen_after_mm"]}
            if sc.obj is not None:
                o = sc.obj
                Binv = metrics.inv_T(np.where(o["tracked"][:, None, None], o["base"], np.eye(4)))
                Wfo = Binv @ res["_obj"]["D"]                    # world <- corrected object (mm)
                Wfo[:, :3, 3] /= 1000.0
                arrays |= {"obj_D": res["_obj"]["D"], "obj_moved": res["_obj"]["movable"],
                           "obj_blocked": res["_obj"]["blocked"], "obj_world_from_obj": Wfo}
            meta = {"dataset": a.dataset, "object_corrected": not a.no_object, "seq": sc.name, "source": str(task), "freeze_mm": optimize.FREEZE_MM,
                    "target_mm": optimize.TARGET_MM, "weights": optimize.W, "iters": optimize.ITERS,
                    "rounds": optimize.ROUNDS, "convention": "MANO flat_hand_mean=False, pose 45-d axis-angle, metres",
                    "seconds": round(res["_seconds"], 1), "torch": torch.__version__}
            npz.parent.mkdir(parents=True, exist_ok=True)
            tmp = npz.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
            fp = a.out / "frames" / f"{sc.name}.parquet"
            fp.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(fp)
            tmp.rename(npz)                 # the npz marks the sequence done, so it is written last
            act = df[~df.frozen]
            print(f"[{j + 1}/{len(tasks)}] {sc.name}: {res['_seconds']:.0f}s, active {len(act)}/{len(df)}, "
                  f"still > {optimize.FREEZE_MM} mm: {int((act.pen_after_mm > optimize.FREEZE_MM).sum())}, "
                  f"elapsed {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # keep going; errors are listed by --summarize
            key = Path(str(task)).name
            (a.out / "errors" / f"{key}.txt").write_text(f"{task}: {e!r}\n{traceback.format_exc()}")
            print("ERROR", task, repr(e), flush=True)


if __name__ == "__main__":
    main()
