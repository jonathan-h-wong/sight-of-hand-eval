"""Central hand-object penetration audit. One script, any dataset with a source in hoiaudit/sources.

For every (sequence, hand, frame) it computes signed-distance penetration of the (1x subdivided) MANO surface into
the object(s) plus temporal smoothness (see hoiaudit/metrics.py), and writes:

    OUT/frames.parquet   all metrics, one row per evaluated (seq, hand, frame)
    OUT/mask.parquet     slim mask: dataset, seq, hand, frame, pen_max_mm, min_sd_mm, contact, grounded
                         grounded = pen_max_mm < --thresh (strict). Frames without valid GT hands are absent.
    OUT/summary.json     parameters, counts, percentiles, error list

The raw data is never modified; `seq` + `frame` index straight into the raw files (frame = row index in the raw
arrays). Re-threshold any time from pen_max_mm without rerunning.

    python scripts/audit.py --dataset arctic  --root data/arctic           --out out/arctic
    python scripts/audit.py --dataset grab    --root data/grab/unz          --out out/grab
    python scripts/audit.py --dataset dexycb  --root data/dexycb            --out out/dexycb
    python scripts/audit.py --dataset graspxl --root data/graspxl           --out out/graspxl
    python scripts/audit.py --dataset ho3d    --root data/ho3d              --out out/ho3d
    python scripts/audit.py --dataset hoi4d   --root data/hoi4d             --out out/hoi4d
    python scripts/audit.py --dataset oakink  --root data/oakink            --out out/oakink
"""
import argparse
import importlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hoiaudit import metrics  # noqa: E402
from hoiaudit.sources import common  # noqa: E402

DATASETS = ("arctic", "grab", "dexycb", "graspxl", "ho3d", "hoi4d", "oakink")
_G = {}


def _init(mano_dir, ds):
    common.MANO_DIR = Path(mano_dir)
    _G["src"] = importlib.import_module(f"hoiaudit.sources.{ds}")
    _G["hs"] = metrics.HandSurface(common.mano("right", False).faces)


def _work(args):
    task, stride = args
    try:
        seq = _G["src"].load(task, stride=stride)
        return metrics.evaluate(seq, _G["hs"]), None
    except Exception as e:  # keep going; report at the end
        import traceback
        return None, f"{task}: {e!r}\n{traceback.format_exc(limit=4)}"


def _work_tagged(args):
    return args[0], *_work(args)


def main():
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mano-dir", default=os.environ.get("MANO_DIR", "data/mano/mano_v1_2/models"))
    ap.add_argument("--thresh", type=float, default=2.0, help="grounded if max penetration < thresh mm")
    ap.add_argument("--stride", type=int, default=1, help="evaluate every Nth raw frame (1 = all)")
    ap.add_argument("--limit", type=int, default=0, help="only N sequences, spread over the list (0 = all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    a = ap.parse_args()

    src = importlib.import_module(f"hoiaudit.sources.{a.dataset}")
    tasks = src.list_tasks(a.root)
    if a.limit:
        tasks = [tasks[i] for i in np.linspace(0, len(tasks) - 1, min(a.limit, len(tasks))).astype(int)]
    # biggest first so the pool does not end on one long straggler (size proxy: the task file, else 0)
    def _size(t):
        for p in (t, os.path.join(t, "pose.npz")):
            if os.path.isfile(p):
                return os.path.getsize(p)
        return 0
    tasks.sort(key=lambda t: -_size(t))
    print(f"{a.dataset}: {len(tasks)} sequences, {a.workers} workers, stride {a.stride}", flush=True)

    # checkpointing: finished sequences are flushed to OUT/parts/*.parquet; a rerun skips tasks already there
    parts = a.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    done_file = parts / "done.txt"  # appended only after the part holding a task's rows is written
    done = set(done_file.read_text().split("\n")) if done_file.exists() else set()
    todo = [t for t in tasks if t not in done]
    print(f"  resuming: {len(done)} done, {len(todo)} to go", flush=True)
    t0 = time.time()
    buf, buf_tasks, errs, n_part = [], [], [], len(list(parts.glob("*.parquet")))

    def flush():
        nonlocal buf, buf_tasks, n_part
        if buf:
            pd.concat(buf, ignore_index=True).to_parquet(parts / f"part_{n_part:05d}.parquet")
            n_part += 1
        with open(done_file, "a") as f:
            f.writelines(t + "\n" for t in buf_tasks)
        buf, buf_tasks = [], []

    with Pool(a.workers, initializer=_init, initargs=(a.mano_dir, a.dataset), maxtasksperchild=200) as pool:
        for i, (task, df, err) in enumerate(pool.imap_unordered(_work_tagged, [(t, a.stride) for t in todo],
                                                                  chunksize=1)):
            if df is not None:
                buf.append(df)
            if err:
                errs.append(err)  # errored tasks are not marked done, so a rerun retries them
            else:
                buf_tasks.append(task)
            if len(buf_tasks) >= 25:
                flush()
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)}  {time.time() - t0:.0f}s  errors={len(errs)}", flush=True)
    flush()

    out = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("*.parquet"))], ignore_index=True)

    for c in out.columns:  # object columns -> category keeps the parquet small
        if out[c].dtype == object:
            out[c] = out[c].astype("category")
    out.to_parquet(a.out / "frames.parquet")
    mask = out[["dataset", "seq", "hand", "frame", "pen_max_mm", "min_sd_mm", "contact"]].copy()
    mask["grounded"] = mask.pen_max_mm < a.thresh
    mask.to_parquet(a.out / "mask.parquet")

    def pct(x):
        return {f"p{q}": round(float(np.percentile(x, q)), 3) for q in (50, 90, 95, 99)} | {"max": round(float(x.max()), 3)}

    con = out[out.contact]
    summary = {
        "dataset": a.dataset, "root": str(a.root), "thresh_mm": a.thresh, "stride": a.stride,
        "contact_mm": metrics.CONTACT_MM, "sequences": int(out.seq.nunique()), "hand_frames": len(out),
        "grounded_frac": round(float(mask.grounded.mean()), 4),
        "contact_frac": round(float(out.contact.mean()), 4),
        "grounded_frac_in_contact": round(float((con.pen_max_mm < a.thresh).mean()), 4) if len(con) else None,
        "pen_max_mm_all": pct(out.pen_max_mm), "pen_max_mm_in_contact": pct(con.pen_max_mm) if len(con) else None,
        "by_hand": {h: {"hand_frames": len(g), "grounded_frac": round(float((g.pen_max_mm < a.thresh).mean()), 4)}
                    for h, g in out.groupby("hand", observed=True)},
        "seq_all_grounded_frac": round(float(mask.groupby("seq", observed=True).grounded.all().mean()), 4),
        "errors": errs, "seconds": round(time.time() - t0),
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, indent=1))
    for e in errs[:10]:
        print("ERROR", e)


if __name__ == "__main__":
    main()
