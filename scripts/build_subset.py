"""Build a physically grounded subset index from an audit and (optionally) publish it.

    python scripts/build_subset.py --dataset dexycb                                  # original labels, 1 mm, 2 mm gap
    python scripts/build_subset.py --dataset arctic --labels rectified --pen-mm 2
    python scripts/build_subset.py --dataset ho3d --audit out/ho3d/frames.parquet --publish
    python scripts/build_subset.py --all --publish                                    # every audited dataset, the
                                                                                      # standard criteria set below
Writes OUT/<dataset>/grounded_v1/<tag>/{index.parquet, summary.json} (OUT defaults to out/) and with --publish syncs
it to s3://xray-hands/eval/<dataset>/grounded_v1/<tag>/. See hoiaudit/subset.py for the criteria and index columns,
and scripts/fetch_grounded.py to materialize the frames.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hoiaudit import subset  # noqa: E402

# the plan's frozen sweep: headline 1 mm on original labels, 2 mm as secondary, rectified 2 mm where available
STANDARD = [dict(pen_mm=1.0), dict(pen_mm=2.0), dict(pen_mm=2.0, labels="rectified")]


def build(ds, c, audit, out, publish):
    idx, summ = subset.build_index(ds, c, audit)
    d = out / ds / "grounded_v1" / c.tag
    d.mkdir(parents=True, exist_ok=True)
    for col in idx.columns:
        if idx[col].dtype == object:
            idx[col] = idx[col].astype("category")
    idx.to_parquet(d / "index.parquet")
    (d / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"{ds} {c.tag}: {summ['grounded_frames']} grounded hand-frames "
          f"({summ['grounded_frac_of_contact']:.1%} of contact), {summ['sequences']} seqs, {summ['runs']} runs", flush=True)
    if publish:
        subprocess.run(["aws", "s3", "sync", "--quiet", str(d), f"s3://{subset.BUCKET}/eval/{ds}/grounded_v1/{c.tag}/"],
                       check=True)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(subset.AUDIT))
    ap.add_argument("--all", action="store_true", help="every dataset whose audit exists, STANDARD criteria")
    ap.add_argument("--audit", type=Path, help="local audit frames.parquet (default: the published one on S3)")
    ap.add_argument("--labels", choices=["original", "rectified"], default="original")
    ap.add_argument("--pen-mm", type=float, default=1.0)
    ap.add_argument("--gap-mm", type=float, default=2.0)
    ap.add_argument("--contact-mm", type=float, default=5.0)
    ap.add_argument("--jitter-mm", type=float)
    ap.add_argument("--min-run", type=int, default=1)
    ap.add_argument("--include-free", action="store_true", help="also keep non-contact frames")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    if a.all:
        import boto3
        s3 = boto3.client("s3")
        for ds, key in subset.AUDIT.items():
            try:
                s3.head_object(Bucket=subset.BUCKET, Key=key)
            except Exception:
                print(f"{ds}: no audit at {key}, skipped")
                continue
            for kw in STANDARD:
                if kw.get("labels") == "rectified" and ds not in subset.RECTIFIED:
                    continue
                build(ds, subset.Criteria(**kw), None, a.out, a.publish)
        return
    c = subset.Criteria(pen_mm=a.pen_mm, gap_mm=a.gap_mm, contact_mm=a.contact_mm, jitter_mm=a.jitter_mm,
                        min_run=a.min_run, labels=a.labels, include_free=a.include_free)
    build(a.dataset, c, a.audit, a.out, a.publish)


if __name__ == "__main__":
    main()
