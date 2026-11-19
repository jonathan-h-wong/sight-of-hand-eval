"""Fetch a stratified sample of GraspXL MANO sequences (+ their object meshes) from S3 via ranged reads.

No full download: reads each zip's central directory, samples objects per size bucket, and pulls only those entries.
mano_dataset_N are split archives (.zip.part00 + .zip.part01) and are read as one. If a part is missing, only entries
in the parts present are sampled (front_missing_bytes in the manifest).

Local layout written (mirrors archive paths):
    OUT/recorded/<archive>/<size>/<obj_id>/<file>.npy
    OUT/object_dataset/<size>/<obj_id>/<obj_id>.obj
    OUT/sample_manifest.json

    python scripts/fetch_graspxl_sample.py --out data/graspxl --objects-per-archive 600 --seed 0
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hoiaudit import s3zip  # noqa: E402

BUCKET = "xray-hands"
ARCHIVES = {  # name -> key or [part keys]
    "mano_tabletop": "raw/graspxl/mano_tabletop.zip",
    # released as split archives; read as the concatenation of their parts
    **{f"mano_dataset_{n}": [f"raw/graspxl/mano_dataset_{n}.zip.part00", f"raw/graspxl/mano_dataset_{n}.zip.part01"]
       for n in (1, 2, 3)},
}
OBJECTS = "raw/graspxl/object_dataset.zip"


def obj_key(name, archive):
    """entry name -> (size, obj_id)."""
    p = name.split("/")
    if archive != "mano_tabletop":
        p = p[1:]
    return p[0], p[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--objects-per-archive", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    print("reading object_dataset central directory", flush=True)
    oz = s3zip.central_dir(BUCKET, OBJECTS)
    meshes = {}
    for e in oz["entries"]:
        if e["name"].endswith(".obj"):
            _, size, oid, _ = e["name"].split("/")
            meshes[(size, oid)] = e

    manifest = {"seed": a.seed, "objects_per_archive": a.objects_per_archive, "archives": {}}
    jobs = []
    for arch, key in ARCHIVES.items():
        print("reading", arch, flush=True)
        z = s3zip.central_dir(BUCKET, key)
        by_obj = defaultdict(list)
        n_all, n_present = 0, 0
        for e in z["entries"]:
            if not e["name"].endswith(".npy"):
                continue
            n_all += 1
            if e["offset"] - z["shift"] < 0:
                continue  # lives in the missing front part of a truncated archive
            n_present += 1
            by_obj[obj_key(e["name"], arch)].append(e)
        by_size = defaultdict(list)
        for k in by_obj:
            if k in meshes:
                by_size[k[0]].append(k)
        # stratify proportionally to the number of objects per size bucket
        tot = sum(len(v) for v in by_size.values())
        picked = []
        for size, ks in sorted(by_size.items()):
            n = max(1, round(a.objects_per_archive * len(ks) / tot))
            picked += rnd.sample(sorted(ks), min(n, len(ks)))
        seqs = [e for k in picked for e in by_obj[k]]
        manifest["archives"][arch] = {
            "key": key, "front_missing_bytes": z["shift"], "npy_entries_total": n_all,
            "npy_entries_present": n_present, "objects_present": len(by_obj),
            "objects_sampled": len(picked), "sequences_sampled": len(seqs),
            "objects_by_size_present": {s: len(v) for s, v in by_size.items()},
        }
        print(f"  {arch}: {n_present}/{n_all} sequences present, sampled {len(picked)} objects / {len(seqs)} sequences",
              flush=True)
        for e in seqs:
            jobs.append((key, z["shift"], e, a.out / "recorded" / arch / Path(*obj_key(e["name"], arch)) / Path(e["name"]).name))
        for k in picked:
            jobs.append((OBJECTS, 0, meshes[k], a.out / "object_dataset" / k[0] / k[1] / f"{k[1]}.obj"))

    def get(job):
        key, shift, e, dst = job
        if dst.exists() and dst.stat().st_size == e["usize"]:
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = s3zip.read_entry(BUCKET, key, e, shift)
        assert len(data) == e["usize"], (e["name"], len(data), e["usize"])
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.rename(dst)

    print(f"fetching {len(jobs)} files", flush=True)
    with ThreadPoolExecutor(a.threads) as ex:
        for i, _ in enumerate(ex.map(get, jobs)):
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "sample_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("done")


if __name__ == "__main__":
    main()
