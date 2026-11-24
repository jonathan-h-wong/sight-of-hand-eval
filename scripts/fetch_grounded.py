"""Fetch a physically grounded subset: labels (hand meshes, object poses and meshes) and, optionally, images.

Reads a subset index (scripts/build_subset.py; published at s3://xray-hands/eval/<ds>/grounded_v1/<tag>/index.parquet)
and materializes only the grounded frames:

    OUT/labels/<seq>.npz       frames (N,), T_wo (N,4,4) world <- object base, mm; part_T (N,K,4,4) or angles (N,)
                               for articulated objects; per hand: <hand>_verts (N,778,3) mm (dataset world/camera
                               frame), <hand>_grounded (N,) bool = the row is in the subset; mesh ids (meta)
                               with --labels rectified also <hand>_verts_rectified and the object correction
    OUT/meshes/<id>.npz        V (mm, part-local), F [, top for articulated ARCTIC objects]
    OUT/images/<member path>   images of grounded frames (--images), named as inside the raw archive
    OUT/manifest.parquet       the index rows + label_file, image_file columns

Label materialization runs the same source loader as the audit, so it needs the dataset's labels prepared locally
(`bash scripts/prepare_data.sh DATA <dataset>`, then --label-root DATA/<dataset dir>).

    python scripts/fetch_grounded.py --dataset dexycb --tag original_pen1_gap2 --label-root data/dexycb \\
        --out grounded/dexycb --images all
    python scripts/fetch_grounded.py --dataset arctic --tag rectified_pen2_gap2 --label-root data/arctic \\
        --out grounded/arctic --images cams --cams 0     # egocentric view only
"""
import argparse
import hashlib
import io
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hoiaudit import metrics, s3zip, subset  # noqa: E402
from hoiaudit.sources import common  # noqa: E402

BUCKET = subset.BUCKET
AXES = {  # frame of the materialized hands/objects, so a loader can project them onto the images
    "ho3d": "camera frame, OpenGL axes (z < 0 in front): X_opencv = X * (1, -1, -1), then project with meta camMat",
    "hoi4d": "camera frame, OpenCV axes: project with camera_params/<cam>/intrin.npy",
    "dexycb": "master camera frame, OpenCV axes: other cameras via calibration extrinsics",
    "arctic": "ARCTIC world frame: project with meta/misc.json world2cam + intris_mat per subject and view",
    "oakink": "camera frame of the evaluated camera (lowest cam id with the frame), OpenCV axes",
}


def task_path(dataset, root, seq):
    root = Path(root)
    return str({
        "arctic": root / "raw_seqs" / f"{seq}.mano.npy",
        "dexycb": root / seq,
        "grab": root / f"{seq}.npz",
        "graspxl": root / "recorded" / f"{seq}.npy",
        "ho3d": root / seq,
        "hoi4d": root / "seqs" / seq,
        "oakink": root / "image" / "anno" / "seqs" / seq,
    }[dataset])


def save_mesh(out, V, F, top=None):
    mid = hashlib.sha1(np.ascontiguousarray(V, np.float32).tobytes()).hexdigest()[:16]
    p = out / "meshes" / f"{mid}.npz"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, V=V.astype(np.float32), F=F.astype(np.int32), **({} if top is None else {"top": top}))
    return mid


def rectified_verts(key, hand, frames):
    """Rectified MANO params (scripts/rectify.py: flat_hand_mean=False, 45-d axis-angle, metres) -> verts mm."""
    body = __import__("boto3").client("s3").get_object(Bucket=BUCKET, Key=key)["Body"].read()
    z = np.load(io.BytesIO(body))
    m = common.mano(hand, False)
    V = m(z[f"{hand}_betas"], z[f"{hand}_rot"][frames], z[f"{hand}_pose"][frames], z[f"{hand}_trans"][frames]) * 1000.0
    extra = {k: z[k][frames] for k in ("obj_world_from_obj", "obj_moved") if k in z}
    return V, extra


def fetch_labels(a, idx, out):
    import importlib
    src = importlib.import_module(f"hoiaudit.sources.{a.dataset}")
    hs_ok = []
    files = {}
    for seq, g in idx.groupby("seq", observed=True, sort=False):
        seq = str(seq)
        dst = out / "labels" / f"{seq}.npz"
        files[seq] = str(dst.relative_to(out))
        if dst.exists():
            continue
        S = src.load(task_path(a.dataset, a.label_root, seq))
        keep = np.isin(S.frame_idx, g.frame.to_numpy())
        arrays = {"frames": S.frame_idx[keep], "T_wo": S.T_wo[keep]}
        if isinstance(S.parts, metrics.Articulated):
            meshes = [save_mesh(out, S.parts.V, S.parts.F, S.parts.top)]
            arrays["angles"] = S.part_T[keep]
        else:
            meshes = [save_mesh(out, p.V, p.F) for p in S.parts]
            if S.part_T is not None:
                arrays["part_T"] = S.part_T[keep]
        for hand, (V, valid) in S.hands.items():
            sel = g[g.hand.astype(str) == hand].frame.to_numpy()
            arrays[f"{hand}_verts"] = V[keep].astype(np.float32)
            arrays[f"{hand}_grounded"] = np.isin(arrays["frames"], sel)
            if a.labels == "rectified" and len(sel):
                Vr, extra = rectified_verts(g.rectified_key.iloc[0], hand, arrays["frames"])
                arrays[f"{hand}_verts_rectified"] = Vr.astype(np.float32)
                arrays |= {f"{k}_rectified": v for k, v in extra.items()}
        meta = {"dataset": a.dataset, "seq": seq, "meshes": meshes, "labels": a.labels, "units": "mm",
                "frame_note": "T_wo maps object base -> the dataset's world/camera frame of the hands",
                "axes": AXES.get(a.dataset, "")}
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, meta=json.dumps(meta), **arrays)
        hs_ok.append(seq)
    print(f"labels: {len(files)} sequences ({len(hs_ok)} new)", flush=True)
    return files


def cam_template(dataset, name):
    """Archive member -> (template with {cam}, cam), matching locate()'s image_member. (name, None) if no camera."""
    parts = name.split("/")
    if dataset in ("arctic", "dexycb") and len(parts) >= 2:
        cam = parts[-2]
        return "/".join(parts[:-2] + ["{cam}", parts[-1]]), cam
    if dataset == "oakink" and "_" in parts[-1]:
        cam, rest = parts[-1].split("_", 1)
        return "/".join(parts[:-1] + ["{cam}_" + rest]), cam
    return name, None


def wanted_members(idx):
    """image_key -> set of image_member templates of the grounded rows."""
    rows = idx.dropna(subset=["image_member"])
    want = defaultdict(set)
    for key, mem in zip(rows.image_key.astype(str), rows.image_member.astype(str)):
        want[key].add(mem)
    return want


def _hit(dataset, name, templates, cams):
    """Member name -> relative template it satisfies (also with the archive's top folder stripped), else None."""
    for cand in (name, name.split("/", 1)[1] if "/" in name else None):
        if cand is None:
            continue
        t, cam = cam_template(dataset, cand)
        if t in templates and (cam is None or not cams or cam in cams):
            return cand
        if cand in templates:
            return cand
    return None


def fetch_images(a, idx, out):
    cams = set(a.cams.split(",")) if (a.cams and a.images == "cams") else None
    want = wanted_members(idx)
    s3 = __import__("boto3").client("s3")
    got = defaultdict(list)
    for key, wanted in want.items():
        if key.endswith((".tar", ".tar.gz")):  # stream once, keep wanted members (ARCTIC .tar, DexYCB .tar.gz)
            body = s3.get_object(Bucket=BUCKET, Key=key)["Body"]
            with tarfile.open(fileobj=body, mode="r|gz" if key.endswith(".gz") else "r|") as tf:
                for m in tf:
                    if m.isfile() and _hit(a.dataset, m.name, wanted, cams):
                        dst = out / "images" / m.name
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(tf.extractfile(m).read())
                        got[key].append(m.name)
        elif key.endswith(".zip") and a.dataset == "hoi4d":  # one mp4 per sequence -> decode the grounded frames
            import cv2
            z = s3zip.central_dir(BUCKET, key)
            ents = {e["name"]: e for e in z["entries"]}
            for mem in wanted:
                hit = [n for n in ents if n.endswith(mem)]
                if not hit:
                    continue
                frames = set(idx[idx.image_member == mem].image_frame.astype(int))
                tmp = out / "_tmp.mp4"
                tmp.write_bytes(s3zip.read_entry(BUCKET, key, ents[hit[0]], z["shift"]))
                cap, i = cv2.VideoCapture(str(tmp)), 0
                while frames:
                    ok, im = cap.read()
                    if not ok:
                        break
                    if i in frames:
                        dst = out / "images" / mem.replace("image.mp4", f"{i:05d}.jpg")
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(dst), im)
                        got[key].append(str(dst.relative_to(out / "images")))
                        frames.discard(i)
                    i += 1
                cap.release()
                tmp.unlink()
        else:  # zip / split zip: ranged reads of single entries (HO3D, OakInk)
            parts = key if not key.endswith("oakink_image_v2.zip") else \
                [key.replace(".zip", f".z{i:02d}") for i in range(1, 11)] + [key]
            z = s3zip.central_dir(BUCKET, parts)
            for e in z["entries"]:
                name = e["name"]
                if name.endswith("/") or _hit(a.dataset, name, wanted, cams) is None:
                    continue
                dst = out / "images" / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(s3zip.read_entry(BUCKET, parts, e, z["shift"]))
                got[key].append(name)
        print(f"images: {key}: {len(got[key])} files", flush=True)
    return got


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(subset.AUDIT))
    ap.add_argument("--tag", help="published subset tag, e.g. original_pen1_gap2")
    ap.add_argument("--index", type=Path, help="local index.parquet instead of --tag")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label-root", type=Path, help="prepared labels (scripts/prepare_data.sh); omit to skip labels")
    ap.add_argument("--mano-dir", default="data/mano/mano_v1_2/models")
    ap.add_argument("--images", choices=["none", "all", "cams"], default="none")
    ap.add_argument("--cams", help="comma list for --images cams (ARCTIC 0-8, DexYCB serials, OakInk cam names)")
    ap.add_argument("--seqs", type=int, default=0, help="only the first N sequences (0 = all)")
    a = ap.parse_args()
    common.MANO_DIR = Path(a.mano_dir)
    idx = pd.read_parquet(a.index) if a.index else subset.read_index(a.dataset, a.tag)
    a.labels = "rectified" if "rectified_key" in idx else "original"
    if a.seqs:
        idx = idx[idx.seq.isin(idx.seq.unique()[: a.seqs])]
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"{a.dataset}: {len(idx)} grounded hand-frames in {idx.seq.nunique()} sequences", flush=True)
    if a.label_root:
        files = fetch_labels(a, idx, a.out)
        idx["label_file"] = idx.seq.astype(str).map(files)
    if a.images != "none":
        fetch_images(a, idx, a.out)
    idx.to_parquet(a.out / "manifest.parquet")
    print("done", a.out)


if __name__ == "__main__":
    main()
