"""Physically grounded subsets: select frames from an audit, index them, and point back into raw/.

A subset is defined by a `Criteria` and built from one audit table (eval/<ds>/penetration_v1/frames.parquet, or the
rectified pseudo-GT report for ARCTIC/DexYCB). The index has one row per grounded (seq, hand, frame) with everything a
loader needs to fetch it: the audit numbers, the run it belongs to, and the raw label/image locations.

    from hoiaudit import subset
    c = subset.Criteria(pen_mm=1.0, gap_mm=2.0)
    idx = subset.build_index("dexycb", c)               # pandas DataFrame
    idx = subset.read_index("dexycb", c.tag)            # a published one, from s3://xray-hands/eval/<ds>/grounded_v1/

Selection (all thresholds in mm, applied to the exact audit metric):
  contact   min_sd_mm <= contact_mm (default 5: the audit's contact definition). With include_free=True, frames
            with no contact at all (min_sd_mm > contact_mm) are kept too and flagged contact=False.
  pen       pen_max_mm <= pen_mm
  gap       min_sd_mm <= gap_mm for contact frames: the hand really touches (rejects hovering 2-5 mm away)
  jitter    rel_hf_mm <= jitter_mm when set (frames whose jitter cannot be evaluated pass)
  min_run   keep only runs of >= min_run consecutive grounded frames (step = the audit stride)
  labels    "original" (dataset GT) or "rectified" (ARCTIC/DexYCB pseudo-GT from scripts/rectify.py; unresolved
            frames are excluded)
"""
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

BUCKET = "xray-hands"
AUDIT = {  # dataset -> audit table (S3 key)
    "arctic": "eval/arctic/penetration_v1/frames.parquet",
    "dexycb": "eval/dexycb/penetration_v1/frames.parquet",
    "grab": "eval/grab/penetration_v1/frames.parquet",
    "graspxl": "eval/graspxl/penetration_v2_sample/frames.parquet",
    "ho3d": "eval/ho3d/penetration_v1/frames.parquet",
    "hoi4d": "eval/hoi4d/penetration_v1/frames.parquet",
    "oakink": "eval/oakink/penetration_v1/frames.parquet",
}
RECTIFIED = {"arctic": "work/rectify_v1/out/arctic", "dexycb": "work/rectify_v1/out/dexycb"}
STEP = {}  # audit stride per dataset if not 1


@dataclass(frozen=True)
class Criteria:
    pen_mm: float = 1.0
    gap_mm: float = 2.0
    contact_mm: float = 5.0
    jitter_mm: float | None = None
    min_run: int = 1
    labels: str = "original"
    include_free: bool = False

    @property
    def tag(self):
        t = f"{self.labels}_pen{self.pen_mm:g}_gap{self.gap_mm:g}"
        if self.jitter_mm is not None:
            t += f"_jit{self.jitter_mm:g}"
        if self.min_run > 1:
            t += f"_run{self.min_run}"
        if self.include_free:
            t += "_free"
        return t


def s3_read_parquet(key, columns=None):
    import io

    import boto3
    body = boto3.client("s3").get_object(Bucket=BUCKET, Key=key)["Body"].read()
    return pd.read_parquet(io.BytesIO(body), columns=columns)


def load_audit(dataset, labels="original", path=None):
    """Normalized audit table: seq, hand, frame, pen_max_mm, min_sd_mm, rel_hf_mm (+ dataset extras)."""
    if labels == "original":
        df = pd.read_parquet(path) if path else s3_read_parquet(AUDIT[dataset])
        df["unresolved"] = False
        return df
    if labels != "rectified" or dataset not in RECTIFIED:
        raise ValueError(f"no {labels} labels for {dataset}")
    df = pd.read_parquet(path) if path else s3_read_parquet(RECTIFIED[dataset] + "/frames.parquet")
    return df.rename(columns={"pen_after_mm": "pen_max_mm", "min_sd_after_mm": "min_sd_mm",
                              "rel_hf_after_mm": "rel_hf_mm"})


def select(df, c: Criteria, step=1):
    """Adds contact, grounded, run_id, run_len columns; returns the grounded rows only."""
    df = df.sort_values(["seq", "hand", "frame"]).reset_index(drop=True)
    contact = df.min_sd_mm <= c.contact_mm
    ok = (df.pen_max_mm <= c.pen_mm) & ~df.unresolved.astype(bool)
    ok &= np.where(contact, df.min_sd_mm <= c.gap_mm, c.include_free)
    if c.jitter_mm is not None and "rel_hf_mm" in df:
        ok &= ~(df.rel_hf_mm > c.jitter_mm)  # NaN (not evaluable) passes
    df["contact"] = contact
    df["grounded"] = ok
    new = (df.seq != df.seq.shift()) | (df.hand != df.hand.shift()) | (df.frame.diff() != step) | (ok != ok.shift())
    df["run_id"] = new.cumsum()
    df["run_len"] = df.groupby("run_id").frame.transform("size")
    out = df[df.grounded & (df.run_len >= c.min_run)].copy()
    out["run_id"] = pd.factorize(out.run_id)[0]
    return out


# ------------------------------------------------------------------------------------------------ raw locations
def _arctic_offsets():
    """ARCTIC image index = mocap frame + ioi_offset[subject] (meta/misc.json)."""
    from . import s3zip
    key = "raw/arctic/data/meta.zip"
    z = s3zip.central_dir(BUCKET, key)
    e = next(e for e in z["entries"] if e["name"].endswith("misc.json"))
    misc = json.loads(s3zip.read_entry(BUCKET, key, e))
    return {sid: int(v["ioi_offset"]) for sid, v in misc.items()}


def locate(dataset, idx, labels="original"):
    """Adds raw pointer columns: label_key [+ label_member], image_key, image_member (a template with {cam} where the
    camera is a free choice), image_frame. Images are absent for GRAB and GraspXL (no images released)."""
    s = idx.seq.astype(str)
    if dataset == "arctic":
        sid, seq = s.str.split("/").str[0], s.str.split("/").str[1]
        off = _arctic_offsets()
        idx["label_key"] = "raw/arctic/data/raw_seqs.zip"
        idx["label_member"] = "raw_seqs/" + s + ".mano.npy"
        idx["image_frame"] = idx.frame + sid.map(off).astype(int)
        idx["image_key"] = "raw/arctic/unpacked/arctic_data/data/images/" + s + ".tar"
        idx["image_member"] = ("arctic_data/data/images/" + s + "/{cam}/"
                               + idx.image_frame.map("{:05d}".format) + ".jpg")  # cam 0 = egocentric, 1-8 allocentric
    elif dataset == "dexycb":
        subj = s.str.split("/").str[0]
        idx["label_key"] = "raw/dexycb/" + subj + ".tar.gz"
        idx["label_member"] = s + "/pose.npz"
        idx["image_frame"] = idx.frame
        idx["image_key"] = idx.label_key
        idx["image_member"] = s + "/{cam}/color_" + idx.frame.map("{:06d}".format) + ".jpg"  # cam = camera serial
    elif dataset == "grab":
        sn = s.str.split("/").str[0]
        idx["label_key"] = "raw/grab/grab__" + sn + ".zip"
        idx["label_member"] = s + ".npz"
    elif dataset == "graspxl":
        arch = s.str.split("/").str[0]
        idx["label_key"] = "raw/graspxl/" + arch + ".zip"  # mano_dataset_N: the .part00 + .part01 concatenation
        idx["label_member"] = s + ".npy"
    elif dataset == "ho3d":
        # HO3D v3 films most captures from up to 5 cameras: <name>1<k> (two trailing digits) is capture <name>1 seen
        # from camera k, with identical poses in each camera frame (checked: equal penetration on every frame).
        # Evaluate or bootstrap per capture, not per sequence.
        name = s.str.split("/").str[1]
        two = name.str.match(r"^[A-Za-z]+\d\d$")
        idx["capture"] = np.where(two, name.str[:-1], name)
        idx["camera"] = np.where(two, name.str[-1], "0").astype(int)
        idx["label_key"] = "raw/ho3d_v3/HO3D_v3/HO3D_v3.zip"
        idx["label_member"] = s + "/meta/" + idx.frame.map("{:04d}".format) + ".pkl"
        idx["image_frame"] = idx.frame
        idx["image_key"] = idx.label_key
        idx["image_member"] = s + "/rgb/" + idx.frame.map("{:04d}".format) + ".jpg"
    elif dataset == "hoi4d":
        idx["label_key"] = "raw/hoi4d/HOI4D_annotations.zip"
        idx["label_member"] = s + "/objpose/" + idx.frame.astype(str) + ".json"
        idx["image_frame"] = idx.frame
        idx["image_key"] = "raw/hoi4d/HOI4D_release.zip"
        idx["image_member"] = s + "/align_rgb/image.mp4"  # decode frame image_frame
    elif dataset == "oakink":
        idx["label_key"] = "raw/oakink_v1/image/anno_v2_1.zip"
        idx["image_frame"] = idx.frame
        idx["image_key"] = "raw/oakink_v1/image/stream_zipped/oakink_image_v2.zip"  # + .z01-.z10, one split archive
        idx["image_member"] = "stream_release_v2/" + s + "/{cam}_" + idx.frame.astype(str) + ".png"
    if labels == "rectified":
        idx["rectified_key"] = RECTIFIED[dataset] + "/seqs/" + s + ".npz"
    return idx


def summarize(df_all, idx, c: Criteria, dataset):
    by = "object" if "object" in idx else ("grasp_obj" if "grasp_obj" in idx else None)
    con = df_all[df_all.min_sd_mm <= c.contact_mm]
    out = {"dataset": dataset, "criteria": asdict(c), "tag": c.tag,
           "hand_frames_audited": int(len(df_all)), "contact_frames": int(len(con)),
           "grounded_frames": int(len(idx)), "grounded_contact_frames": int(idx.contact.sum()),
           "grounded_frac_of_contact": round(float(idx.contact.sum()) / max(1, len(con)), 4),
           "sequences": int(idx.seq.nunique()), "runs": int(idx.run_id.nunique()),
           "runs_ge15": int((idx.groupby("run_id").size() >= 15).sum()),
           "longest_run": int(idx.groupby("run_id").size().max()) if len(idx) else 0}
    if by:
        out["by_object"] = {str(k): int(v) for k, v in idx.groupby(by, observed=True).size().sort_values().items()}
    return out


def build_index(dataset, c: Criteria, path=None):
    df = load_audit(dataset, c.labels, path)
    idx = select(df, c, STEP.get(dataset, 1))
    return locate(dataset, idx, c.labels), summarize(df, idx, c, dataset)


def read_index(dataset, tag):
    return s3_read_parquet(f"eval/{dataset}/grounded_v1/{tag}/index.parquet")
