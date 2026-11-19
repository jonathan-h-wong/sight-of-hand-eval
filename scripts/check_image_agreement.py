"""Does the rectified hand agree with the real images as well as the original GT does? (DexYCB)

For every camera and frame, the scene (tracked objects + hand) is ray-cast twice, once with the GT hand and once with
the rectified one, and compared against two image sources:

  seg    DexYCB's per-camera segmentation (hand = 255). Tests first whether these masks are rendered from the GT
         itself: if so, agreement with them is circular, and they cannot judge the rectification.
  depth  the real aligned depth sensor image (mm). The rectified render uses the corrected object pose too, when
         the rectifier corrected it. Independent evidence. Scored on the pixels where the two renders
         differ (> 2 mm, or hand visible in only one), i.e. exactly where the rectification changed the picture,
         plus all hand pixels. Lower |render - sensor| = agrees better with the real camera.

Frozen frames (GT <= 2 mm, untouched) are scored too: they give the noise floor of GT-vs-sensor agreement.

    python scripts/check_image_agreement.py --rect out/rectify/dexycb --root data/dexycb --images data_img \
        --mano-dir data/mano/mano_v1_2/models --out out/rectify/dexycb/image_agreement.parquet
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hoiaudit.metrics import inv_T  # noqa: E402
from hoiaudit.rectify import scene as rscene  # noqa: E402
from hoiaudit.sources import common  # noqa: E402

HAND_ID = 255
CHANGED_MM = 2.0


def cast(meshes, K, W2C, box, size):
    """meshes: list of (V world metres, F); returns depth (h,w) mm (inf = miss) and geometry index (-1 = miss)
    for the pixel box (u0,v0,u1,v1)."""
    import open3d as o3d
    sc = o3d.t.geometry.RaycastingScene()
    for V, F in meshes:
        Vc = V @ W2C[:3, :3].T + W2C[:3, 3]
        sc.add_triangles(o3d.core.Tensor(Vc.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    u0, v0, u1, v1 = box
    uu, vv = np.meshgrid(np.arange(u0, u1) + 0.5, np.arange(v0, v1) + 0.5)
    d = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)   # z = 1 rays
    rays = np.concatenate([np.zeros_like(d), d], -1).reshape(-1, 6).astype(np.float32)
    r = sc.cast_rays(o3d.core.Tensor(rays))
    t = r["t_hit"].numpy().reshape(uu.shape) * 1000.0            # z = 1 rays: t is depth (m) -> mm
    gid = r["geometry_ids"].numpy().astype(np.int64).reshape(uu.shape)
    gid[~np.isfinite(t)] = -1
    return t, gid


def main():
    import pandas as pd
    from PIL import Image
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rect", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True, help="dir holding <subject>/<seq>/<serial>/{labels,aligned_depth}*")
    ap.add_argument("--mano-dir", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frozen-every", type=int, default=4, help="score every Nth frozen frame (noise floor)")
    a = ap.parse_args()
    common.MANO_DIR = Path(a.mano_dir)
    rows = []
    for npz in sorted((a.rect / "seqs").rglob("*.npz")):
        key = str(npz.relative_to(a.rect / "seqs"))[:-4]
        img_dir = a.images / key
        if not img_dir.exists():
            print("no images for", key)
            continue
        sc = rscene.dexycb(a.root / key, a.rect / "sdf_cache")
        z = np.load(npz)
        side = list(sc.hands)[0]
        m = common.mano(side, False)
        V_gt, valid = sc.seq.hands[side]
        frozen, unres = z[f"{side}_frozen"], z[f"{side}_unresolved"]
        rect = valid & ~frozen & ~unres
        V_rc = V_gt.copy()
        ri = np.where(rect)[0]
        V_rc[ri] = m(z[f"{side}_betas"], z[f"{side}_rot"][ri], z[f"{side}_pose"][ri], z[f"{side}_trans"][ri]) * 1000.0
        fz = np.where(valid & frozen)[0][::a.frozen_every]
        T_co = np.stack([inv_T(sc.seq.part_T[:, k]) for k in range(len(sc.seq.parts))], 1)      # (T,K,4,4) mm
        T_co_rect = T_co.copy()
        if "obj_world_from_obj" in z.files:                     # corrected grasped object (rectified render only)
            g = sc.seq.grasp_k
            mv = z["obj_moved"]
            Wfo = z["obj_world_from_obj"].copy()
            Wfo[:, :3, 3] *= 1000.0
            T_co_rect[mv, g] = Wfo[mv]
        tracked = np.abs(sc.seq.part_T[:, :, :3, 3]).max(-1) < 1e6
        for t in np.concatenate([ri, fz]):
            objs = [((sc.seq.parts[k].V @ T_co[t, k, :3, :3].T + T_co[t, k, :3, 3]) / 1000.0, sc.seq.parts[k].F)
                    for k in range(len(sc.seq.parts)) if tracked[t, k]]
            objs_r = [((sc.seq.parts[k].V @ T_co_rect[t, k, :3, :3].T + T_co_rect[t, k, :3, 3]) / 1000.0, sc.seq.parts[k].F)
                      for k in range(len(sc.seq.parts)) if tracked[t, k]]
            for c in sc.cams:
                lab = img_dir / c["name"] / f"labels_{t:06d}.npz"
                dep = img_dir / c["name"] / f"aligned_depth_to_color_{t:06d}.png"
                if not lab.exists() or not dep.exists():
                    continue
                W2C, K = c["W2C"][0], c["K"]
                uv = []
                for V in (V_gt[t], V_rc[t]):
                    X = V / 1000.0 @ W2C[:3, :3].T + W2C[:3, 3]
                    uv.append((X @ K.T)[:, :2] / (X @ K.T)[:, 2:])
                uv = np.concatenate(uv)
                w, h = c["size"]
                box = (max(0, int(uv[:, 0].min()) - 20), max(0, int(uv[:, 1].min()) - 20),
                       min(w, int(uv[:, 0].max()) + 20), min(h, int(uv[:, 1].max()) + 20))
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                hid = len(objs)
                dg, gg = cast(objs + [(V_gt[t] / 1000.0, m.faces)], K, W2C, box, (w, h))
                dr, gr = cast(objs_r + [(V_rc[t] / 1000.0, m.faces)], K, W2C, box, (w, h))
                hand_g, hand_r = gg == hid, gr == hid
                if not (hand_g.any() or hand_r.any()):
                    continue
                u0, v0, u1, v1 = box
                seg = np.load(lab)["seg"][v0:v1, u0:u1] == HAND_ID
                ds = np.asarray(Image.open(dep), np.float64)[v0:v1, u0:u1]
                ok = ds > 0
                iou = lambda A, B: float((A & B).sum() / max(1, (A | B).sum()))  # noqa: E731
                both = np.isfinite(dg) & np.isfinite(dr) & ok
                changed = both & ((np.abs(dg - dr) > CHANGED_MM) | (hand_g != hand_r))
                handpx = both & (hand_g | hand_r)
                med = lambda D, M: float(np.median(np.abs(D[M] - ds[M]))) if M.any() else np.nan  # noqa: E731
                rows.append({"seq": key, "frame": int(t), "cam": c["name"], "status": "rectified" if rect[t] else "frozen",
                             "iou_seg_gt": iou(hand_g, seg), "iou_seg_rect": iou(hand_r, seg),
                             "n_changed_px": int(changed.sum()), "n_hand_px": int(handpx.sum()),
                             "depth_err_changed_gt": med(dg, changed), "depth_err_changed_rect": med(dr, changed),
                             "depth_err_hand_gt": med(dg, handpx), "depth_err_hand_rect": med(dr, handpx)})
        print(key, "rectified", len(ri), "frozen sampled", len(fz), flush=True)
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out)
    r, f = df[df.status == "rectified"], df[df.status == "frozen"]
    c = r[r.n_changed_px >= 10]
    s = {
        "seg_is_rendered_from_gt": {"iou_gt_p50_frozen": round(float(f.iou_seg_gt.median()), 4),
                                    "iou_gt_p50_rectified": round(float(r.iou_seg_gt.median()), 4),
                                    "iou_rect_p50_rectified": round(float(r.iou_seg_rect.median()), 4)},
        "depth_noise_floor_frozen_hand_px_mm_p50": round(float(f.depth_err_hand_gt.median()), 2),
        "rectified_view_frames": len(r), "with_>=10_changed_px": len(c),
        "changed_px_depth_err_mm_p50": {"gt": round(float(c.depth_err_changed_gt.median()), 2),
                                        "rect": round(float(c.depth_err_changed_rect.median()), 2)},
        "changed_px_rect_better_frac": round(float((c.depth_err_changed_rect < c.depth_err_changed_gt).mean()), 3),
        "hand_px_depth_err_mm_p50": {"gt": round(float(r.depth_err_hand_gt.median()), 2),
                                     "rect": round(float(r.depth_err_hand_rect.median()), 2)},
    }
    a.out.with_suffix(".json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
