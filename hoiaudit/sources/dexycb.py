"""DexYCB (Chao et al. 2021). Only labels are needed; extract from raw/dexycb/*.tar.gz:

    ROOT/<date>-subject-NN/<seq>/{meta.yml,pose.npz}
    ROOT/calibration/mano_<id>/mano.yml          (calibration.tar.gz; per-subject betas)
    ROOT/models/<ycb_name>/textured_simple.obj   (models.tar.gz; metres)

Conventions (dex-ycb-toolkit layers/mano_layer.py, sequence_loader.py):
  hand   : manopth ManoLayer(use_pca=True, ncomps=45, flat_hand_mean=False), pose_m[:, h] = 48 (3 rot + 45 PCA) + 3 trans
           betas from the subject calibration; all frames with pose_m == 0 are untracked.
  objects: pose_y[:, k] = quaternion (x, y, z, w) + translation; world = R v + t. Identity rows are untracked.
  Everything is in the master camera frame; 30 fps; one hand per sequence.
The hand is evaluated against the union of ALL objects in the scene (each object is a Part with its own pose),
so penetration into non-grasped neighbours is caught too. Per-object depth for the grasped object is reported
as an extra column.
Mask key: <subject>/<seq>.
"""
from pathlib import Path

import numpy as np
import yaml

from ..metrics import Part, Sequence, inv_T
from .common import T_from, load_mesh, mano

YCB = {1: "002_master_chef_can", 2: "003_cracker_box", 3: "004_sugar_box", 4: "005_tomato_soup_can",
       5: "006_mustard_bottle", 6: "007_tuna_fish_can", 7: "008_pudding_box", 8: "009_gelatin_box",
       9: "010_potted_meat_can", 10: "011_banana", 11: "019_pitcher_base", 12: "021_bleach_cleanser",
       13: "024_bowl", 14: "025_mug", 15: "035_power_drill", 16: "036_wood_block", 17: "037_scissors",
       18: "040_large_marker", 19: "051_large_clamp", 20: "052_extra_large_clamp", 21: "061_foam_brick"}
_PARTS = {}


def list_tasks(root):
    return sorted(str(p.parent) for p in Path(root).glob("*-subject-*/*/pose.npz"))


def _part(root, yid):
    if yid not in _PARTS:
        V, F = load_mesh(Path(root) / "models" / YCB[yid] / "textured_simple.obj")
        _PARTS[yid] = Part(V * 1000.0, F)
    return _PARTS[yid]


def _quat_R(q):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(q).as_matrix()


def load(task, stride=1):
    task = Path(task)
    root = task.parent.parent
    meta = yaml.safe_load(open(task / "meta.yml"))
    z = np.load(task / "pose.npz")
    pose_y, pose_m = z["pose_y"], z["pose_m"]
    N = len(pose_m)
    fr = np.arange(0, N, stride)
    ids = list(meta["ycb_ids"])
    parts = [_part(root, i) for i in ids]
    K = len(ids)
    # object k pose (camera <- object k); part_T maps the base (camera) frame -> object k local
    Py = pose_y[fr].reshape(-1, 7)
    T_co = T_from(_quat_R(Py[:, :4]), Py[:, 4:] * 1000.0).reshape(len(fr), K, 4, 4)
    tracked = ~np.all(np.isclose(pose_y[fr], [0, 0, 0, 1, 0, 0, 0]), axis=2)  # (n,K)
    part_T = inv_T(T_co.reshape(-1, 4, 4)).reshape(len(fr), K, 4, 4)
    # untracked objects: move far away so they never register as near
    far = np.eye(4)
    far[:3, 3] = 1e7
    part_T[~tracked] = far
    T_wo = np.broadcast_to(np.eye(4), (len(fr), 4, 4)).copy()
    side = meta["mano_sides"][0]
    betas = np.array(yaml.safe_load(open(root / "calibration" / f"mano_{meta['mano_calib'][0]}" / "mano.yml"))["betas"])
    pm = pose_m[fr, 0]
    valid = np.any(pm != 0, axis=1)
    V = np.full((len(fr), 778, 3), np.nan)
    if valid.any():
        m = mano(side, False)
        V[valid] = m(betas, pm[valid, :3], m.pca_to_pose(pm[valid, 3:48]), pm[valid, 48:51]) * 1000.0
    g = int(meta["ycb_grasp_ind"])
    extra = {"grasp_obj": np.full(len(fr), YCB[ids[g]]), "grasp_obj_tracked": tracked[:, g],
             "n_objects": np.full(len(fr), K)}
    seq = Sequence("dexycb", f"{task.parent.name}/{task.name}", 30.0, stride, fr, T_wo, parts, part_T,
                   {side: (V, valid)}, extra)
    seq.grasp_k = g
    return seq
