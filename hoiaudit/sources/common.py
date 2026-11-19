"""Shared helpers for dataset sources. Every source turns one raw sequence into a metrics.Sequence (mm units)."""
from pathlib import Path

import numpy as np

from ..mano_np import MANO, rodrigues

MANO_DIR = Path("data/mano/mano_v1_2/models")  # overridden by audit.py --mano-dir


def T_from(R, t):
    """R (N,3,3), t (N,3) -> (N,4,4)."""
    T = np.zeros((len(R), 4, 4))
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1
    return T


def aa_to_R(aa):
    return rodrigues(np.asarray(aa, np.float64))


_MANO = {}


def mano(side, flat_hand_mean):
    """Cached stock MANO layer (no left-hand shapedirs patch: smplx/manopth use the stock pkl)."""
    k = (side, flat_hand_mean)
    if k not in _MANO:
        _MANO[k] = MANO(MANO_DIR / f"MANO_{side.upper()}.pkl", flat_hand_mean=flat_hand_mean)
    return _MANO[k]


def load_mesh(path):
    import trimesh
    m = trimesh.load(path, process=False, force="mesh")
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
