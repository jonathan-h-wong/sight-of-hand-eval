"""Signed-distance grids of object meshes for fast, differentiable lookup inside the optimizer.

A grid is sampled once per mesh with the same signed distance as the audit (libigl fast winding number, negative
inside) and cached as .npz. Lookup is trilinear (torch grid_sample), so gradients point along the SDF gradient.
The grid is only an optimization proxy: every before/after number is recomputed with the exact metric
(hoiaudit.metrics.penetration_metrics).
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

PAD_MM = 20.0     # grid extends this far past the mesh bounds; outside it the SDF reads as +PAD_MM
MAX_DIM = 256     # grid cells per axis at most; resolution = max(RES_MM, extent / (MAX_DIM - 1))
RES_MM = 1.0


def build(V, F, res_mm=RES_MM, pad_mm=PAD_MM, cache_dir=None):
    """V (nv,3) mm, F (nf,3) -> dict(G (nx,ny,nz) float32 mm, lo (3,), h float)."""
    import igl
    V = np.ascontiguousarray(V, np.float64)
    F = np.ascontiguousarray(F, np.int64)
    key = hashlib.sha1(V.tobytes() + F.tobytes() + f"{res_mm},{pad_mm},{MAX_DIM}".encode()).hexdigest()[:16]
    if cache_dir is not None:
        p = Path(cache_dir) / f"sdf_{key}.npz"
        if p.exists():
            z = np.load(p)
            return {"G": z["G"], "lo": z["lo"], "h": float(z["h"])}
    lo, hi = V.min(0) - pad_mm, V.max(0) + pad_mm
    h = max(res_mm, float((hi - lo).max()) / (MAX_DIM - 1))
    n = np.ceil((hi - lo) / h).astype(int) + 1
    axes = [lo[i] + h * np.arange(n[i]) for i in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    out = np.empty(len(P))
    for s in range(0, len(P), 2_000_000):
        out[s:s + 2_000_000], *_ = igl.signed_distance(P[s:s + 2_000_000], V, F,
                                                       sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)
    g = {"G": out.reshape(*n).astype(np.float32), "lo": lo, "h": h}
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp.npz")   # atomic: parallel processes may build the same grid
        np.savez_compressed(tmp, **g)
        os.replace(tmp, p)
    return g


class Grid(torch.nn.Module):
    def __init__(self, g, device="cpu"):
        super().__init__()
        G = torch.as_tensor(g["G"], dtype=torch.float32, device=device)
        self.register_buffer("G", G[None, None])          # (1,1,nx,ny,nz)
        self.register_buffer("lo", torch.as_tensor(g["lo"], dtype=torch.float32, device=device))
        self.n = torch.tensor(G.shape, dtype=torch.float32, device=device)
        self.h = float(g["h"])

    def forward(self, X):
        """X (..., 3) mm in the mesh frame -> (...,) SDF mm (outside the grid: +PAD_MM)."""
        shp = X.shape[:-1]
        u = (X.reshape(-1, 3).float() - self.lo) / self.h                 # voxel coordinates
        norm = 2 * u / (self.n - 1) - 1                                   # [-1, 1] per axis (align_corners)
        # grid_sample takes (x, y, z) = (W, H, D) = our (k, j, i)
        s = torch.nn.functional.grid_sample(self.G, norm.flip(-1).reshape(1, -1, 1, 1, 3), mode="bilinear",
                                            padding_mode="border", align_corners=True).reshape(-1)
        inside = ((norm > -1) & (norm < 1)).all(-1)
        return torch.where(inside, s, torch.full_like(s, PAD_MM)).reshape(shp).to(X.dtype)
