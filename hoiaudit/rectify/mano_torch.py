"""Differentiable MANO in PyTorch, built from the NumPy model (hoiaudit.mano_np.MANO) so both share one pkl loader
and produce identical meshes (checked in rectify.scene at load time)."""
import numpy as np
import torch


def rodrigues(aa):
    """(..., 3) axis-angle -> (..., 3, 3), same epsilon handling as smplx / mano_np."""
    angle = torch.linalg.norm(aa + 1e-8, dim=-1, keepdim=True)
    axis = aa / angle
    s, c = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(aa.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).expand(K.shape)
    return eye + s * K + (1 - c) * (K @ K)


class TorchMANO(torch.nn.Module):
    def __init__(self, m, dtype=torch.float32):
        super().__init__()
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype)  # noqa: E731
        self.register_buffer("v_template", t(m.v_template))
        self.register_buffer("shapedirs", t(m.shapedirs))
        self.register_buffer("posedirs", t(m.posedirs))
        self.register_buffer("J_regressor", t(m.J_regressor))
        self.register_buffer("weights", t(m.weights))
        self.register_buffer("pose_mean", t(m.pose_mean))
        self.parents = [int(p) for p in m.parents]

    def forward(self, betas, global_orient, hand_pose, transl):
        """betas (10,) or (B,10); global_orient (B,3); hand_pose (B,45); transl (B,3). Metres in, metres out.
        Returns verts (B,778,3)."""
        B = global_orient.shape[0]
        betas = betas.reshape(-1, 10).expand(B, 10)
        full = torch.cat([global_orient, hand_pose], 1) + self.pose_mean
        v_shaped = self.v_template + torch.einsum("vck,bk->bvc", self.shapedirs, betas)
        J = torch.einsum("jv,bvc->bjc", self.J_regressor, v_shaped)
        R = rodrigues(full.reshape(B, 16, 3))
        eye = torch.eye(3, dtype=R.dtype, device=R.device)
        v_posed = v_shaped + ((R[:, 1:] - eye).reshape(B, 135) @ self.posedirs).reshape(B, 778, 3)
        par = self.parents
        rel_J = torch.cat([J[:, :1], J[:, 1:] - J[:, par[1:]]], 1)
        # rigid transforms along the kinematic chain, kept as (R, t) pairs
        Rs, ts = [R[:, 0]], [rel_J[:, 0]]
        for i in range(1, 16):
            p = par[i]
            Rs.append(Rs[p] @ R[:, i])
            ts.append((Rs[p] @ rel_J[:, i, :, None])[..., 0] + ts[p])
        Rw, tw = torch.stack(Rs, 1), torch.stack(ts, 1)              # (B,16,3,3), (B,16,3)
        tw = tw - (Rw @ J[..., None])[..., 0]                         # remove rest-pose joint location
        Wr = torch.einsum("vj,bjik->bvik", self.weights, Rw)
        Wt = torch.einsum("vj,bji->bvi", self.weights, tw)
        return (Wr @ v_posed[..., None])[..., 0] + Wt + transl[:, None, :]
