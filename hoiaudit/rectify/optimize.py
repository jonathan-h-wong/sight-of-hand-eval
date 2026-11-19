"""Sequence-level GT rectification (ContactOpt-style pose optimization + camera consistency + temporal smoothness).

Per hand and sequence, MANO global rotation, articulation (45-d) and translation get a per-frame correction.
Hand shape is never changed. The object (ARCTIC: the articulated object; DexYCB: the grasped object) gets a per-frame
rigid correction too (MOVE_OBJECT), optimized jointly with the hands and held to the same camera / smoothness terms
(on 400 surface points). Penetration can be the object's pose error as much as the hand's; depth-sensor checks on
DexYCB showed deep penetrations agree with the sensor better when only the hand is pushed out (see
scripts/check_image_agreement.py). The object only moves at frames where every hand in contact with it is being
rectified, so a grounded (frozen) hand's relation to the object is never changed; its articulation angle is kept.

Unresolved: active frames still > FREEZE_MM after optimization (a local push cannot fix them, e.g. a finger passing
through a thin part) are reset to the untouched GT, flagged, and frozen; the others are then re-optimized so the
corrections fade smoothly into them as well.

Frozen: every hand-frame whose GT max penetration (exact metric) is <= FREEZE_MM, and every frame without a valid
GT hand. Their corrections are identically zero, so their output parameters are bit-identical to the GT. Only the
remaining "active" frames move, and the temporal terms make the correction fade smoothly to zero at the frozen
neighbours, so no step is introduced at the boundaries.

Loss (all terms in mm^2, means over frames):
  pen      sum_p relu(-sdf(p) - TARGET_MM)^2   dense hand surface (1x subdivided MANO) vs union of object grids
  contact  per hand region (16 bones, by skinning weight) that touched the object (closest point < CONTACT_MM):
           relu(min_p sdf(p) - max(min_p sdf0(p), 0) - CONTACT_SLACK_MM)^2
           a finger segment that touched may lift out of the object but must stay on its surface. Per region, not
           per point: lifting a finger out raises all of its points, which a per-point rule would forbid.
  cam      per visible camera and joint: pixel change converted to mm at the joint's depth (|d uv| * z / f), i.e.
           the displacement a camera can see. Motion along a camera's ray is free in that camera only; with several
           views the union pins the joint. Cameras that do not see a joint do not constrain it.
  j3d      small 3D joint-displacement term (keeps joints no camera sees from wandering)
  vel/acc  first and second differences in time of the joint displacement (frozen frames contribute zeros)
  prior    |d pose|^2 (rad^2, scaled) keeps the articulation close to the GT
  ocam / oj3d / ovel / oacc / oprior   the same terms for the object's sample points (and its rotation)

Hand points are the 778 MANO vertices plus edge midpoints (= metrics.HandSurface), so the optimized surface is
the same one the audit measures. Final numbers always come from the exact metric, not the grid.
"""
import time

import numpy as np
import torch

from .. import metrics
from ..sources import common
from . import sdfgrid
from .mano_torch import TorchMANO

FREEZE_MM = 2.0        # GT hand-frames with max penetration <= this are never modified
TARGET_MM = 1.0        # optimizer target (below FREEZE_MM: margin for the grid's interpolation error)
CONTACT_MM = 5.0
CONTACT_SLACK_MM = 0.5  # a touching region's closest point may end this much farther away than before for free
TIPS = [744, 320, 443, 554, 671]
W = {"pen": 30.0, "contact": 1.0, "cam": 20.0, "j3d": 0.5, "vel": 20.0, "acc": 20.0, "prior": 100.0,
     "ocam": 20.0, "oj3d": 0.5, "ovel": 20.0, "oacc": 20.0, "oprior": 100.0}
MOVE_OBJECT = True
ITERS, ROUNDS, FINAL_PASSES = 300, 2, 2
LR = {"rot": 2e-3, "pose": 5e-3, "trans": 0.1, "orot": 1e-3, "otrans": 0.1}   # rad / mm per step


def _consecutive(valid):
    """index pairs (t, t+1) and triples (t-1, t, t+1) of consecutive valid frames."""
    v = valid
    p = np.where(v[:-1] & v[1:])[0]
    q = np.where(v[:-2] & v[1:-1] & v[2:])[0] + 1
    return p, q


def _T4(R, t):
    T = torch.zeros(R.shape[:-2] + (4, 4), dtype=R.dtype, device=R.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1
    return T


def _cam_terms(cams, X_mm, uv0, zf0, vis0):
    """Pixel change per visible camera, converted to mm at depth. X_mm (n,P,3) world."""
    out = []
    for k, (W2C, K, _) in enumerate(cams):
        uv, _, _ = _project(W2C, K, None, X_mm)
        e = ((uv - uv0[k]).norm(dim=-1) * zf0[k]) ** 2
        out.append((e * vis0[k]).sum() / vis0[k].sum().clamp(min=1))
    return torch.stack(out).mean()


def _project(W2C, K, size, X_mm):
    X = torch.einsum("nij,npj->npi", W2C[:, :3, :3], X_mm / 1000.0) + W2C[:, None, :3, 3]
    z = X[..., 2]
    uv = (X @ K.T)[..., :2] / z[..., None].clamp(min=1e-3)
    if size is None:
        return uv, None, None
    w, h = size
    vis = (z > 0.05) & (uv[..., 0] > -0.1 * w) & (uv[..., 0] < 1.1 * w) & (uv[..., 1] > -0.1 * h) & (uv[..., 1] < 1.1 * h)
    f = 0.5 * (K[0, 0] + K[1, 1])
    return uv, z * 1000.0 / f, vis


class _ObjectDelta:
    """Per-frame rigid correction D_t of one object, in its own frame (mm): base_t -> inv(D_t) @ base_t."""

    def __init__(self, scene, movable, dev, dt):
        o = scene.obj
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)  # noqa: E731
        self.T = len(movable)
        self.dev, self.dt = dev, dt
        self.tracked = o["tracked"]
        self.d_rot = torch.zeros(self.T, 3, dtype=dt, device=dev, requires_grad=True)
        self.d_trans = torch.zeros(self.T, 3, dtype=dt, device=dev, requires_grad=True)
        self.mask = t(movable.astype(np.float64))[:, None]
        base = np.asarray(o["base"], np.float64)
        self.base = t(base)
        self.Binv = t(metrics.inv_T(np.where(self.tracked[:, None, None], base, np.eye(4))))
        self.movable_parts = set(o["movable"])
        # pre_k: L_k = pre_k @ base  (identity for a rigid object, the hinge rotation for ARCTIC's top part)
        self.pre = {}
        for k in self.movable_parts:
            L = np.asarray(scene.objects[k][1], np.float64)
            L = L if len(L) > 1 else np.repeat(L, self.T, 0)
            self.pre[k] = t(L @ metrics.inv_T(np.where(self.tracked[:, None, None], base, np.eye(4))))
        self.pts = t(o["pts"])
        self.mi = np.where(movable)[0]                                  # frames that may move (at start)
        self.cams = []
        for c in scene.cams:
            W2C = c["W2C"][self.mi] if len(c["W2C"]) > 1 else np.repeat(c["W2C"], len(self.mi), 0)
            self.cams.append((t(W2C), t(c["K"]), c["size"]))
        with torch.no_grad():
            X0 = self.world_pts(zero=True)
            self.X0 = X0
            pr = [_project(W, K, sz, X0[self.mi]) for W, K, sz in self.cams]
            self.uv0, self.zf0, self.vis0 = zip(*pr)
        p, q = _consecutive(self.tracked)
        self.p0, self.p1 = (torch.as_tensor(x, device=dev) for x in (p, p + 1))
        self.q0, self.q1, self.q2 = (torch.as_tensor(x, device=dev) for x in (q - 1, q, q + 1))

    def params(self):
        return [{"params": [self.d_rot], "lr": LR["orot"]}, {"params": [self.d_trans], "lr": LR["otrans"]}]

    def D(self, idx=None, zero=False):
        r, tr, m = (self.d_rot, self.d_trans, self.mask) if idx is None else (self.d_rot[idx], self.d_trans[idx], self.mask[idx])
        m = m * 0 if zero else m
        from .mano_torch import rodrigues
        return rodrigues(m * r), m * tr

    def invD(self, idx):
        R, t = self.D(idx)
        Rt = R.transpose(-1, -2)
        return _T4(Rt, -(Rt @ t[..., None])[..., 0])

    def local(self, k, idx, L):
        """grid k's world -> grid transform at frames idx (L: its uncorrected (n,4,4))."""
        if k not in self.movable_parts:
            return L
        return self.pre[k][idx] @ self.invD(idx) @ self.base[idx]

    def world_pts(self, zero=False):
        R, t = self.D(zero=zero)
        P = torch.einsum("nij,npj->npi", R, self.pts) + t[:, None]
        return torch.einsum("nij,npj->npi", self.Binv[:, :3, :3], P) + self.Binv[:, None, :3, 3]

    def loss(self):
        X = self.world_pts()
        dX = X - self.X0
        L = {"ocam": _cam_terms(self.cams, X[self.mi], self.uv0, self.zf0, self.vis0),
             "oj3d": (dX[self.mi] ** 2).sum(-1).mean(),
             "ovel": ((dX[self.p1] - dX[self.p0]) ** 2).sum(-1).mean() if len(self.p0) else dX.sum() * 0,
             "oacc": ((dX[self.q2] - 2 * dX[self.q1] + dX[self.q0]) ** 2).sum(-1).mean() if len(self.q0) else dX.sum() * 0,
             "oprior": ((self.mask * self.d_rot) ** 2).sum(-1).mean()}
        frac = max(len(self.mi), 1) / max(int(self.tracked.sum()), 1)
        for k in ("ovel", "oacc"):
            L[k] = L[k] / frac
        return sum(W[k] * v for k, v in L.items()), L

    def lock(self, frames):
        if len(frames):
            with torch.no_grad():
                rows = torch.as_tensor(np.asarray(frames), device=self.dev)
                self.mask[rows] = 0.0
                self.d_rot[rows] = 0.0
                self.d_trans[rows] = 0.0

    def numpy_D(self):
        with torch.no_grad():
            R, t = self.D()
            return _T4(R, t).double().cpu().numpy()


class _HandProblem:
    def __init__(self, scene, side, h, active, grids, dev, dt, obj=None):
        self.obj = obj
        self.ai_t = None
        self.T = len(h["valid"])
        self.valid = h["valid"]
        self.vi = np.where(self.valid)[0]
        self.active = active
        self.ai = np.where(active)[0]
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dt, device=dev)  # noqa: E731
        m = common.mano(side, h["flat"])
        self.mano = TorchMANO(m, dt).to(dev)
        self.betas = t(h["betas"])
        self.rot0, self.pose0, self.trans0 = t(h["rot"][self.vi]), t(h["pose"][self.vi]), t(h["trans"][self.vi])
        nv = len(self.vi)
        self.d_rot = torch.zeros(nv, 3, dtype=dt, device=dev, requires_grad=True)
        self.d_pose = torch.zeros(nv, 45, dtype=dt, device=dev, requires_grad=True)
        self.d_trans = torch.zeros(nv, 3, dtype=dt, device=dev, requires_grad=True)   # mm
        self.amask = t(active[self.vi].astype(np.float64))[:, None]                       # 1 = may move
        self.a_in_v = torch.as_tensor(np.searchsorted(self.vi, self.ai), device=dev)
        hs = metrics.HandSurface(m.faces)
        coo = hs.S.tocoo()
        self.S = torch.sparse_coo_tensor(np.vstack([coo.row, coo.col]), coo.data, coo.shape, dtype=dt, device=dev)
        self.J_reg = t(m.J_regressor)
        reg = m.weights.argmax(1)                                       # region (bone) of each MANO vertex
        dense_reg = np.asarray(hs.S.tocsr()[:, :].argmax(1)).ravel()    # midpoints take one endpoint's region
        self.region = torch.as_tensor(reg[dense_reg], device=dev)
        self.n_reg = int(reg.max()) + 1
        # temporal structure over valid frames (indices into vi)
        pv, qv = _consecutive(self.valid)
        pos = -np.ones(self.T, int)
        pos[self.vi] = np.arange(nv)
        self.p0, self.p1 = (torch.as_tensor(pos[x], device=dev) for x in (pv, pv + 1))
        self.q0, self.q1, self.q2 = (torch.as_tensor(pos[x], device=dev) for x in (qv - 1, qv, qv + 1))
        # objects: world(mm) -> grid frame, for the active frames
        self.grids = grids
        self.L = [t(L[self.ai] if len(L) > 1 else np.repeat(L, len(self.ai), 0)) for _, L in scene.objects]
        self.ai_t = torch.as_tensor(self.ai, device=dev)
        # cameras: world(m) -> cam, for valid frames
        self.cams = []
        for c in scene.cams:
            W2C = c["W2C"][self.vi] if len(c["W2C"]) > 1 else np.repeat(c["W2C"], nv, 0)
            self.cams.append((t(W2C), t(c["K"]), c["size"]))
        with torch.no_grad():
            V0, J0 = self.forward(zero=True)
            self.J0 = J0
            self.uv0, self.zf0, self.vis0 = zip(*[self._proj(J0, k) for k in range(len(self.cams))])
            if len(self.ai):
                self.rmin0 = self._region_min(self.sdf(V0))
                self.reg_touch = (self.rmin0 < CONTACT_MM).to(dt)
        self.wf = torch.ones(len(self.ai), dtype=dt, device=dev)   # per-active-frame penetration weight

    def forward(self, zero=False):
        m = 0.0 if zero else self.amask
        V = self.mano(self.betas, self.rot0 + m * self.d_rot, self.pose0 + m * self.d_pose,
                      self.trans0 + m * self.d_trans / 1000.0) * 1000.0     # mm
        J = torch.cat([torch.einsum("jv,bvc->bjc", self.J_reg, V), V[:, TIPS]], 1)
        return V, J

    def _proj(self, J, k):
        W2C, K, (w, h) = self.cams[k]
        X = torch.einsum("nij,npj->npi", W2C[:, :3, :3], J / 1000.0) + W2C[:, None, :3, 3]
        z = X[..., 2]
        uv = (X @ K.T)[..., :2] / z[..., None].clamp(min=1e-3)
        vis = (z > 0.05) & (uv[..., 0] > -0.1 * w) & (uv[..., 0] < 1.1 * w) & (uv[..., 1] > -0.1 * h) & (uv[..., 1] < 1.1 * h)
        f = 0.5 * (K[0, 0] + K[1, 1])
        return uv, z * 1000.0 / f, vis          # mm per pixel at the joint's depth

    def _region_min(self, sd):
        """(n,P) -> (n, regions) closest signed distance per hand region."""
        idx = self.region.expand(sd.shape[0], -1)
        init = torch.full((sd.shape[0], self.n_reg), sdfgrid.PAD_MM, dtype=sd.dtype, device=sd.device)
        return init.scatter_reduce(1, idx, sd, reduce="amin", include_self=True)

    def dense(self, V):
        n = V.shape[0]
        X = torch.sparse.mm(self.S, V.permute(1, 0, 2).reshape(V.shape[1], n * 3))
        return X.reshape(-1, n, 3).permute(1, 0, 2)

    def sdf(self, V):
        P = self.dense(V[self.a_in_v])
        out = None
        for k, (g, L) in enumerate(zip(self.grids, self.L)):
            if self.obj is not None:
                L = self.obj.local(k, self.ai_t, L)
            s = g(torch.einsum("nij,npj->npi", L[:, :3, :3], P) + L[:, None, :3, 3])
            out = s if out is None else torch.minimum(out, s)
        return out

    def loss(self):
        V, J = self.forward()
        L = {}
        if len(self.ai):
            sd = self.sdf(V)
            L["pen"] = (self.wf[:, None] * torch.relu(-sd - TARGET_MM) ** 2).sum(1).mean()
            rmin = self._region_min(sd)
            L["contact"] = (self.reg_touch * torch.relu(rmin - self.rmin0.clamp(min=0) - CONTACT_SLACK_MM) ** 2).sum(1).mean()
        dJ = J - self.J0
        cam = []
        for k in range(len(self.cams)):
            uv, _, _ = self._proj(J, k)
            e = ((uv - self.uv0[k]).norm(dim=-1) * self.zf0[k]) ** 2
            cam.append((e * self.vis0[k]).sum() / self.vis0[k].sum().clamp(min=1))
        L["cam"] = torch.stack(cam).mean()
        L["j3d"] = (dJ ** 2).sum(-1).mean()
        L["vel"] = ((dJ[self.p1] - dJ[self.p0]) ** 2).sum(-1).mean() if len(self.p0) else dJ.sum() * 0
        L["acc"] = ((dJ[self.q2] - 2 * dJ[self.q1] + dJ[self.q0]) ** 2).sum(-1).mean() if len(self.q0) else dJ.sum() * 0
        L["prior"] = ((self.amask * self.d_pose) ** 2).sum(-1).mean()
        # mean-over-frames terms are rescaled to the active frames so weights do not depend on how much is frozen
        frac = max(len(self.ai), 1) / max(len(self.vi), 1)
        for k in ("cam", "j3d", "vel", "acc", "prior"):
            L[k] = L[k] / frac
        return sum(W[k] * v for k, v in L.items()), L

    def params(self, h):
        """Full-length float64 parameters: the GT values, plus the correction on active frames only (frozen and
        invalid frames are copied from the GT bit for bit)."""
        with torch.no_grad():
            out = {}
            for k, d, s in (("rot", self.d_rot, 1.0), ("pose", self.d_pose, 1.0), ("trans", self.d_trans, 1e-3)):
                full = np.array(h[k], np.float64, copy=True)
                full[self.ai] += (self.amask[self.a_in_v] * d[self.a_in_v]).double().cpu().numpy() * s
                out[k] = full
        return out


def _exact(seq, side, V_mm, idx):
    T_ow = metrics.inv_T(seq.T_wo[idx])
    hs = metrics.HandSurface(common.mano(side, False).faces)
    return metrics.penetration_metrics(hs, metrics.apply_T(T_ow, V_mm[idx]), seq.parts,
                                       None if seq.part_T is None else seq.part_T[idx])


def rectify(scene, device="cpu", dtype=torch.float32, iters=ITERS, rounds=ROUNDS, log=None, move_object=None):
    """-> dict side -> dict(params, per-frame report arrays); "_obj" -> object correction; "_seconds"."""
    t0 = time.time()
    move_object = MOVE_OBJECT if move_object is None else move_object
    grids = [sdfgrid.Grid(g, device) for g, _ in scene.objects]
    T = len(scene.seq.frame_idx)
    out, st = {}, {}
    for side, h in scene.hands.items():
        valid = h["valid"]
        m = common.mano(side, h["flat"])
        V0 = np.full((len(valid), 778, 3), np.nan)
        vi = np.where(valid)[0]
        V0[vi] = m(h["betas"], h["rot"][vi], h["pose"][vi], h["trans"][vi]) * 1000.0
        pen0, sd0 = np.full(len(valid), np.nan), np.full(len(valid), np.nan)
        if len(vi):
            e0 = _exact(scene.seq, side, V0, vi)
            pen0[vi], sd0[vi] = e0["pen_max_mm"], e0["min_sd_mm"]
        st[side] = {"m": m, "h": h, "V0": V0, "pen0": pen0, "sd0": sd0, "valid": valid,
                    "active": valid & (pen0 > FREEZE_MM), "unresolved": np.zeros(len(valid), bool)}
    any_active = np.zeros(T, bool)
    blocked = np.zeros(T, bool)          # a frozen hand touches the object: it must not move
    for s_ in st.values():
        any_active |= s_["active"]
        blocked |= s_["valid"] & ~s_["active"] & (s_["sd0"] < CONTACT_MM)
    obj = None
    if move_object and scene.obj is not None:
        movable = any_active & ~blocked & scene.obj["tracked"]
        if movable.any():
            obj = _ObjectDelta(scene, movable, device, dtype)
    probs = {s: _HandProblem(scene, s, st[s]["h"], st[s]["active"], grids, device, dtype, obj)
             for s in st if st[s]["active"].any()}
    groups = []
    for p in probs.values():
        groups += [{"params": [p.d_rot], "lr": LR["rot"]}, {"params": [p.d_pose], "lr": LR["pose"]},
                   {"params": [p.d_trans], "lr": LR["trans"]}]
    if obj is not None:
        groups += obj.params()
    opt = torch.optim.Adam(groups) if groups else None

    def evaluate():
        D = obj.numpy_D() if obj is not None else np.broadcast_to(np.eye(4), (T, 4, 4)).copy()
        seq1 = scene.obj["move"](D) if obj is not None else scene.seq
        res = {}
        for side, s_ in st.items():
            h, m, V0 = s_["h"], s_["m"], s_["V0"]
            if side in probs:
                P = probs[side].params(h)
                ai = probs[side].ai
                V1 = V0.copy()
                V1[ai] = m(h["betas"], P["rot"][ai], P["pose"][ai], P["trans"][ai]) * 1000.0
            else:
                P = {k: h[k].copy() for k in ("rot", "pose", "trans")}
                V1 = V0.copy()
            vi = np.where(s_["valid"])[0]
            pen1, sd1 = np.full(T, np.nan), np.full(T, np.nan)
            if len(vi):
                e1 = _exact(seq1, side, V1, vi)
                pen1[vi], sd1[vi] = e1["pen_max_mm"], e1["min_sd_mm"]
            res[side] = (P, V1, pen1, sd1)
        return D, seq1, res

    def run(n_iter, tag):
        parts = {}
        for _ in range(n_iter):
            opt.zero_grad()
            loss = 0.0
            for p in probs.values():
                l, pp = p.loss()
                loss = loss + l
                parts |= {f"{k}": v for k, v in pp.items()}
            if obj is not None:
                l, pp = obj.loss()
                loss = loss + l
                parts |= pp
            loss.backward()
            opt.step()
        D, seq1, res = evaluate()
        bad = {s: st[s]["active"] & ~st[s]["unresolved"] & (res[s][2] > FREEZE_MM) for s in probs}
        if log:
            log(f"{scene.name} {tag}: " + ", ".join(f"{s} active {int(st[s]['active'].sum())} unresolved "
                                                    f"{int(st[s]['unresolved'].sum())} >{FREEZE_MM}mm {int(bad[s].sum())}"
                                                    for s in probs)
                + (f", object movable {int(obj.mask.sum())}" if obj is not None else "")
                + " | " + ", ".join(f"{k} {float(v):.3g}" for k, v in parts.items()))
        return D, seq1, res, bad

    if probs:
        for r in range(rounds):
            D, seq1, res, bad = run(iters, f"round {r}")
            if not any(b.any() for b in bad.values()) or r == rounds - 1:
                break
            for s, b in bad.items():          # frames the first pass did not fix get a stronger penetration term
                p = probs[s]
                p.wf[torch.as_tensor(np.where(b[p.ai])[0], device=p.wf.device)] *= 10.0
        for f in range(FINAL_PASSES):
            if not any(b.any() for b in bad.values()):
                break
            # Still > FREEZE_MM: a local push cannot fix these (e.g. a finger passing through a thin part). They go
            # back to the untouched GT, flagged unresolved, and are treated like frozen frames (the object is held
            # there too), so neighbouring corrections fade into them; the rest is re-optimized.
            for s, b in bad.items():
                p = probs[s]
                st[s]["unresolved"] |= b
                with torch.no_grad():
                    rows = torch.as_tensor(np.searchsorted(p.vi, np.where(b)[0]), device=p.amask.device)
                    p.amask[rows] = 0.0
                    for d in (p.d_rot, p.d_pose, p.d_trans):
                        d[rows] = 0.0
                    p.wf[torch.as_tensor(np.where(b[p.ai])[0], device=p.wf.device)] = 0.0
                if obj is not None:
                    obj.lock(np.where(b)[0])
            D, seq1, res, bad = run(iters, f"final pass {f}")
        for s, b in bad.items():
            st[s]["unresolved"] |= b
            if obj is not None:
                obj.lock(np.where(b)[0])
        D, seq1, res = evaluate()                 # object held at unresolved frames: recompute with final D
    else:
        D, seq1, res = evaluate()
    for side, s_ in st.items():
        P, V1, pen1, sd1 = res[side]
        u = s_["unresolved"]
        for k in ("rot", "pose", "trans"):
            P[k][u] = s_["h"][k][u]
        V1[u] = s_["V0"][u]
        out[side] = {"params": P, "betas": s_["h"]["betas"], "V": V1, "valid": s_["valid"], "frozen": ~s_["active"],
                     "unresolved": u, "pen_before_mm": s_["pen0"], "min_sd_before_mm": s_["sd0"],
                     "pen_after_mm": pen1, "min_sd_after_mm": sd1}
    moved = obj.mask[:, 0].cpu().numpy() > 0 if obj is not None else np.zeros(T, bool)
    out["_obj"] = {"D": D, "movable": moved, "blocked": blocked, "seq": seq1}
    out["_seconds"] = time.time() - t0
    return out


def report(scene, res):
    """Per-frame table: penetration before/after, joint shift, pixel change per camera, jitter before/after."""
    import pandas as pd
    hs = metrics.HandSurface(common.mano("right", False).faces)
    seqs = {}
    for tag in ("before", "after"):
        hands = {s: (res[s]["V"] if tag == "after" else scene.seq.hands[s][0], res[s]["valid"]) for s in scene.hands}
        base = res["_obj"]["seq"] if tag == "after" else scene.seq
        sq = metrics.Sequence(scene.seq.dataset, scene.seq.name, scene.seq.fps, 1, scene.seq.frame_idx,
                              base.T_wo, base.parts, base.part_T, hands, {})
        seqs[tag] = metrics.evaluate(sq, hs).set_index(["hand", "frame"])
    # object correction: largest displacement of its surface samples (mm) and in pixels over the cameras
    o = scene.obj
    T = len(scene.seq.frame_idx)
    obj_mm, obj_px = np.zeros(T), np.zeros(T)
    if o is not None and res["_obj"]["movable"].any():
        D = res["_obj"]["D"]
        Binv = metrics.inv_T(np.where(o["tracked"][:, None, None], o["base"], np.eye(4)))
        X0 = metrics.apply_T(Binv, o["pts"])
        X1 = metrics.apply_T(Binv, metrics.apply_T(D, o["pts"]))
        obj_mm = np.linalg.norm(X1 - X0, axis=-1).max(1)
        for c in scene.cams:
            W2C = c["W2C"] if len(c["W2C"]) > 1 else np.repeat(c["W2C"], T, 0)
            uv = []
            for X in (X0, X1):
                Xc = np.einsum("nij,npj->npi", W2C[:, :3, :3], X / 1000.0) + W2C[:, None, :3, 3]
                q = Xc @ c["K"].T
                uv.append(q[..., :2] / q[..., 2:])
            w, hgt = c["size"]
            vis = (Xc[..., 2] > 0.05) & (uv[0][..., 0] >= 0) & (uv[0][..., 0] < w) & (uv[0][..., 1] >= 0) & (uv[0][..., 1] < hgt)
            obj_px = np.maximum(obj_px, np.where(vis, np.linalg.norm(uv[1] - uv[0], axis=-1), 0).max(1))
    rows = []
    for side in scene.hands:
        r = res[side]
        vi = np.where(r["valid"])[0]
        V0, V1 = scene.seq.hands[side][0], r["V"]
        Jr = common.mano(side, False).J_regressor
        J0 = np.concatenate([np.einsum("jv,nvc->njc", Jr, V0[vi]), V0[vi][:, TIPS]], 1)
        J1 = np.concatenate([np.einsum("jv,nvc->njc", Jr, V1[vi]), V1[vi][:, TIPS]], 1)
        dj = np.linalg.norm(J1 - J0, axis=-1)
        px = []
        for c in scene.cams:
            W2C = c["W2C"][vi] if len(c["W2C"]) > 1 else np.repeat(c["W2C"], len(vi), 0)
            uvs = []
            for J in (J0, J1):
                X = np.einsum("nij,npj->npi", W2C[:, :3, :3], J / 1000.0) + W2C[:, None, :3, 3]
                uv = X @ c["K"].T
                uvs.append(uv[..., :2] / uv[..., 2:])
            w, hgt = c["size"]
            vis = (X[..., 2] > 0.05) & (uvs[0][..., 0] >= 0) & (uvs[0][..., 0] < w) & (uvs[0][..., 1] >= 0) & (uvs[0][..., 1] < hgt)
            px.append(np.where(vis, np.linalg.norm(uvs[1] - uvs[0], axis=-1), np.nan))
        px = np.stack(px, 1)                                  # (n, cams, 21)
        with np.errstate(all="ignore"):
            px_max = np.nan_to_num(np.nanmax(px.reshape(len(vi), -1), 1), nan=0.0)
        b, a = seqs["before"].loc[side], seqs["after"].loc[side]
        rows.append(pd.DataFrame({
            "dataset": scene.dataset, "seq": scene.name, "hand": side, "frame": vi,
            "frozen": r["frozen"][vi], "unresolved": r["unresolved"][vi], "pen_before_mm": r["pen_before_mm"][vi], "pen_after_mm": r["pen_after_mm"][vi],
            "min_sd_before_mm": r["min_sd_before_mm"][vi], "min_sd_after_mm": r["min_sd_after_mm"][vi],
            "dj_mean_mm": dj.mean(1), "dj_max_mm": dj.max(1), "px_max": px_max,
            "rel_hf_before_mm": b.loc[vi, "rel_hf_mm"].to_numpy(), "rel_hf_after_mm": a.loc[vi, "rel_hf_mm"].to_numpy(),
            "rel_acc_before": b.loc[vi, "rel_acc_ms2"].to_numpy(), "rel_acc_after": a.loc[vi, "rel_acc_ms2"].to_numpy(),
            "obj_moved": res["_obj"]["movable"][vi], "obj_shift_mm": obj_mm[vi], "obj_px_max": obj_px[vi],
        }))
    return pd.concat(rows, ignore_index=True)
