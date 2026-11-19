"""GPU Levenberg-Marquardt bundle adjustment with Schur complement (3D points eliminated).

Same model as refine.BA (scipy TRF/LSMR): per-frame pose (w, dC; frame 0 fixed), 3D points, log focal,
k1/k2 (libmv Polynomial), weak pose / focal / distortion / point priors, IRLS robust weights.
Additions:
- zoom=True: one log focal per frame (prior to Pi4's per-frame focal + smoothness between frames).
- Cost of a step ~ sum over points of (observations per point)^2 for the reduced camera system plus one
  dense Cholesky of the camera system (size ~6 x frames): it scales to many thousands of tracks.

State is a dict of tensors; refine.py talks to it through the same small interface as refine.BA:
x0, set_obs, obs_errors, solve, unpack, f_frames.
"""
import time

import numpy as np
import torch


class LMBA:
    def __init__(self, c2w, f0, plate, obs, X0, prior_sigma, device, focal_sigma=0.3, k_sigma=0.5,
                 zoom=False, f_init=None, zoom_smooth=0.01, max_pairs_chunk=400_000):
        from refine import _project
        self.dev = device
        self.T = c2w.shape[0]
        self.W, self.H = plate
        self.cx, self.cy = (self.W - 1) / 2, (self.H - 1) / 2
        self.f0 = float(f0)
        self.R0 = torch.tensor(np.transpose(c2w[:, :3, :3], (0, 2, 1)), dtype=torch.float64, device=device)
        self.C0 = torch.tensor(c2w[:, :3, 3], dtype=torch.float64, device=device)
        self.prior_sigma = float(prior_sigma)
        self.focal_sigma = float(focal_sigma)
        self.k_sigma = float(k_sigma)
        self.zoom = bool(zoom)
        self.zoom_smooth = float(zoom_smooth)
        self.chunk = int(max_pairs_chunk)
        T = self.T
        self.nf = T if self.zoom else 1
        if self.zoom and f_init is not None:
            lf = np.log(np.asarray(f_init, np.float64) / self.f0)
        else:
            lf = np.zeros(self.nf)
        self.logf_prior = torch.tensor(lf, dtype=torch.float64, device=device)
        # camera-side layout: [k1, k2, logf (nf), w (T-1)x3, dC (T-1)x3] + 1 dummy column (fixed frame 0)
        self.o_f = 2
        self.o_w = 2 + self.nf
        self.o_c = self.o_w + 3 * (T - 1)
        self.Nc = self.o_c + 3 * (T - 1)
        self.nP = X0.shape[0]
        self.X_init = torch.tensor(X0, dtype=torch.float64, device=device)
        first = np.full(self.nP, -1)
        fr_np, pt_np = obs[:, 0].astype(int), obs[:, 1].astype(int)
        for j in np.argsort(fr_np, kind="stable")[::-1]:
            first[pt_np[j]] = fr_np[j]
        dist = np.linalg.norm(X0 - c2w[np.clip(first, 0, None), :3, 3], axis=1)
        self.X_sigma = torch.tensor(np.where(first >= 0, np.maximum(dist, 1e-3), 1.0), dtype=torch.float64,
                                    device=device)
        self.x0 = dict(k=torch.zeros(2, dtype=torch.float64, device=device), logf=self.logf_prior.clone(),
                       w=torch.zeros(T, 3, dtype=torch.float64, device=device),
                       dC=torch.zeros(T, 3, dtype=torch.float64, device=device), X=self.X_init.clone())
        jac = torch.func.jacfwd(_project, argnums=(0, 1, 2, 3))
        self._jac = torch.func.vmap(jac, in_dims=(0, 0, 0, 0, 0, None, None, None))
        self._proj = torch.func.vmap(_project, in_dims=(0, 0, 0, 0, 0, None, None, None))
        self.set_obs(obs)

    # ------------------------------------------------------------------ observations / structure
    def set_obs(self, obs):
        d = self.dev
        self.fr = torch.tensor(obs[:, 0].astype(np.int64), device=d)
        self.pt = torch.tensor(obs[:, 1].astype(np.int64), device=d)
        self.uv = torch.tensor(obs[:, 2:4], dtype=torch.float64, device=d)
        self.nobs = len(obs)
        self.wsq = torch.ones(self.nobs, dtype=torch.float64, device=d)
        fidx = (self.fr if self.zoom else torch.zeros_like(self.fr)) + self.o_f
        fm1 = (self.fr - 1).clamp_min(0)
        a3 = torch.arange(3, device=d)
        wc = torch.where(self.fr[:, None] > 0, self.o_w + 3 * fm1[:, None] + a3, self.Nc)
        cc = torch.where(self.fr[:, None] > 0, self.o_c + 3 * fm1[:, None] + a3, self.Nc)
        self.cols = torch.cat([fidx[:, None], torch.zeros_like(fidx)[:, None], torch.ones_like(fidx)[:, None],
                               wc, cc], 1)                                           # (n, 9)
        self.movable = (self.fr > 0).to(torch.float64)
        # all observation pairs sharing a point (for the Schur complement)
        so = torch.argsort(self.pt, stable=True)
        m = torch.bincount(self.pt, minlength=self.nP)
        s = torch.cumsum(m, 0) - m
        rep = m[self.pt[so]]
        a_pos = torch.repeat_interleave(torch.arange(self.nobs, device=d), rep)
        blk_start = torch.repeat_interleave(torch.cumsum(rep, 0) - rep, rep)
        off = torch.arange(len(a_pos), device=d) - blk_start
        b_pos = s[self.pt[so[a_pos]]] + off
        self.pa, self.pb = so[a_pos], so[b_pos]

    # ------------------------------------------------------------------ model
    def _intr_obs(self, st):
        lf = st["logf"][self.fr] if self.zoom else st["logf"][0].expand(self.nobs)
        return torch.stack([lf, st["k"][0].expand(self.nobs), st["k"][1].expand(self.nobs)], 1)

    def _C(self, st):
        return self.C0 + st["dC"]

    def _uv(self, st):
        return self._proj(self._intr_obs(st), st["w"][self.fr], self._C(st)[self.fr], st["X"][self.pt],
                          self.R0[self.fr], self.f0, self.cx, self.cy)

    def cost(self, st):
        r = (self._uv(st) - self.uv) * self.wsq[:, None]
        c = (r * r).sum()
        c = c + ((st["dC"][1:] / self.prior_sigma) ** 2).sum()
        c = c + ((st["k"] / self.k_sigma) ** 2).sum()
        c = c + (((st["logf"] - self.logf_prior) / self.focal_sigma) ** 2).sum()
        if self.zoom:
            c = c + (((st["logf"][1:] - st["logf"][:-1]) / self.zoom_smooth) ** 2).sum()
        c = c + ((((st["X"] - self.X_init) / self.X_sigma[:, None])) ** 2).sum()
        return float(c)

    def obs_errors(self, st):
        return torch.linalg.norm(self._uv(st) - self.uv, dim=1).cpu().numpy()

    # ------------------------------------------------------------------ one LM linearisation
    def _linearise(self, st):
        n, N1 = self.nobs, self.Nc + 1
        intr = self._intr_obs(st)
        Ji, Jw, Jc, Jx = self._jac(intr, st["w"][self.fr], self._C(st)[self.fr], st["X"][self.pt],
                                   self.R0[self.fr], self.f0, self.cx, self.cy)
        ws = self.wsq[:, None, None]
        mv = self.movable[:, None, None]
        Jcam = torch.cat([Ji, Jw * mv, Jc * mv], 2) * ws                      # (n, 2, 9)
        Jp = Jx * ws                                                         # (n, 2, 3)
        r = (self._proj(intr, st["w"][self.fr], self._C(st)[self.fr], st["X"][self.pt], self.R0[self.fr],
                        self.f0, self.cx, self.cy) - self.uv) * self.wsq[:, None]
        Hcc = torch.zeros(N1 * N1, dtype=torch.float64, device=self.dev)
        idx = (self.cols[:, :, None] * N1 + self.cols[:, None, :]).reshape(-1)
        Hcc.index_add_(0, idx, torch.einsum("nki,nkj->nij", Jcam, Jcam).reshape(-1))
        Hcc = Hcc.view(N1, N1)
        gc = torch.zeros(N1, dtype=torch.float64, device=self.dev)
        gc.index_add_(0, self.cols.reshape(-1), torch.einsum("nki,nk->ni", Jcam, r).reshape(-1))
        Hpp = torch.zeros(self.nP, 3, 3, dtype=torch.float64, device=self.dev)
        Hpp.index_add_(0, self.pt, torch.einsum("nki,nkj->nij", Jp, Jp))
        gp = torch.zeros(self.nP, 3, dtype=torch.float64, device=self.dev)
        gp.index_add_(0, self.pt, torch.einsum("nki,nk->ni", Jp, r))
        Hcp = torch.einsum("nki,nkj->nij", Jcam, Jp)                         # (n, 9, 3)
        # priors
        ip = 1.0 / self.X_sigma ** 2
        Hpp += torch.diag_embed(ip[:, None].expand(-1, 3))
        gp += (st["X"] - self.X_init) * ip[:, None]
        dg = torch.zeros(N1, dtype=torch.float64, device=self.dev)
        cidx = torch.arange(self.o_c, self.Nc, device=self.dev)
        dg[cidx] += 1 / self.prior_sigma ** 2
        gc[cidx] += st["dC"][1:].reshape(-1) / self.prior_sigma ** 2
        dg[0:2] += 1 / self.k_sigma ** 2
        gc[0:2] += st["k"] / self.k_sigma ** 2
        fidx = torch.arange(self.o_f, self.o_f + self.nf, device=self.dev)
        dg[fidx] += 1 / self.focal_sigma ** 2
        gc[fidx] += (st["logf"] - self.logf_prior) / self.focal_sigma ** 2
        Hcc += torch.diag(dg)
        if self.zoom and self.T > 1:
            a, b = fidx[:-1], fidx[1:]
            q = 1 / self.zoom_smooth ** 2
            dlf = (st["logf"][1:] - st["logf"][:-1]) * q
            Hcc[a, a] += q
            Hcc[b, b] += q
            Hcc[a, b] -= q
            Hcc[b, a] -= q
            gc.index_add_(0, a, -dlf)
            gc.index_add_(0, b, dlf)
        Hcc[self.Nc, :] = 0
        Hcc[:, self.Nc] = 0
        Hcc[self.Nc, self.Nc] = 1.0
        gc[self.Nc] = 0
        return Hcc, gc, Hpp, gp, Hcp

    def _step(self, lin, lam):
        Hcc, gc, Hpp, gp, Hcp = lin
        N1 = self.Nc + 1
        S = Hcc + torch.diag(lam * torch.diagonal(Hcc) + 1e-9)
        Hd = Hpp + torch.diag_embed(lam * torch.diagonal(Hpp, dim1=1, dim2=2))
        Hinv = torch.linalg.inv(Hd)                                          # (nP, 3, 3)
        Y = Hcp @ Hinv[self.pt]                                              # (n, 9, 3)
        Sflat = S.reshape(-1)
        for i in range(0, len(self.pa), self.chunk):
            pa, pb = self.pa[i:i + self.chunk], self.pb[i:i + self.chunk]
            blk = Y[pa] @ Hcp[pb].transpose(1, 2)                            # (k, 9, 9)
            idx = (self.cols[pa][:, :, None] * N1 + self.cols[pb][:, None, :]).reshape(-1)
            Sflat.index_add_(0, idx, -blk.reshape(-1))
        S = Sflat.view(N1, N1)
        S[self.Nc, :] = 0
        S[:, self.Nc] = 0
        S[self.Nc, self.Nc] = 1.0
        b = -gc
        b.index_add_(0, self.cols.reshape(-1), torch.einsum("nij,nj->ni", Y, gp[self.pt]).reshape(-1))
        b[self.Nc] = 0
        S = 0.5 * (S + S.T)
        L, info = torch.linalg.cholesky_ex(S)
        if int(info) != 0:
            return None
        dc = torch.cholesky_solve(b[:, None], L)[:, 0]
        dc[self.Nc] = 0
        t = torch.zeros(self.nP, 3, dtype=torch.float64, device=self.dev)
        t.index_add_(0, self.pt, torch.einsum("nij,ni->nj", Hcp, dc[self.cols]))
        dp = torch.einsum("pij,pj->pi", Hinv, -gp - t)
        # predicted decrease of the quadratic model (for the gain ratio)
        pred = float(-(gc @ dc) - (gp * dp).sum())
        return dc, dp, pred

    def _apply(self, st, dc, dp):
        T = self.T
        w = st["w"].clone()
        dC = st["dC"].clone()
        w[1:] += dc[self.o_w:self.o_c].view(T - 1, 3)
        dC[1:] += dc[self.o_c:self.Nc].view(T - 1, 3)
        return dict(k=st["k"] + dc[0:2], logf=st["logf"] + dc[self.o_f:self.o_f + self.nf], w=w, dC=dC,
                    X=st["X"] + dp)

    def lm(self, st, max_iter=60, tol=1e-10):
        lam, c = 1e-3, self.cost(st)
        it = 0
        for it in range(max_iter):
            lin = self._linearise(st)
            accepted = False
            for _ in range(12):
                stp = self._step(lin, lam)
                if stp is None:
                    lam *= 10
                    continue
                dc, dp, pred = stp
                new = self._apply(st, dc, dp)
                c_new = self.cost(new)
                rho = (c - c_new) / max(pred, 1e-30)
                if c_new < c and np.isfinite(c_new):
                    rel = (c - c_new) / max(c, 1e-30)
                    st, c = new, c_new
                    lam = max(lam * max(1 / 3, 1 - (2 * rho - 1) ** 3), 1e-9)
                    accepted = True
                    break
                lam *= 4
            if not accepted or rel < tol:
                break
        return st, c, it + 1

    # ------------------------------------------------------------------ robust solve (IRLS around LM)
    def solve(self, st, max_nfev, log, loss="huber", f_scale=1.5, irls_iters=4, on_iter=None):
        t0 = time.time()
        iters = 0
        for it in range(irls_iters):
            e = self.obs_errors(st)
            if on_iter:
                on_iter(it, irls_iters, float(np.median(e[np.isfinite(e)])))
            e = np.where(np.isfinite(e), e, 1e6)
            if loss == "huber":
                wgt = np.where(e <= f_scale, 1.0, f_scale / np.maximum(e, 1e-12))
            elif loss == "cauchy":
                wgt = 1.0 / (1.0 + (e / f_scale) ** 2)
            else:
                wgt = np.ones_like(e)
            self.wsq = torch.tensor(np.sqrt(wgt), dtype=torch.float64, device=self.dev)
            st, _, k = self.lm(st, max_iter=max_nfev)
            iters += k
        self.wsq = torch.ones(self.nobs, dtype=torch.float64, device=self.dev)
        log(f"[refine] BA-LM ({loss}, {f_scale}px, IRLS x{irls_iters}): {self.nobs} obs, {self.nP} pts, "
            f"{self.Nc} camera params{' (zoom)' if self.zoom else ''}, {len(self.pa)} Schur pairs, "
            f"{iters} LM iterations, {time.time() - t0:.1f}s")
        return st

    def unpack(self, st):
        lf = st["logf"]
        intr = torch.cat([lf[:1], st["k"]]) if not self.zoom else None
        return intr, st["w"], self._C(st), st["X"], st["dC"]

    def f_frames(self, st):
        lf = st["logf"].cpu().numpy()
        return self.f0 * np.exp(lf if self.zoom else np.full(self.T, lf[0]))

    def k12(self, st):
        return float(st["k"][0]), float(st["k"][1])
