"""Phase 2: plate-resolution refinement of a Pi4 solve.

1. Decode the same frames Pi4 saw, at full resolution.
2. KLT-track corners through the shot (re-seeded every frame, forward-backward checked),
   with Pi4's motion mask keeping moving objects out.
3. Initialise each track's 3D point from Pi4's depth at its first observation.
4. Robust bundle adjustment (scipy trust-region + analytic sparse Jacobian from torch):
   all poses (frame 0 fixed), all points, shared focal, k1, k2 (Blender "Polynomial" model).
   Weak priors on Pi4's camera centres fix scale and stabilise low-parallax shots.

Pixel convention for everything here: OpenCV (pixel centres at integers), principal point
fixed at the image centre ((W-1)/2, (H-1)/2).
"""
import os
import time

import cv2
import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import least_squares

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")  # same filter as pi4.io.load_images


# --------------------------------------------------------------------------- frames
def iter_frames(src, frame_indices):
    """Yield (index, BGR frame) at full resolution for exactly the frames Pi4 used."""
    wanted = set(int(i) for i in frame_indices)
    last = max(wanted)
    if os.path.isdir(src):
        files = sorted(f for f in os.listdir(src) if os.path.splitext(f)[1].lower() in IMG_EXT)
        for i in sorted(wanted):
            yield i, cv2.imread(os.path.join(src, files[i]), cv2.IMREAD_COLOR)
        return
    cap = cv2.VideoCapture(src)
    i = 0
    try:
        while i <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                yield i, frame
            i += 1
    finally:
        cap.release()


# --------------------------------------------------------------------------- tracking
def klt_tracks(src, frame_indices, dyn, max_active=600, min_len=4, fb_thresh=0.7, quality=0.001, log=print,
               progress=None):
    """Returns obs (N,4)[frame_pos, track_id, u, v], colors per track (M,3 uint8 RGB), plate (W,H)."""
    T, Hm, Wm = dyn.shape
    obs, colors = [], []
    prev = None
    pts = np.zeros((0, 2), np.float32)
    ids = np.zeros((0,), np.int64)
    next_id = 0
    lk = dict(winSize=(21, 21), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    plate = None
    for k, (_, bgr) in enumerate(iter_frames(src, frame_indices)):
        if k >= T:
            break
        if progress and k % 5 == 0:
            progress("klt", k / T, f"KLT tracking frame {k}/{T} ({len(pts)} active tracks)")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        plate = (W, H)
        margin = max(8, int(0.01 * max(W, H)))
        static = cv2.resize((dyn[k] < 0.5).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        static = cv2.erode(static, np.ones((15, 15), np.uint8))  # keep away from moving objects
        static[:margin], static[-margin:], static[:, :margin], static[:, -margin:] = 0, 0, 0, 0

        if prev is not None and len(pts):
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, **lk)
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, p1, None, **lk)
            fb = np.linalg.norm(p0r - pts, axis=1)
            ui, vi = np.round(p1[:, 0]).astype(int), np.round(p1[:, 1]).astype(int)
            inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            ok = (st1[:, 0] == 1) & (st2[:, 0] == 1) & (fb < fb_thresh) & inside
            ok[inside] &= static[vi[inside], ui[inside]] > 0
            pts, ids = p1[ok], ids[ok]
            obs.extend((k, int(t), float(u), float(v)) for t, (u, v) in zip(ids, pts))

        # Re-seed in empty static regions.
        need = max_active - len(pts)
        if need > max_active // 10:
            min_dist = max(6.0, 0.5 * np.sqrt(W * H / max_active))
            mask = static.copy()
            for u, v in pts:
                cv2.circle(mask, (int(u), int(v)), int(min_dist), 0, -1)
            new = cv2.goodFeaturesToTrack(gray, maxCorners=need, qualityLevel=quality,
                                          minDistance=min_dist, mask=mask, blockSize=7)
            if new is not None:
                new = cv2.cornerSubPix(gray, new.astype(np.float32), (5, 5), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
                new = new.reshape(-1, 2)
                new_ids = np.arange(next_id, next_id + len(new))
                next_id += len(new)
                rgb = bgr[np.clip(new[:, 1].round().astype(int), 0, H - 1),
                          np.clip(new[:, 0].round().astype(int), 0, W - 1)][:, ::-1]
                colors.extend(rgb)
                obs.extend((k, int(t), float(u), float(v)) for t, (u, v) in zip(new_ids, new))
                pts = np.vstack([pts, new]).astype(np.float32)
                ids = np.concatenate([ids, new_ids])
        prev = gray
    obs = np.array(obs, np.float64).reshape(-1, 4)
    colors = np.array(colors, np.uint8).reshape(-1, 3)
    # Drop short tracks, renumber.
    tid = obs[:, 1].astype(np.int64)
    counts = np.bincount(tid, minlength=len(colors))
    keep_t = counts >= min_len
    obs = obs[keep_t[tid]]
    remap = -np.ones(len(colors), np.int64)
    remap[keep_t] = np.arange(keep_t.sum())
    obs[:, 1] = remap[obs[:, 1].astype(np.int64)]
    log(f"[refine] KLT: {int(keep_t.sum())} tracks, {len(obs)} observations, plate {plate}")
    return obs, colors[keep_t], plate


# --------------------------------------------------------------------------- geometry
def _skew(w):
    z = torch.zeros_like(w[..., 0])
    return torch.stack([torch.stack([z, -w[..., 2], w[..., 1]], -1),
                        torch.stack([w[..., 2], z, -w[..., 0]], -1),
                        torch.stack([-w[..., 1], w[..., 0], z], -1)], -2)


def _expmap(w):
    th2 = (w * w).sum(-1, keepdim=True)[..., None]
    small = th2 < 1e-10
    th = torch.sqrt(torch.where(small, torch.ones_like(th2), th2))
    A = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    B = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / (th * th))
    K = _skew(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand_as(K)
    return eye + A * K + B * (K @ K)


def _project(intr, w, C, X, R0, f0, cx, cy):
    """Single observation. intr = (log f/f0, k1, k2); pose = (w, C) world->cam R = exp(w) R0."""
    f = f0 * torch.exp(intr[0])
    R = _expmap(w) @ R0
    Xc = R @ (X - C)
    z = Xc[2]
    x, y = Xc[0] / z, Xc[1] / z
    r2 = x * x + y * y
    d = 1 + intr[1] * r2 + intr[2] * r2 * r2
    return torch.stack([f * x * d + cx, f * y * d + cy])


class BA:
    def __init__(self, c2w, f0, plate, obs, X0, prior_sigma, device, focal_sigma=0.3, k_sigma=0.5):
        self.dev = device
        self.T = c2w.shape[0]
        self.W, self.H = plate
        self.cx, self.cy = (self.W - 1) / 2, (self.H - 1) / 2
        self.f0 = float(f0)
        self.R0 = torch.tensor(np.transpose(c2w[:, :3, :3], (0, 2, 1)), dtype=torch.float64, device=device)
        self.C0 = c2w[:, :3, 3].copy()
        self.prior_sigma = prior_sigma
        self.focal_sigma = focal_sigma
        self.k_sigma = k_sigma  # tiny = distortion locked at 0 (rectilinear / cinema lenses)
        self.set_obs(obs)
        self.nP = X0.shape[0]
        # Weak per-point prior (sigma = the point's own depth from its first camera) so
        # depth-ambiguous points on low-parallax rays cannot run off to infinity.
        first = np.full(self.nP, -1)
        fr_np, pt_np = obs[:, 0].astype(int), obs[:, 1].astype(int)
        for j in np.argsort(fr_np, kind="stable")[::-1]:
            first[pt_np[j]] = fr_np[j]
        dist = np.linalg.norm(X0 - self.C0[np.clip(first, 0, None)], axis=1)
        self.X_init = torch.tensor(X0, dtype=torch.float64, device=device)
        self.X_sigma = torch.tensor(np.where(first >= 0, np.maximum(dist, 1e-3), 1.0), dtype=torch.float64,
                                    device=device)
        # layout: [intr 3 | w (T-1)*3 | dC (T-1)*3 | X nP*3]; frame 0 fixed.
        self.x0 = np.concatenate([np.zeros(3), np.zeros(6 * (self.T - 1)), X0.reshape(-1)])
        self.o_w = 3
        self.o_c = 3 + 3 * (self.T - 1)
        self.o_x = 3 + 6 * (self.T - 1)
        jac = torch.func.jacfwd(_project, argnums=(0, 1, 2, 3))
        self._jac = torch.func.vmap(jac, in_dims=(None, 0, 0, 0, 0, None, None, None))
        self._proj = torch.func.vmap(_project, in_dims=(None, 0, 0, 0, 0, None, None, None))

    def set_obs(self, obs):
        self.fr = torch.tensor(obs[:, 0].astype(np.int64), device=self.dev)
        self.pt = torch.tensor(obs[:, 1].astype(np.int64), device=self.dev)
        self.uv = torch.tensor(obs[:, 2:4], dtype=torch.float64, device=self.dev)
        self.nobs = len(obs)
        self.wsq = torch.ones(self.nobs, dtype=torch.float64, device=self.dev)  # sqrt IRLS weights

    def unpack(self, x):
        x = torch.as_tensor(x, dtype=torch.float64, device=self.dev)
        intr = x[:3]
        w = torch.cat([torch.zeros(1, 3, dtype=x.dtype, device=self.dev), x[self.o_w:self.o_c].view(-1, 3)])
        dC = torch.cat([torch.zeros(1, 3, dtype=x.dtype, device=self.dev), x[self.o_c:self.o_x].view(-1, 3)])
        C = torch.as_tensor(self.C0, device=self.dev) + dC
        X = x[self.o_x:].view(-1, 3)
        return intr, w, C, X, dC

    def residuals(self, x):
        intr, w, C, X, dC = self.unpack(x)
        uv = self._proj(intr, w[self.fr], C[self.fr], X[self.pt], self.R0[self.fr], self.f0, self.cx, self.cy)
        r_obs = ((uv - self.uv) * self.wsq[:, None]).reshape(-1)
        r_prior = (dC[1:] / self.prior_sigma).reshape(-1)
        r_intr = torch.stack([intr[0] / self.focal_sigma, intr[1] / self.k_sigma, intr[2] / self.k_sigma])
        r_pts = ((X - self.X_init) / self.X_sigma[:, None]).reshape(-1)
        return torch.cat([r_obs, r_prior, r_intr, r_pts])

    def fun(self, x):
        return self.residuals(x).cpu().numpy()

    def jac(self, x):
        intr, w, C, X, _ = self.unpack(x)
        Ji, Jw, Jc, Jx = self._jac(intr, w[self.fr], C[self.fr], X[self.pt], self.R0[self.fr],
                                   self.f0, self.cx, self.cy)
        n = self.nobs
        ws = self.wsq[:, None, None]
        Ji, Jw, Jc, Jx = Ji * ws, Jw * ws, Jc * ws, Jx * ws
        rows = torch.arange(2 * n, device=self.dev).view(n, 2)
        blocks, cols = [Ji], [torch.arange(3, device=self.dev).expand(n, 3)]
        fm1 = (self.fr - 1).clamp_min(0)
        movable = (self.fr > 0).to(torch.float64)[:, None, None]
        blocks += [Jw * movable, Jc * movable]
        cols += [self.o_w + 3 * fm1[:, None] + torch.arange(3, device=self.dev),
                 self.o_c + 3 * fm1[:, None] + torch.arange(3, device=self.dev)]
        blocks.append(Jx)
        cols.append(self.o_x + 3 * self.pt[:, None] + torch.arange(3, device=self.dev))
        R, Cc, V = [], [], []
        for b, c in zip(blocks, cols):  # b: (n,2,k), c: (n,k)
            k = c.shape[1]
            R.append(rows[:, :, None].expand(n, 2, k).reshape(-1))
            Cc.append(c[:, None, :].expand(n, 2, k).reshape(-1))
            V.append(b.reshape(-1))
        R, Cc, V = torch.cat(R), torch.cat(Cc), torch.cat(V)
        nz = V != 0
        R, Cc, V = R[nz].cpu().numpy(), Cc[nz].cpu().numpy(), V[nz].cpu().numpy()
        # priors
        npri = 3 * (self.T - 1)
        pr = 2 * n + np.arange(npri)
        npts = 3 * self.nP
        base = 2 * n + npri + 3
        R = np.concatenate([R, pr, 2 * n + npri + np.arange(3), base + np.arange(npts)])
        Cc = np.concatenate([Cc, self.o_c + np.arange(npri), np.arange(3), self.o_x + np.arange(npts)])
        V = np.concatenate([V, np.full(npri, 1 / self.prior_sigma), [1 / self.focal_sigma, 1 / self.k_sigma, 1 / self.k_sigma],
                            (1 / self.X_sigma).repeat_interleave(3).cpu().numpy()])
        return sp.csr_matrix((V, (R, Cc)), shape=(base + npts, len(x)))

    def obs_errors(self, x):
        """Unweighted reprojection error (pixels) per observation."""
        intr, w, C, X, _ = self.unpack(x)
        uv = self._proj(intr, w[self.fr], C[self.fr], X[self.pt], self.R0[self.fr], self.f0, self.cx, self.cy)
        return torch.linalg.norm(uv - self.uv, dim=1).cpu().numpy()

    def solve(self, x, max_nfev, log, loss="huber", f_scale=1.5, irls_iters=4, on_iter=None):
        """Robust BA by iteratively reweighted least squares around scipy's linear-loss TRF
        (scipy's built-in robust losses stall on this problem). on_iter(i, irls_iters, median_err_px)."""
        t0 = time.time()
        nfev = 0
        for it in range(irls_iters):
            if on_iter:
                e_now = self.obs_errors(x)
                on_iter(it, irls_iters, float(np.median(e_now[np.isfinite(e_now)])))
            e = self.obs_errors(x)
            e = np.where(np.isfinite(e), e, 1e6)
            if loss == "huber":
                wgt = np.where(e <= f_scale, 1.0, f_scale / np.maximum(e, 1e-12))
            elif loss == "cauchy":
                wgt = 1.0 / (1.0 + (e / f_scale) ** 2)
            else:
                wgt = np.ones_like(e)
            self.wsq = torch.tensor(np.sqrt(wgt), dtype=torch.float64, device=self.dev)
            res = least_squares(self.fun, x, jac=self.jac, method="trf", tr_solver="lsmr",
                                x_scale="jac", max_nfev=max_nfev, ftol=1e-8, xtol=1e-10, gtol=1e-10)
            x, nfev = res.x, nfev + res.nfev
        self.wsq = torch.ones(self.nobs, dtype=torch.float64, device=self.dev)
        log(f"[refine] BA ({loss}, {f_scale}px, IRLS x{irls_iters}): {self.nobs} obs, {len(x)} params, "
            f"{nfev} evals, {time.time() - t0:.1f}s")
        return x


# --------------------------------------------------------------------------- driver
def _stats(e):
    return dict(median=float(np.median(e)), mean=float(np.mean(e)), p90=float(np.percentile(e, 90)),
                rms=float(np.sqrt(np.mean(e ** 2))))


def refine(src, result, cache, log=print, max_active=600, prior_frac=0.1, max_nfev=50, quality=0.001,
           focal_sigma=0.3, progress=None, solve_distortion=True):
    """result: dict from result.npz; cache: dict(depth [T,Hm,Wm] metric, dyn [T,Hm,Wm]).
    progress(stage, frac, msg) is called with stages 'klt' and 'ba'."""
    prog = progress or (lambda *a, **k: None)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c2w = result["c2w"].astype(np.float64)
    T = c2w.shape[0]
    Wm, Hm = (int(v) for v in result["model_size"])
    depth, dyn = cache["depth"].astype(np.float32), cache["dyn"].astype(np.float32) / 255.0

    obs, colors, plate = klt_tracks(src, result["frame_indices"], dyn, max_active=max_active, quality=quality,
                                    log=log, progress=progress)
    # Slow shots give very long tracks; neighbouring observations add little but cost a lot of solver time.
    # Keep at most MAX_OBS evenly spaced observations per track (first and last always kept).
    MAX_OBS = 40
    o_tid = obs[:, 1].astype(np.int64)
    order = np.lexsort((obs[:, 0], o_tid))
    obs, o_tid = obs[order], o_tid[order]
    starts = np.r_[0, np.nonzero(np.diff(o_tid))[0] + 1]
    lens = np.diff(np.r_[starts, len(obs)])
    keep = np.ones(len(obs), bool)
    for st, n in zip(starts[lens > MAX_OBS], lens[lens > MAX_OBS]):
        keep[st:st + n] = False
        keep[st + np.unique(np.round(np.linspace(0, n - 1, MAX_OBS)).astype(int))] = True
    if (~keep).any():
        log(f"[refine] long tracks thinned: {len(obs)} -> {int(keep.sum())} observations (max {MAX_OBS}/track)")
    obs = obs[keep]
    prog("ba", 0.0, "initialising 3D points")
    W, H = plate
    sx = W / Wm
    f0 = float(np.median(result["fx"])) * sx
    cx, cy = (W - 1) / 2, (H - 1) / 2

    # Init 3D points from Pi4 depth at each track's first observation.
    tid = obs[:, 1].astype(np.int64)
    first = np.full(len(colors), -1)
    order = np.argsort(obs[:, 0], kind="stable")
    for j in order[::-1]:
        first[tid[j]] = j
    fo = obs[first]
    fr = fo[:, 0].astype(int)
    um = np.clip((fo[:, 2] + 0.5) * Wm / W - 0.5, 0, Wm - 1)
    vm = np.clip((fo[:, 3] + 0.5) * Hm / H - 0.5, 0, Hm - 1)
    Z = depth[fr, vm.round().astype(int), um.round().astype(int)]
    xn, yn = (fo[:, 2] - cx) / f0, (fo[:, 3] - cy) / f0
    Xc = np.stack([xn * Z, yn * Z, Z], 1)
    X0 = np.einsum("nij,nj->ni", c2w[fr, :3, :3], Xc) + c2w[fr, :3, 3]
    # Better init where there is parallax: midpoint triangulation over all observations
    # with Pi4's poses (min eigenvalue of sum(I - dd^T) per observation ~ squared ray angle).
    ofr = obs[:, 0].astype(int)
    d = np.stack([(obs[:, 2] - cx) / f0, (obs[:, 3] - cy) / f0, np.ones(len(obs))], 1)
    d = np.einsum("nij,nj->ni", c2w[ofr, :3, :3], d)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    Pm = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    A = np.zeros((len(colors), 3, 3)); b = np.zeros((len(colors), 3))
    np.add.at(A, tid, Pm)
    np.add.at(b, tid, np.einsum("nij,nj->ni", Pm, c2w[ofr, :3, 3]))
    nobs_t = np.bincount(tid, minlength=len(colors))[:, None]
    lam_min = np.linalg.eigvalsh(A)[:, 0] / nobs_t[:, 0]
    tri = lam_min > (np.deg2rad(1.0) ** 2) / 4
    Xt = np.linalg.solve(A[tri] + 1e-9 * np.eye(3), b[tri][..., None])[..., 0]
    X0[tri] = Xt
    good_pt = np.isfinite(X0).all(1) & (Z > 0)
    X0[~good_pt] = 0.0  # unused (their observations are dropped) but must be finite
    log(f"[refine] init: {int(tri.sum())} triangulated, {int((~tri & good_pt).sum())} from Pi4 depth")

    med_depth = float(np.median(depth[depth > 0]))
    obs_g = obs[good_pt[tid]]
    ba = BA(c2w, f0, plate, obs_g, X0, prior_sigma=prior_frac * med_depth, device=dev,
            focal_sigma=focal_sigma, k_sigma=0.5 if solve_distortion else 1e-5)
    x = ba.x0.copy()
    e_init = ba.obs_errors(x)
    log(f"[refine] init reprojection: median {np.median(e_init):.2f}px p90 {np.percentile(e_init, 90):.1f}px")

    # Gross-outlier pre-filter (points behind cameras, bad depth init, KLT drift).
    thr0 = max(0.03 * max(W, H), 5 * float(np.median(e_init)))
    ok0 = np.isfinite(e_init) & (e_init < thr0)
    obs_g = obs_g[ok0]
    ba.set_obs(obs_g)
    # Stage 1: heavy-tailed loss to pull everything in. Stage 2: reject, Huber.
    def it_cb(stage_i, label):
        return lambda i, n, med: prog("ba", (stage_i * n + i) / (2 * n),
                                      f"bundle adjustment {label} {i + 1}/{n}, median error {med:.2f}px")

    x = ba.solve(x, max_nfev, log, loss="cauchy", f_scale=3.0, on_iter=it_cb(0, "robust pass"))
    e1 = ba.obs_errors(x)
    thr = max(2.0, 3 * float(np.median(e1)))
    kept_obs = obs_g[e1 < thr]
    ba.set_obs(kept_obs)
    log(f"[refine] outlier rejection > {thr:.2f}px: kept {len(kept_obs)}/{len(obs_g)} observations")
    x = ba.solve(x, max_nfev, log, loss="huber", f_scale=1.0, on_iter=it_cb(1, "final pass"))
    prog("ba", 1.0, "bundle adjustment done")
    e2 = ba.obs_errors(x)

    intr, w, C, X, _ = ba.unpack(x)
    R = (_expmap(w) @ ba.R0).cpu().numpy()  # world->cam
    c2w_ref = np.tile(np.eye(4), (T, 1, 1))
    c2w_ref[:, :3, :3] = np.transpose(R, (0, 2, 1))
    c2w_ref[:, :3, 3] = C.cpu().numpy()
    f = float(f0 * np.exp(float(intr[0])))
    k1, k2 = float(intr[1]), float(intr[2])

    # Per-frame median error and the points that survived (â‰¥2 inlier observations).
    kf = kept_obs[:, 0].astype(int)
    per_frame = np.array([np.median(e2[kf == i]) if (kf == i).any() else np.nan for i in range(T)])
    kt = kept_obs[:, 1].astype(int)
    support = np.bincount(kt, minlength=len(colors))
    pts_ok = support >= 2
    # Inlier observations of the kept points (frame position, index into track_points, u, v) so later
    # tools know which points each frame actually saw.
    new_id = -np.ones(len(colors), np.int64)
    new_id[pts_ok] = np.arange(int(pts_ok.sum()))
    tob = kept_obs[pts_ok[kt]].copy()
    tob[:, 1] = new_id[tob[:, 1].astype(np.int64)]
    out = dict(
        track_obs=tob.astype(np.float32),
        refined=np.array(True), c2w_refined=c2w_ref, f_px=np.array(f), k1=np.array(k1), k2=np.array(k2),
        plate_size=np.array([W, H]), principal_px=np.array([cx, cy]),
        track_points=X.cpu().numpy()[pts_ok].astype(np.float32), track_colors=colors[pts_ok],
        reproj_per_frame_px=per_frame.astype(np.float32),
    )
    stats = dict(init=_stats(e_init[ok0]), round1=_stats(e1), final=_stats(e2), outlier_thresh_px=thr,
                 prefilter_thresh_px=thr0, prefilter_kept=float(ok0.mean()),
                 tracks=int(pts_ok.sum()), observations=int(len(kept_obs)), plate=[W, H],
                 f_px=f, f_px_pi4=f0, k1=k1, k2=k2)
    log(f"[refine] final reprojection: median {stats['final']['median']:.3f}px, p90 {stats['final']['p90']:.3f}px, "
        f"f {f0:.1f}->{f:.1f}px, k1 {k1:.4f}, k2 {k2:.4f}")
    return out, stats

