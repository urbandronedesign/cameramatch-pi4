"""Verify BA analytic Jacobian against finite differences on a small synthetic problem.
Run inside the container: python /tests/check_ba_jacobian.py  (or with server/ on sys.path)."""
import sys

import numpy as np
import torch

sys.path.insert(0, "/app")
import refine as rf  # noqa: E402

rng = np.random.default_rng(0)
T, N = 6, 40
W, H = 640, 360
c2w = np.tile(np.eye(4), (T, 1, 1))
for i in range(T):
    ang = 0.05 * i
    c2w[i, :3, :3] = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    c2w[i, :3, 3] = [0.3 * i, 0.05 * i, 0.1 * i]
X = rng.uniform([-3, -2, 6], [3, 2, 12], (N, 3))
f0 = 500.0
obs = []
for i in range(T):
    Rwc = c2w[i, :3, :3].T
    Xc = (X - c2w[i, :3, 3]) @ Rwc.T
    u = f0 * Xc[:, 0] / Xc[:, 2] + (W - 1) / 2
    v = f0 * Xc[:, 1] / Xc[:, 2] + (H - 1) / 2
    for j in range(N):
        obs.append((i, j, u[j] + rng.normal(0, 0.5), v[j] + rng.normal(0, 0.5)))
obs = np.array(obs)
ba = rf.BA(c2w, f0, (W, H), obs, X + rng.normal(0, 0.1, X.shape), prior_sigma=1.0, device="cpu")
x = ba.x0 + rng.normal(0, 1e-3, ba.x0.shape)
x[:3] = [0.02, 0.01, -0.005]
J = ba.jac(x).toarray()
eps = 1e-6
Jn = np.zeros_like(J)
f = ba.fun(x)
for k in range(len(x)):
    xp = x.copy(); xp[k] += eps
    Jn[:, k] = (ba.fun(xp) - f) / eps
err = np.abs(J - Jn)
print("max abs err", err.max(), "rel", err.max() / np.abs(Jn).max())
blocks = {"intr": slice(0, 3), "w": slice(ba.o_w, ba.o_c), "C": slice(ba.o_c, ba.o_x), "X": slice(ba.o_x, None)}
for k, s in blocks.items():
    print(f"{k:5s} max err {err[:, s].max():.3e}  max |J| {np.abs(Jn[:, s]).max():.3e}")
x_opt = ba.solve(x, 100, print)
e = ba.obs_errors(x_opt)
print("synthetic BA final median px", np.median(e), "f", f0 * np.exp(x_opt[0]), "k", x_opt[1:3])
