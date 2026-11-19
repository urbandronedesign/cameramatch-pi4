"""Scan translation scale s: project frame-0 static points into frame t with pose (R, s*t),
compare with Pi4's own 2D tracks. Also report the s that upstream's formula implies (s=1)."""
import sys
import numpy as np
import torch
from pi4 import Pi4
from pi4.io import load_images

src, nmax = sys.argv[1], int(sys.argv[2])
frames, meta = load_images(src, stride=1, max_frames=nmax, max_size=512)
T, H, W, _ = frames.shape
img = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float().div_(255).unsqueeze(0).cuda()
model = Pi4.from_pretrained("yyfz233/Pi4").cuda()
p = model(img, query_frame=0)
g = lambda k: p[k][0].float().cpu().numpy()
P, C, conf, dyn, uv, vis = g("local_points"), g("camera_poses"), g("confidence"), g("dynamic_mask"), g("tracks_2d"), g("visibility")
m = float(p["metric_scale"][0])
print("T", T, "metric", m)

sel = (dyn[0] < 0.3) & (conf[0] > np.quantile(conf[0], 0.5))
ys, xs = np.nonzero(sel)
i = np.random.default_rng(0).choice(len(ys), min(5000, len(ys)), replace=False)
ys, xs = ys[i], xs[i]
P0 = P[0, ys, xs].astype(np.float64)


def focal(t):
    Z = P[t, ..., 2]; u = np.arange(W) + 0.5 - W / 2; v = np.arange(H)[:, None] + 0.5 - H / 2
    x = P[t, ..., 0] / Z; y = P[t, ..., 1] / Z
    return (np.sum(u * x) + np.sum(v * y)) / (np.sum(x * x) + np.sum(y * y))


f0 = focal(0)
scales = [0, 0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.15, 0.25, 0.5, 1.0]
for t in [1, 5, 15, 30, 60, T - 1]:
    if t >= T:
        continue
    ok = vis[t, ys, xs] > 0.5
    tu, tv = uv[t, ys, xs, 0], uv[t, ys, xs, 1]
    R0, t0, Rt, tt = C[0, :3, :3], C[0, :3, 3], C[t, :3, :3], C[t, :3, 3]
    errs = []
    for s in scales:
        Xw = P0 @ R0.T + s * t0
        Xc = (Xw - s * tt) @ Rt  # R_t^T (X - s t_t)
        good = ok & (Xc[:, 2] > 1e-6)
        ft = focal(t)
        u = Xc[:, 0] / Xc[:, 2] * ft + W / 2 - 0.5
        v = Xc[:, 1] / Xc[:, 2] * ft + H / 2 - 0.5
        errs.append(np.median(np.hypot(u - tu, v - tv)[good]) if good.any() else np.nan)
    best = scales[int(np.nanargmin(errs))]
    print(f"frame {t:3d}: visible {ok.mean():.2f}  err(px) " + " ".join(f"{s}:{e:.1f}" for s, e in zip(scales, errs)) + f"  -> best s={best}")
print("1/metric =", 1 / m)
