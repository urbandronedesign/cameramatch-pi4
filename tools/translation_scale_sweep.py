"""Is the depth/camera inconsistency a translation-scale mismatch?

python /tools/translation_scale_sweep.py /work/<job> [lag]
Warp frame f-lag into f through Pi4 depth_f with the refined cameras, but with relative translations
scaled by s. Error on textured static pixels vs s (and vs no warp).
"""
import os
import sys

import cv2
import numpy as np

job = sys.argv[1]
lag = int(sys.argv[2]) if len(sys.argv) > 2 else 2
r = np.load(os.path.join(job, "result.npz"))
with np.load(os.path.join(job, "pi4_cache.npz")) as z:
    depth, src = z["depth"].astype(np.float32), str(z["src"])
W, H = (int(v) for v in r["plate_size"])
f, k1, k2 = float(r["f_px"]), float(r["k1"]), float(r["k2"])
cx, cy = (W - 1) / 2, (H - 1) / 2
sp = os.path.normpath(os.path.join(job, src))
frames = []
if os.path.isdir(sp):
    frames = [cv2.imread(os.path.join(sp, fn), 0).astype(np.float32) for fn in sorted(os.listdir(sp))]
else:
    cap = cv2.VideoCapture(sp)
    while True:
        ok, x = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32))
C = r["c2w_refined"]
vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
xd, yd = (uu - cx) / f, (vv - cy) / f
xn, yn = xd.copy(), yd.copy()
for _ in range(30):
    q = xn * xn + yn * yn
    d = 1 + k1 * q + k2 * q * q
    xn, yn = xd / d, yd / d


def alpha(i):
    a = cv2.imread(os.path.join(job, "matte", "fused", f"alpha_{i + 1:04d}.png"), 0)
    return (a > 30) if a is not None else np.zeros((H, W), bool)


scales = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
err = {s: [] for s in scales}
base = []
for fidx in range(lag + 5, len(C), 12):
    g = fidx - lag
    Z = cv2.resize(depth[fidx], (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    static = ~alpha(fidx) & ~cv2.dilate(alpha(g).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    gm = np.hypot(cv2.Sobel(frames[fidx], cv2.CV_32F, 1, 0), cv2.Sobel(frames[fidx], cv2.CV_32F, 0, 1))
    static &= gm > np.percentile(gm, 70)
    base.append(np.abs(frames[fidx] - frames[g])[static].mean())
    Xc = np.stack([xn * Z, yn * Z, Z], -1)
    rel = np.linalg.inv(C[g]) @ C[fidx]
    for s in scales:
        Xg = Xc @ rel[:3, :3].T + s * rel[:3, 3]
        x, y = Xg[..., 0] / Xg[..., 2], Xg[..., 1] / Xg[..., 2]
        q = x * x + y * y
        d = 1 + k1 * q + k2 * q * q
        u, v = f * x * d + cx, f * y * d + cy
        w = cv2.remap(frames[g], u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderValue=-1)
        m = static & (w >= 0) & (Xg[..., 2] > 0)
        err[s].append(np.abs(frames[fidx] - w)[m].mean())
print(f"lag {lag}: no-warp {np.mean(base):.2f}")
for s in scales:
    print(f"  translation x{s:<5}: {np.mean(err[s]):.2f}")
