"""Per-frame scale between Pi4 depth and the bundle-adjusted geometry (track points), robust median.

python /tools/depth_scale_check.py /work/<job>   -> prints stats, writes depth_scale.npy (per frame)
"""
import os
import sys

import cv2
import numpy as np

job = sys.argv[1]
r = np.load(os.path.join(job, "result.npz"))
with np.load(os.path.join(job, "pi4_cache.npz")) as z:
    depth = z["depth"].astype(np.float32)
c2w = r["c2w_refined"]
X = r["track_points"].astype(np.float64)
W, H = (int(v) for v in r["plate_size"])
f, k1, k2 = float(r["f_px"]), float(r["k1"]), float(r["k2"])
cx, cy = (float(v) for v in r["principal_px"])
T = len(c2w)
obs = r["track_obs"]  # (frame_pos, point index, u, v) - only points actually tracked in that frame
scales, counts = np.full(T, np.nan), np.zeros(T, int)
for i in range(T):
    o = obs[obs[:, 0].astype(int) == i]
    if len(o) < 15:
        continue
    w2c = np.linalg.inv(c2w[i])
    Xc = X[o[:, 1].astype(int)] @ w2c[:3, :3].T + w2c[:3, 3]
    D = cv2.resize(depth[i], (W, H), interpolation=cv2.INTER_LINEAR)
    zp = D[np.clip(o[:, 3].round().astype(int), 0, H - 1), np.clip(o[:, 2].round().astype(int), 0, W - 1)]
    ok = (Xc[:, 2] > 1e-3) & (zp > 1e-3)
    lr = np.log(Xc[ok, 2] / zp[ok])
    if len(lr) < 15:
        continue
    scales[i] = float(np.exp(np.median(lr)))
    counts[i] = int(len(lr))
    if i % 25 == 0:
        print(f"    frame {i + 1}: log-ratio spread (IQR) {np.subtract(*np.percentile(lr, [75, 25])):.3f}")
good = np.isfinite(scales)
print(f"frames with estimate: {good.sum()}/{T}")
q = np.percentile(scales[good], [5, 25, 50, 75, 95])
print("scale (BA depth / Pi4 depth) pct 5/25/50/75/95:", np.round(q, 3))
for i in range(0, T, 25):
    print(f"  frame {i + 1:3d}: scale {scales[i]:.3f}  support {counts[i]}")
# smooth + fill for use
idx = np.arange(T)
s = np.interp(idx, idx[good], scales[good])
k = 9
s_s = np.convolve(np.pad(np.log(s), k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), "valid")
np.save(os.path.join(job, "depth_scale.npy"), np.exp(s_s))
print("saved depth_scale.npy")
