"""Which camera set is consistent with Pi4 depth?  Warp frame f-LAG into frame f through depth_f.

python /tools/depth_pose_consistency.py /work/<job> [lag]
For camera set A (refined BA cameras + refined k1/k2 lens) and B (Pi4's own cameras + Pi4 pinhole fx):
lift depth_f to 3D in camera f, project into camera f-lag, sample plate f-lag, compare with plate f
(static pixels only). Also reports the plain no-warp baseline.
"""
import os
import sys

import cv2
import numpy as np

job = sys.argv[1]
lag = int(sys.argv[2]) if len(sys.argv) > 2 else 8
r = np.load(os.path.join(job, "result.npz"))
with np.load(os.path.join(job, "pi4_cache.npz")) as z:
    depth, src = z["depth"].astype(np.float32), str(z["src"])
W, H = (int(v) for v in r["plate_size"])
Wm = int(r["model_size"][0])
sp = os.path.normpath(os.path.join(job, src))
if os.path.isdir(sp):
    frames = [cv2.imread(os.path.join(sp, fn), 0).astype(np.float32) for fn in sorted(os.listdir(sp))]
else:
    cap = cv2.VideoCapture(sp)
    frames = []
    while True:
        ok, x = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32))
T = len(r["c2w"])
def alpha(i):
    a = cv2.imread(os.path.join(job, "matte", "fused", f"alpha_{i + 1:04d}.png"), 0)
    return (a > 30) if a is not None else np.zeros((H, W), bool)
vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)


def lens_refined():
    f, k1, k2 = float(r["f_px"]), float(r["k1"]), float(r["k2"])
    cx, cy = (W - 1) / 2, (H - 1) / 2
    xd, yd = (uu - cx) / f, (vv - cy) / f
    xn, yn = xd.copy(), yd.copy()
    for _ in range(30):
        q = xn * xn + yn * yn
        d = 1 + k1 * q + k2 * q * q
        xn, yn = xd / d, yd / d

    def proj(X):
        x, y = X[..., 0] / X[..., 2], X[..., 1] / X[..., 2]
        q = x * x + y * y
        d = 1 + k1 * q + k2 * q * q
        return f * x * d + cx, f * y * d + cy
    return xn, yn, proj


def lens_pi4(i):
    f = float(r["fx"][i]) * W / Wm
    cx, cy = W / 2 - 0.5, H / 2 - 0.5
    return (uu - cx) / f, (vv - cy) / f, (lambda X: (f * X[..., 0] / X[..., 2] + cx, f * X[..., 1] / X[..., 2] + cy))


res = {"refined": [], "pi4": [], "nowarp": []}
for fidx in range(lag + 5, T, 15):
    g = fidx - lag
    Z = cv2.resize(depth[fidx], (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    static = ~alpha(fidx) & ~cv2.dilate(alpha(g).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    gm = np.hypot(cv2.Sobel(frames[fidx], cv2.CV_32F, 1, 0), cv2.Sobel(frames[fidx], cv2.CV_32F, 0, 1))
    static &= gm > np.percentile(gm, 70)  # textured pixels only (misalignment is visible there)
    res["nowarp"].append(np.abs(frames[fidx] - frames[g])[static].mean())
    for name in ("refined", "pi4"):
        if name == "refined":
            C = r["c2w_refined"]
            xn, yn, proj = lens_refined()
        else:
            C = r["c2w"]
            xn, yn, proj = lens_pi4(fidx)
            _, _, proj = lens_pi4(g)
        Xc = np.stack([xn * Z, yn * Z, Z], -1)
        rel = np.linalg.inv(C[g]) @ C[fidx]
        Xg = Xc @ rel[:3, :3].T + rel[:3, 3]
        u, v = proj(Xg)
        warped = cv2.remap(frames[g], u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                           borderValue=-1)
        m = static & (warped >= 0) & (Xg[..., 2] > 0)
        res[name].append(np.abs(frames[fidx] - warped)[m].mean())
    print(f"frame {fidx + 1:3d}: no-warp {res['nowarp'][-1]:5.1f} | refined cams {res['refined'][-1]:5.1f} | "
          f"Pi4 cams {res['pi4'][-1]:5.1f}")
print({k: round(float(np.mean(v)), 2) for k, v in res.items()})
