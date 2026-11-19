"""Score projector renders (tests/check_floor_reprojection.py) against the UNDISTORTED plate.

python /tools/floor_check_eval.py /work/<job> <render_dir> <lag> [frames...]
Blender renders through an undistorted pinhole camera, so the plate is undistorted with the solved
Polynomial k1/k2 first. Reports |render - plate_f| next to the no-geometry baseline |plate_f - plate_(f-lag)|.
"""
import os
import sys

import cv2
import numpy as np

job, rdir, lag = sys.argv[1], sys.argv[2], int(sys.argv[3])
fl = [int(x) for x in sys.argv[4:]] or [40, 100, 150, 200, 260]
r = np.load(os.path.join(job, "result.npz"))
W, H = (int(v) for v in r["plate_size"])
f, k1, k2 = float(r["f_px"]), float(r["k1"]), float(r["k2"])
y, x = np.mgrid[0:H, 0:W].astype(np.float64)
xn, yn = (x + 0.5 - W / 2) / f, (y + 0.5 - H / 2) / f          # Blender render pixel -> pinhole ray
q = xn * xn + yn * yn
d = 1 + k1 * q + k2 * q * q
mx = (f * xn * d + (W - 1) / 2).astype(np.float32)            # where that ray lands on the plate
my = (f * yn * d + (H - 1) / 2).astype(np.float32)
cap = cv2.VideoCapture(os.path.join(job, "input", os.listdir(os.path.join(job, "input"))[0]))
und, i = {}, 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    i += 1
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    und[i] = cv2.remap(g, mx, my, cv2.INTER_LINEAR, borderValue=-1)
    al = cv2.imread(os.path.join(job, "matte", "fused", f"alpha_{i:04d}.png"), 0)
    und[(i, "m")] = cv2.remap(al, mx, my, cv2.INTER_LINEAR) > 30 if al is not None else np.zeros((H, W), bool)
tot = []
for fr in fl:
    rd4 = cv2.imread(os.path.join(rdir, f"lag_{fr:04d}.png"), cv2.IMREAD_UNCHANGED)
    rd = rd4[..., :3]
    cover = rd4[..., 3] > 250 if rd4.shape[2] == 4 else rd.max(2) > 0
    rg = cv2.cvtColor(rd, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fg = und[(fr, "m")] | cv2.dilate(und[(fr - lag, "m")].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    m = cover & ~fg & (und[fr] >= 0) & (und[fr - lag] >= 0)
    m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    base = np.abs(und[fr] - und[fr - lag])[m].mean()
    rep = np.abs(und[fr] - rg)[m].mean()
    tot.append((base, rep))
    print(f"frame {fr}: floor px {m.sum():6d} | no-geometry {base:5.1f} | reprojected {rep:5.1f} | "
          f"reduced {100 * (1 - rep / base):4.0f}%")
b, p = np.mean(tot, 0)
print(f"MEAN no-geometry {b:.2f} reprojected {p:.2f} reduced {100 * (1 - p / b):.0f}%")
