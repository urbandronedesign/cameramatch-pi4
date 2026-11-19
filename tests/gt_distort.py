"""Apply a known Polynomial (libmv/Blender) distortion to the overscan renders.

python gt_distort.py <gt_dir> <k1> <k2>
For each output pixel (distorted, plate res) solve x_u from x_d = x_u (1 + k1 r_u^2 + k2 r_u^4),
then sample the undistorted overscan render at f*x_u + c.
"""
import json
import os
import sys

import cv2
import numpy as np

d, k1, k2 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
gt = json.load(open(os.path.join(d, "gt.json")))
W, H = gt["plate"]
Wr, Hr = gt["render"]
f_frames = gt.get("f_px_frames") or [gt["f_px"]]
u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))


def maps(f):
    xd, yd = (u - (W - 1) / 2) / f, (v - (H - 1) / 2) / f
    xu, yu = xd.copy(), yd.copy()
    for _ in range(50):
        r2 = xu * xu + yu * yu
        s = 1 + k1 * r2 + k2 * r2 * r2
        xu, yu = xd / s, yd / s
    return (xu * f + (Wr - 1) / 2).astype(np.float32), (yu * f + (Hr - 1) / 2).astype(np.float32)


mapx, mapy = maps(f_frames[0])
oob = (mapx < 0) | (mapx > Wr - 1) | (mapy < 0) | (mapy > Hr - 1)
print("out-of-render fraction (first frame)", oob.mean())
os.makedirs(os.path.join(d, "plate"), exist_ok=True)
files = sorted(os.listdir(os.path.join(d, "undist")))
for i, fn in enumerate(files):
    if len(f_frames) > 1:                                  # zoom: per-frame focal
        mapx, mapy = maps(f_frames[i])
    img = cv2.imread(os.path.join(d, "undist", fn))
    cv2.imwrite(os.path.join(d, "plate", fn), cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC))
gt.update(k1=k1, k2=k2)
json.dump(gt, open(os.path.join(d, "gt.json"), "w"))
print("distorted", len(files), "frames")
