"""Compose rolling-shutter frames from sub-frame renders (tests/gt_render_rs.py).

python gt_rolling_compose.py <gt_dir>
Plate row r (of the final H-row plate, centred in the overscan render) is exposed at
(r / (H - 1) - 0.5) * readout frames from the frame centre; its pixels are interpolated between the two
sub-frames around that time. Writes undist/<frame>.png (overscan size) for tests/gt_distort.py.
"""
import json
import os
import sys

import cv2
import numpy as np

d = sys.argv[1]
gt = json.load(open(os.path.join(d, "gt.json")))
W, H = gt["plate"]
Wr, Hr = gt["render"]
K, ro = int(gt["K"]), float(gt["readout"])
os.makedirs(os.path.join(d, "undist"), exist_ok=True)
off = (Hr - H) / 2
rows = np.arange(Hr)
tau = np.clip(((rows - off) / (H - 1) - 0.5), -0.5, 0.5)       # in units of the readout (render rows clamped)
kf = (tau + 0.5) * (K - 1)
k0 = np.clip(np.floor(kf).astype(int), 0, K - 2)
a = (kf - k0)[:, None, None].astype(np.float32)
n = len(gt["c2w_blender"])
for i in range(n):
    subs = np.stack([cv2.imread(os.path.join(d, "sub", f"{i:06d}_{k:02d}.png")).astype(np.float32) for k in range(K)])
    img = subs[k0, rows] * (1 - a) + subs[k0 + 1, rows] * a
    cv2.imwrite(os.path.join(d, "undist", f"{i:06d}.png"), np.clip(img + 0.5, 0, 255).astype(np.uint8))
print("composed", n, "rolling-shutter frames, readout", ro, "frame, K", K)
