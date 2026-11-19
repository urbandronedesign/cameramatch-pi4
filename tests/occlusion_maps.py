"""Export Pi4 occlusion maps for compositing (run in the container).

python occlusion_maps.py <job_dir> <plate_w> <plate_h> <first_scene_frame>
Writes <job_dir>/occlusion/invdepth_####.png (16-bit, value = (1/depth) / INV_SCALE, plate res) and
fgmask_####.png (moving foreground = rider/bike, dilated), numbered by scene frame.
"""
import os
import sys

import cv2  # noqa: E402
import numpy as np  # noqa: E402

job, W, H, f0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
c = np.load(os.path.join(job, "pi4_cache.npz"))
r = np.load(os.path.join(job, "result.npz"))
depth, dyn = c["depth"].astype(np.float32), c["dyn"]
idx = r["frame_indices"].astype(int)
out = os.path.join(job, "occlusion")
os.makedirs(out, exist_ok=True)
k = np.ones((5, 5), np.uint8)
INV_SCALE = 4.0  # 1/depth stored /4 -> depths down to 0.25 units; ~6e-5 precision
for i, fi in enumerate(idx):
    d = cv2.resize(depth[i], (W, H), interpolation=cv2.INTER_LINEAR)
    inv = np.where(d > 1e-6, 1.0 / np.maximum(d, 1e-6), 0.0).astype(np.float32)
    q = np.clip(inv / INV_SCALE * 65535.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(out, f"invdepth_{f0 + fi:04d}.png"), q)
    m = cv2.resize(dyn[i], (W, H), interpolation=cv2.INTER_LINEAR)
    m = (m > 127).astype(np.uint8) * 255
    m = cv2.GaussianBlur(cv2.dilate(m, k, iterations=1), (5, 5), 0)
    cv2.imwrite(os.path.join(out, f"fgmask_{f0 + fi:04d}.png"), m)
print("wrote", len(idx), "frames to", out, "depth range", float(depth.min()), float(np.median(depth)), float(depth.max()))
