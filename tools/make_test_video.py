"""Four-panel test video: plate | matte | Pi4 depth | final composite (H.264).

python /tools/make_test_video.py <job_dir> <comp_dir> <out.mp4> [fps]
"""
import os
import subprocess
import sys

import cv2
import numpy as np

job, comp_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
fps = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
tmp = os.path.join(os.path.dirname(out), "_tv_frames")
os.makedirs(tmp, exist_ok=True)

src = os.path.join(job, "input", next(f for f in os.listdir(os.path.join(job, "input"))))
cap = cv2.VideoCapture(src)
labels = ["1  plate", "2  matte  (SAM 2.1 + MatAnyone 2)", "3  Pi4 depth", "4  solve + CG + occlusion"]
n = 0
while True:
    ok, plate = cap.read()
    if not ok:
        break
    f = n + 1
    comp = cv2.imread(os.path.join(comp_dir, f"occl_{f:04d}.png"))
    alpha = cv2.imread(os.path.join(job, "matte", "fused", f"alpha_{f:04d}.png"), 0)
    inv = cv2.imread(os.path.join(job, "occlusion", f"invdepth_{f:04d}.png"), cv2.IMREAD_UNCHANGED)
    if comp is None or alpha is None or inv is None:
        break
    H, W = plate.shape[:2]
    a = (alpha.astype(np.float32) / 255)[..., None]
    matte = (plate * (1 - 0.6 * a) + np.array([255, 60, 255]) * 0.6 * a).astype(np.uint8)
    d = np.clip(inv.astype(np.float32) / 65535 * 4.0, 0, None)  # 1/depth
    d = np.clip(np.sqrt(d / np.percentile(d, 99)) * 255, 0, 255).astype(np.uint8)
    depth = cv2.applyColorMap(d, cv2.COLORMAP_TURBO)
    panels = [plate, matte, depth, cv2.resize(comp, (W, H))]
    for p, t in zip(panels, labels):
        cv2.rectangle(p, (0, H - 22), (W, H), (0, 0, 0), -1)
        cv2.putText(p, t, (6, H - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    frame = np.hstack(panels)
    cv2.putText(frame, f"frame {f:03d}", (frame.shape[1] - 95, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(tmp, f"{n:05d}.png"), frame)
    n += 1
cap.release()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow", out], check=True)
for fn in os.listdir(tmp):
    os.remove(os.path.join(tmp, fn))
os.rmdir(tmp)
print("wrote", out, n, "frames")
