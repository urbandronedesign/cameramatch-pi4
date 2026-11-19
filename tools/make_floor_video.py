"""Three-panel video: plate | shot composite (gold panels, occlusion) | witness view (reprojected floor).

python /tools/make_floor_video.py <job_dir> <shot_dir> <witness_dir> <out.mp4> [fps]
"""
import os
import subprocess
import sys

import cv2
import numpy as np

job, shot, wit, out = sys.argv[1:5]
fps = float(sys.argv[5]) if len(sys.argv) > 5 else 30.0
tmp = os.path.join(os.path.dirname(out), "_fv_frames")
os.makedirs(tmp, exist_ok=True)
cap = cv2.VideoCapture(os.path.join(job, "input", os.listdir(os.path.join(job, "input"))[0]))
labels = ["plate", "solve + gold panels + occlusion", "witness: floor rebuilt from the shot, video reprojected"]
n = 0
while True:
    ok, plate = cap.read()
    if not ok:
        break
    f = n + 1
    c = cv2.imread(os.path.join(shot, f"occl_{f:04d}.png"))
    w = cv2.imread(os.path.join(wit, f"wit_{f:04d}.png"))
    if c is None or w is None:
        break
    H = plate.shape[0]
    w = cv2.resize(w, (int(w.shape[1] * H / w.shape[0]), H))
    panels = [plate, cv2.resize(c, (plate.shape[1], H)), w]
    for p, t in zip(panels, labels):
        cv2.rectangle(p, (0, H - 22), (p.shape[1], H), (0, 0, 0), -1)
        cv2.putText(p, t, (6, H - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    frame = np.hstack(panels)
    cv2.putText(frame, f"frame {f:03d}", (frame.shape[1] - 95, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(tmp, f"{n:05d}.png"), frame)
    n += 1
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out], check=True)
for fn in os.listdir(tmp):
    os.remove(os.path.join(tmp, fn))
os.rmdir(tmp)
print("wrote", out, n, "frames")
