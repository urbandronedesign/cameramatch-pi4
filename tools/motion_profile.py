"""Camera-motion profile of clips / shots: per-frame similarity fit (scale, rotation, translation) on KLT flow.

python /tools/motion_profile.py <video> [--ranges a-b,c-d] [--step 2]
Reports per range: cumulative image scale (zoom or dolly), translation px/frame, rotation deg/frame, jitter
(high-frequency translation: handheld), and parallax (residual of the similarity fit: real 3D motion).
A zoom = large cumulative scale with LOW parallax residual.
"""
import argparse
import json

import cv2
import numpy as np


def profile(cap, a, b, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    ok, fr = cap.read()
    if not ok:
        return None
    g0 = cv2.cvtColor(cv2.resize(fr, (640, int(640 * fr.shape[0] / fr.shape[1]))), cv2.COLOR_BGR2GRAY)
    scales, tx, ty, rot, resid = [], [], [], [], []
    i = a
    while i + step <= b:
        for _ in range(step):
            ok, fr = cap.read()
        if not ok:
            break
        i += step
        g1 = cv2.cvtColor(cv2.resize(fr, (g0.shape[1], g0.shape[0])), cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(g0, 600, 0.01, 8)
        if p0 is None or len(p0) < 30:
            g0 = g1
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None)
        m = st[:, 0] == 1
        if m.sum() < 30:
            g0 = g1
            continue
        A, inl = cv2.estimateAffinePartial2D(p0[m], p1[m], method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if A is None:
            g0 = g1
            continue
        s = float(np.hypot(A[0, 0], A[1, 0]))
        scales.append(np.log(s))
        tx.append(A[0, 2])
        ty.append(A[1, 2])
        rot.append(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
        pred = (p0[m][:, 0] @ A[:, :2].T) + A[:, 2]
        resid.append(float(np.median(np.linalg.norm(pred - p1[m][:, 0], axis=1))))
        g0 = g1
    if len(scales) < 5:
        return None
    t = np.hypot(tx, ty)
    jit = np.hypot(np.diff(tx), np.diff(ty))
    return dict(frames=[a, b], zoom_total=round(float(np.exp(np.sum(scales))), 3),
                scale_per_step_med=round(float(np.median(np.abs(scales))) * 100, 3),
                trans_px_step=round(float(np.median(t)), 2), rot_deg_step=round(float(np.median(np.abs(rot))), 3),
                jitter_px=round(float(np.median(jit)), 2), parallax_px=round(float(np.median(resid)), 3),
                scale_monotonic=round(float(abs(np.sum(np.sign(scales))) / len(scales)), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--ranges", default="")
    ap.add_argument("--step", type=int, default=2)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rngs = [tuple(int(v) for v in r.split("-")) for r in a.ranges.split(",")] if a.ranges else [(0, n - 1)]
    for r in rngs:
        p = profile(cap, r[0], r[1], a.step)
        if p:
            print(json.dumps(p), flush=True)


if __name__ == "__main__":
    main()
