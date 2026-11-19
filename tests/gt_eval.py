"""Compare a solve against the synthetic ground truth.  python gt_eval.py <gt_dir> <job_dir>"""
import json
import os
import sys

import numpy as np

gt_dir, job = sys.argv[1], sys.argv[2]
gt = json.load(open(os.path.join(gt_dir, "gt.json")))
d = np.load(os.path.join(job, "result.npz"))
idx = d["frame_indices"].astype(int)
G = np.array(gt["c2w_blender"])[idx] @ np.diag([1.0, -1, -1, 1])  # Blender cam -> OpenCV cam axes


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    E[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ E @ Vt
    s = np.trace(np.diag(sig) @ E) / S.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def evaluate(name, C):
    s, R, t = umeyama(C[:, :3, 3], G[:, :3, 3])
    pos = (s * (R @ C[:, :3, 3].T)).T + t
    ate = np.linalg.norm(pos - G[:, :3, 3], axis=1)
    path = np.linalg.norm(np.diff(G[:, :3, 3], axis=0), axis=1).sum()
    rerr = []
    for i in range(len(C)):
        dR = G[i, :3, :3].T @ (R @ C[i, :3, :3])
        rerr.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    rerr = np.array(rerr)
    print(f"{name:8s} ATE rms {np.sqrt((ate**2).mean()):.4f} m ({100*np.sqrt((ate**2).mean())/path:.2f}% of {path:.1f} m path)"
          f" | rot err median {np.median(rerr):.3f} deg max {rerr.max():.3f} | sim3 scale {s:.4f}")


W = gt["plate"][0]
evaluate("Pi4", d["c2w"])
print(f"         focal {np.median(d['fx']) * W / d['model_size'][0]:.1f}px  (GT {gt['f_px']:.1f})")
if "refined" in d:
    evaluate("refined", d["c2w_refined"])
    print(f"         focal {float(d['f_px']):.1f}px (GT {gt['f_px']:.1f})  k1 {float(d['k1']):.4f} (GT {gt.get('k1')})"
          f"  k2 {float(d['k2']):.4f} (GT {gt.get('k2')})")
    print(f"         median reprojection per frame: {np.nanmedian(d['reproj_per_frame_px']):.3f}px")
