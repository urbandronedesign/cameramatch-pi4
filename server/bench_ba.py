"""Benchmark the bundle-adjustment solvers on an existing job (does not modify the job).

python /app/bench_ba.py /work/<job> [--gt /work/gt_scene] [--solvers lm,scipy] [--tracks 600] [--zoom [--zoom-smooth 0.01]]
           [--no-distortion] [--focal-sigma 0.3]
Prints, per solver: BA time, reprojection median/p90, focal, k1/k2 and, with --gt, trajectory / rotation /
focal error against the synthetic ground truth (gt.json from tests/gt_render.py).
"""
import argparse
import json
import os
import time

import numpy as np

import refine as rf


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    E[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ E @ Vt
    s = np.trace(np.diag(sig) @ E) / S.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def gt_metrics(gt_dir, result, out):
    gt = json.load(open(os.path.join(gt_dir, "gt.json")))
    idx = result["frame_indices"].astype(int)
    G = np.array(gt["c2w_blender"])[idx] @ np.diag([1.0, -1, -1, 1])
    P = out["c2w_refined"]
    s, R, t = umeyama(P[:, :3, 3], G[:, :3, 3])
    pos = (s * (R @ P[:, :3, 3].T).T + t)
    path = np.sum(np.linalg.norm(np.diff(G[:, :3, 3], axis=0), axis=1))
    terr = np.linalg.norm(pos - G[:, :3, 3], axis=1).mean() / path * 100
    rerr = []
    for a, b in zip(P, G):
        Rr = (R @ a[:3, :3]).T @ b[:3, :3]
        rerr.append(np.degrees(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1))))
    res = dict(traj_err_pct_path=round(float(terr), 4), rot_err_med_deg=round(float(np.median(rerr)), 4))
    if "f_px_frames" in gt:
        fg = np.array(gt["f_px_frames"])[idx]
        res["focal_err_pct_med"] = round(float(np.median(np.abs(out["f_px_frames"] / fg - 1)) * 100), 3)
    else:
        res["focal_px_gt"] = gt.get("f_px", None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--gt", default="")
    ap.add_argument("--solvers", default="lm,scipy")
    ap.add_argument("--tracks", type=int, default=600)
    ap.add_argument("--zoom", action="store_true")
    ap.add_argument("--zoom-smooth", type=float, default=0.01)
    ap.add_argument("--no-distortion", action="store_true")
    ap.add_argument("--focal-sigma", type=float, default=0.3)
    a = ap.parse_args()
    result = dict(np.load(os.path.join(a.job, "result.npz")))
    with np.load(os.path.join(a.job, "pi4_cache.npz")) as z:
        cache = {k: z[k] for k in z.files}
    src = os.path.normpath(os.path.join(a.job, str(cache["src"])))
    rows = {}
    for solver in a.solvers.split(","):
        lines = []
        t0 = time.time()
        out, st = rf.refine(src, result, cache, log=lambda s: (lines.append(s), print("   ", s, flush=True)),
                            max_active=a.tracks, focal_sigma=a.focal_sigma,
                            solve_distortion=not a.no_distortion, solver=solver, zoom=a.zoom,
                            zoom_smooth=a.zoom_smooth)
        tot = time.time() - t0
        ba_s = sum(float(l.split("evals, " if "evals" in l else "iterations, ")[1].rstrip("s"))
                   for l in lines if l.startswith("[refine] BA"))
        row = dict(total_s=round(tot, 1), ba_s=round(ba_s, 1), median_px=round(st["final"]["median"], 4),
                   p90_px=round(st["final"]["p90"], 4), f_px=round(st["f_px"], 2),
                   f_range=[round(v, 1) for v in st["f_px_range"]], k1=round(st["k1"], 5), k2=round(st["k2"], 5),
                   obs=st["observations"], tracks=st["tracks"])
        if a.gt:
            row.update(gt_metrics(a.gt, result, out))
        rows[solver] = row
        print(f"== {solver}: {json.dumps(row)}", flush=True)
    print("SUMMARY", json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
