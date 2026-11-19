"""Visual solve check: a ground-plane grid drawn into every frame with the solved camera, focal and k1/k2.

python /tools/overlay_check.py /work/<job> [--out overlay] [--scale 0.5]
- Ground plane: RANSAC on the refined track points (falls back to the Pi4 point cloud), normal within 35 deg
  of the first camera's down axis, the plane with most support below the camera.
- Grid: ~14 lines each way over the plane's inlier extent, finely sampled so lens distortion bends it.
A good solve keeps the grid glued to the ground texture; drift / wrong focal makes it slide or swim.
Writes <out>/overlay.mp4 and <out>/sheet.jpg (6 frames), plus plane stats on stdout.
"""
import argparse
import json
import os
import subprocess

import cv2
import numpy as np


def ransac_plane(P, down, cam, iters=4000, tol=None, seed=0):
    """Ground plane: most inliers among planes that lie clearly BELOW the camera (not the canopy / a ceiling /
    an edge-on plane through the camera)."""
    rng = np.random.default_rng(seed)
    ext = np.linalg.norm(np.percentile(P, 90, 0) - np.percentile(P, 10, 0))
    tol = tol or 0.01 * ext
    min_below = 0.15 * float(np.median(np.linalg.norm(P - cam, axis=1)))
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n @ down) < np.cos(np.radians(35)):
            continue
        if n @ down < 0:
            n = -n
        d = -n @ a
        if -(n @ cam + d) < min_below:          # camera must sit above the plane (n points down)
            continue
        inl = np.abs(P @ n + d) < tol
        if inl.sum() > best_n:
            best_n, best = inl.sum(), (n, d, inl)
    if best is None:
        return None
    n, d, inl = best
    # refine by SVD on inliers
    Q = P[inl]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c)[2][-1]
    if n @ down < 0:
        n = -n
    return n, -n @ c, np.abs(P @ n - n @ c) < tol, tol


def project(Xw, c2w, f, k1, k2, cx, cy):
    w2c = np.linalg.inv(c2w)
    Xc = Xw @ w2c[:3, :3].T + w2c[:3, 3]
    z = Xc[:, 2]
    x, y = Xc[:, 0] / z, Xc[:, 1] / z
    r2 = x * x + y * y
    dd = 1 + k1 * r2 + k2 * r2 * r2
    return np.stack([f * x * dd + cx, f * y * dd + cy], 1), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--out", default="overlay")
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()
    d = np.load(os.path.join(a.job, "result.npz"))
    with np.load(os.path.join(a.job, "pi4_cache.npz")) as z:
        src = os.path.normpath(os.path.join(a.job, str(z["src"])))
        dyn = z["dyn"]                                   # Pi4 moving-object mask (model res, 0..255)
    refined = "refined" in d and bool(d["refined"])
    c2w = d["c2w_refined"] if refined else d["c2w"]
    W, H = (int(v) for v in d["plate_size"]) if refined else (None, None)
    P = d["track_points"].astype(np.float64) if refined and len(d["track_points"]) > 50 else d["points"]
    down = c2w[0, :3, 1]                                   # OpenCV camera +y = image down
    # only points below the first camera's optical axis
    rel = (P - c2w[0, :3, 3]) @ down
    Pb = P[rel > 0]
    fit = ransac_plane(Pb, down, c2w[0, :3, 3])
    out = os.path.join(a.job, a.out)
    os.makedirs(out, exist_ok=True)
    if fit is None:
        print(json.dumps({"plane": None}))
        return
    n, dpl, inl, tol = fit
    Q = Pb[inl]
    u = np.cross(n, c2w[0, :3, 2])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    c = Q.mean(0)
    cu, cv_ = (Q - c) @ u, (Q - c) @ v
    lo_u, hi_u = np.percentile(cu, [3, 97])
    lo_v, hi_v = np.percentile(cv_, [3, 97])
    step = max(hi_u - lo_u, hi_v - lo_v) / 14
    lines = []
    for gu in np.arange(lo_u, hi_u + 1e-9, step):
        t = np.linspace(lo_v, hi_v, 120)
        lines.append(c + np.outer(np.full_like(t, gu), u) + np.outer(t, v))
    for gv in np.arange(lo_v, hi_v + 1e-9, step):
        t = np.linspace(lo_u, hi_u, 120)
        lines.append(c + np.outer(t, u) + np.outer(np.full_like(t, gv), v))
    f_fr = d["f_px_frames"] if "f_px_frames" in d else np.full(len(c2w), float(d["f_px"]))
    k1, k2 = float(d["k1"]), float(d["k2"])
    idx = d["frame_indices"].astype(int)
    near = 0.05 * float(np.median(np.linalg.norm(Pb - c2w[0, :3, 3], axis=1)))
    # stick test points: grid intersections (3D, on the fitted ground)
    gi = np.array([c + gu * u + gv * v for gu in np.arange(lo_u, hi_u + 1e-9, step / 2)
                   for gv in np.arange(lo_v, hi_v + 1e-9, step / 2)])
    trk = {}            # id -> (prev gray pos)
    drift = []          # |KLT - projection| over frames for points tracked from their first visible frame
    drift_rate = []
    prev_gray = None
    if os.path.isdir(src):                                # image sequence copied by the add-on: input/seq/000000.png
        first = sorted(os.listdir(src))[0]
        src = os.path.join(src, "%06d" + os.path.splitext(first)[1])
    cap = cv2.VideoCapture(src)
    if W is None:
        W, H = int(cap.get(3)), int(cap.get(4))
    cx, cy = (W - 1) / 2, (H - 1) / 2
    sw, sh = int(W * a.scale) // 2 * 2, int(H * a.scale) // 2 * 2
    vw = cv2.VideoWriter(os.path.join(out, "overlay_raw.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 24, (sw, sh))
    sheet_at = set(np.linspace(0, len(idx) - 1, 6).astype(int))
    tiles = []
    fi = 0
    for k, fr_i in enumerate(idx):
        if k % 10 == 0:
            print(f"@progress {0.95 * k / len(idx):.3f} grid overlay frame {k + 1}/{len(idx)}", flush=True)
        while fi <= fr_i:
            ok, img = cap.read()
            fi += 1
        if not ok:
            break
        img = cv2.resize(img, (W, H)) if img.shape[1] != W else img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        uvg, zg = project(gi, c2w[k], float(f_fr[k]), k1, k2, cx, cy)
        inside = (zg > near) & (uvg[:, 0] > 20) & (uvg[:, 0] < W - 20) & (uvg[:, 1] > 20) & (uvg[:, 1] < H - 20)
        # never measure on moving things (people): the texture there is not the ground
        dm = cv2.dilate((dyn[k] > 100).astype(np.uint8), np.ones((5, 5), np.uint8))
        hm, wm = dm.shape
        mx = np.clip((uvg[:, 0] * wm / W).astype(int), 0, wm - 1)
        my = np.clip((uvg[:, 1] * hm / H).astype(int), 0, hm - 1)
        inside &= dm[my, mx] == 0
        if prev_gray is not None and trk:
            ids = list(trk)
            p0 = np.float32([trk[i][0] for i in ids])[:, None]
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, winSize=(31, 31), maxLevel=4)
            pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, winSize=(31, 31), maxLevel=4)
            fb = np.linalg.norm(pb[:, 0] - p0[:, 0], axis=1)
            for j, i in enumerate(ids):
                if st[j, 0] and st2[j, 0] and fb[j] < 0.5 and inside[i]:
                    age = trk[i][1] + 1
                    trk[i] = (p1[j, 0], age)
                    e = float(np.linalg.norm(p1[j, 0] - uvg[i]))
                    drift.append(e)
                    drift_rate.append(e / age)
                else:
                    del trk[i]
        # start new points where the ground is textured enough and not already tracked
        eig = cv2.cornerMinEigenVal(gray, 7)
        for i in np.nonzero(inside)[0]:
            if i not in trk:
                x, y = int(uvg[i, 0]), int(uvg[i, 1])
                if eig[y, x] > 1e-3:
                    trk[i] = (uvg[i].astype(np.float32), 0)
        prev_gray = gray
        for L in lines:
            uv, z = project(L, c2w[k], float(f_fr[k]), k1, k2, cx, cy)
            good = (z > near) & np.isfinite(uv).all(1)
            seg = []
            for ok_, p in zip(good, uv):
                if ok_ and -W < p[0] < 2 * W and -H < p[1] < 2 * H:
                    seg.append(p)
                elif len(seg) > 1:
                    cv2.polylines(img, [np.int32(np.round(seg))], False, (40, 230, 255), 2, cv2.LINE_AA)
                    seg = []
                else:
                    seg = []
            if len(seg) > 1:
                cv2.polylines(img, [np.int32(np.round(seg))], False, (40, 230, 255), 2, cv2.LINE_AA)
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        vw.write(small)
        if k in sheet_at:
            t = cv2.resize(img, (480, int(480 * H / W)))
            cv2.putText(t, f"frame {fr_i + 1}", (8, 22), 0, 0.6, (255, 255, 255), 2)
            tiles.append(t)
    vw.release()
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(os.path.join(out, "sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", os.path.join(out, "overlay_raw.avi"), "-c:v", "libx264",
                    "-crf", "20", "-pix_fmt", "yuv420p", os.path.join(out, "overlay.mp4")], check=True)
    os.remove(os.path.join(out, "overlay_raw.avi"))
    dr = np.array(drift) if drift else np.array([np.nan])
    print(json.dumps({"plane_inliers": int(inl.sum()), "points_below": int(len(Pb)),
                      "inlier_frac": round(float(inl.mean()), 3), "tol": round(float(tol), 4),
                      "stick_samples": int(len(drift)), "stick_med_px": round(float(np.nanmedian(dr)), 2),
                      "stick_p90_px": round(float(np.nanpercentile(dr, 90)), 2),
                      "drift_px_per_frame_med": round(float(np.median(drift_rate)) if drift_rate else -1, 3)}))


if __name__ == "__main__":
    main()
