"""Reconstruct a floor heightfield (stairs + floors) and building facades from a CameraMatch job.

python /tools/build_scene_geometry.py /work/<job> [--cell C] [--frame_step 2] [--px_step 2]
Uses the refined camera (result.npz: c2w_refined, f_px, k1, k2), Pi4 per-frame depth (pi4_cache.npz)
and the fused foreground matte (matte/fused) to back-project static pixels to 3D.
Writes /work/<job>/geometry.npz in BLENDER world coordinates (same anchoring as the add-on import):
  floor_verts [N,3], floor_faces [M,4], panels [K,4,3] (quad corners), up [3]
and geometry_debug.png (top view).
"""
import argparse
import os

import cv2
import numpy as np

CV_TO_BL_CAM = np.diag([1.0, -1.0, -1.0, 1.0])
CV_TO_BL_WORLD = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def rot_to_z(up):
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, z)
    s, c = np.linalg.norm(v), up @ z
    if s < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def nanmedian_filter(g, r=1):
    p = np.pad(g, r, constant_values=np.nan)
    w = np.lib.stride_tricks.sliding_window_view(p, (2 * r + 1, 2 * r + 1))
    with np.errstate(all="ignore"):
        return np.nanmedian(w.reshape(*g.shape, -1), axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--cell", type=float, default=0.0, help="grid cell (scene units); 0 = auto")
    ap.add_argument("--frame_step", type=int, default=2)
    ap.add_argument("--px_step", type=int, default=2)
    ap.add_argument("--source", choices=["ba", "pi4"], default="ba",
                    help="ba: bundle-adjusted track points (consistent with the solved camera); "
                         "pi4: dense Pi4 depth (denser, but can disagree with the cameras on odd lenses)")
    a = ap.parse_args()

    r = np.load(os.path.join(a.job, "result.npz"))
    with np.load(os.path.join(a.job, "pi4_cache.npz")) as z:
        depth_all = z["depth"].astype(np.float32)
    c2w = r["c2w_refined"]
    T = len(c2w)
    W, H = (int(v) for v in r["plate_size"])
    f, k1, k2 = float(r["f_px"]), float(r["k1"]), float(r["k2"])
    cx, cy = (float(v) for v in r["principal_px"])
    idx = r["frame_indices"].astype(int)
    anchor = CV_TO_BL_WORLD @ np.linalg.inv(c2w[0])
    Rb = anchor[:3, :3]

    # Undistorted normalized rays for the (strided) pixel grid.
    s = a.px_step
    vv, uu = np.mgrid[0:H:s, 0:W:s].astype(np.float64)
    xd, yd = (uu - cx) / f, (vv - cy) / f
    xn, yn = xd.copy(), yd.copy()
    for _ in range(30):
        r2 = xn * xn + yn * yn
        dd = 1 + k1 * r2 + k2 * r2 * r2
        xn, yn = xd / dd, yd / dd

    # Average camera "up" (Blender world) = shot's gravity estimate.
    cam_up = np.array([Rb @ (c2w[i, :3, :3] @ np.array([0, -1.0, 0])) for i in range(T)])
    up = cam_up.mean(0)
    up /= np.linalg.norm(up)
    Rz = rot_to_z(up)  # Blender world -> gravity-aligned frame
    centers = np.array([(anchor @ c2w[i])[:3, 3] for i in range(T)])

    G, Gc, F, Fc, Ffi = [], [], [], [], []
    mdir = os.path.join(a.job, "matte", "fused")
    frames_iter = range(0, T, a.frame_step) if a.source == "pi4" else []
    if a.source == "ba":
        # BA track points -> gravity frame; classify against the nearest camera (path position).
        P = (r["track_points"].astype(np.float64) @ anchor[:3, :3].T + anchor[:3, 3]) @ Rz.T
        Cg = centers @ Rz.T
        d2 = ((P[:, None, :2] - Cg[None, :, :2]) ** 2).sum(-1)
        near_i = d2.argmin(1)
        lat = np.sqrt(d2.min(1))
        h = P[:, 2] - Cg[near_i, 2]
        dist = np.linalg.norm(P - Cg[near_i], axis=1)
        gsel = (h < -0.4) & (lat < 8.0)
        fsel = (h > 1.5)
        G.append(P[gsel]); Gc.append(dist[gsel])
        F.append(P[fsel]); Fc.append(dist[fsel]); Ffi.append(near_i[fsel])
        print(f"BA points {len(P)}: ground cand {gsel.sum()}, facade cand {fsel.sum()}")
    for i in frames_iter:
        Z = cv2.resize(depth_all[i], (W, H), interpolation=cv2.INTER_LINEAR)[::s, ::s].astype(np.float64)
        al = cv2.imread(os.path.join(mdir, f"alpha_{idx[i] + 1:04d}.png"), 0)
        fg = (al[::s, ::s] > 60) if al is not None else np.zeros_like(Z, bool)
        Xc = np.stack([xn * Z, yn * Z, Z], -1)
        du, dv = np.gradient(Xc, axis=1), np.gradient(Xc, axis=0)
        n = np.cross(du, dv)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
        flip = (n * Xc).sum(-1) > 0
        n[flip] *= -1
        gz = np.hypot(*np.gradient(Z)) / np.maximum(Z, 1e-6)
        ok = (~fg) & (Z > 0) & (gz < 0.08)
        M = anchor @ c2w[i]
        Pw = (Xc[ok] @ M[:3, :3].T + M[:3, 3]) @ Rz.T
        Nw = (n[ok] @ M[:3, :3].T) @ Rz.T
        Cz = Rz @ centers[i]
        nz = Nw[:, 2]
        below = Pw[:, 2] < Cz[2]
        ground = (nz > 0.85) & below
        facade = (np.abs(nz) < 0.2) & (Pw[:, 2] > Cz[2] + 1.5)  # building walls, not railings
        G.append(Pw[ground]); Gc.append(Z[ok][ground])
        F.append(Pw[facade]); Fc.append(Z[ok][facade]); Ffi.append(np.full(int(facade.sum()), i))
    G, Gd, F, Fd = np.concatenate(G), np.concatenate(Gc), np.concatenate(F), np.concatenate(Fc)
    Ffi = np.concatenate(Ffi)
    C = centers @ Rz.T
    print(f"ground pts {len(G)}, facade pts {len(F)}, ground depth median {np.median(Gd):.2f}, "
          f"cam height above ground (median) {np.median(C[:, 2]) - np.median(G[:, 2]):.2f}, "
          f"path length {np.linalg.norm(np.diff(C, axis=0), axis=1).sum():.1f}")

    # ---- floor heightfield (gravity frame)
    if a.source == "ba":
        cell = a.cell or 0.35
        Gn = G
    else:
        cell = a.cell or float(np.clip(0.04 * np.median(Gd), 0.05, 2.0))
        near = Gd < np.percentile(Gd, 85)  # far ground is noisy
        Gn = G[near]
    x0, y0 = Gn[:, 0].min(), Gn[:, 1].min()
    ix = ((Gn[:, 0] - x0) / cell).astype(int)
    iy = ((Gn[:, 1] - y0) / cell).astype(int)
    nx, ny = ix.max() + 1, iy.max() + 1
    key = iy * nx + ix
    order = np.argsort(key)
    ks, zs = key[order], Gn[order, 2]
    uk, start, cnt = np.unique(ks, return_index=True, return_counts=True)
    hgrid = np.full(nx * ny, np.nan)
    min_cnt, q = (1, 30) if a.source == "ba" else (3, 50)  # BA: sparse; low quantile favours ground
    good = cnt >= min_cnt
    hgrid[uk[good]] = [np.percentile(zs[st:st + c], q) for st, c in zip(start[good], cnt[good])]
    hgrid = hgrid.reshape(ny, nx)
    hgrid = nanmedian_filter(hgrid, 1)          # despeckle, keeps step edges
    for _ in range(4 if a.source == "ba" else 1):  # close holes (sparse BA needs a few passes)
        filled = nanmedian_filter(hgrid, 1)
        hgrid = np.where(np.isnan(hgrid), filled, hgrid)
    if a.source == "ba":  # keep the corridor around the camera path
        gy_, gx_ = np.mgrid[0:ny, 0:nx]
        cxy = np.stack([x0 + (gx_ + 0.5) * cell, y0 + (gy_ + 0.5) * cell], -1).reshape(-1, 2)
        Cg2 = (centers @ Rz.T)[:, :2]
        dmin = np.full(len(cxy), np.inf)
        for k in range(0, len(Cg2), 3):
            dmin = np.minimum(dmin, np.linalg.norm(cxy - Cg2[k], axis=1))
        hgrid[(dmin > 7.0).reshape(ny, nx)] = np.nan
    valid = ~np.isnan(hgrid)
    print(f"floor grid {nx}x{ny}, cell {cell:.3f}, valid cells {valid.sum()}")

    vid = -np.ones((ny, nx), int)
    vid[valid] = np.arange(valid.sum())
    gy, gx = np.nonzero(valid)
    verts = np.stack([x0 + (gx + 0.5) * cell, y0 + (gy + 0.5) * cell, hgrid[valid]], 1)
    jump = 8 * cell
    q = np.stack([vid[:-1, :-1], vid[:-1, 1:], vid[1:, 1:], vid[1:, :-1]], -1).reshape(-1, 4)
    q = q[(q >= 0).all(1)]
    hz = verts[q][:, :, 2]
    faces = q[(hz.max(1) - hz.min(1)) < jump]

    # ---- facades: sequential RANSAC of vertical planes (= 2D lines in the gravity frame's XY), run in
    # sliding frame windows so every building is found while the camera is near it, then de-duplicated.
    rng = np.random.default_rng(0)

    def ransac_facades(Pts, thr, max_planes, min_pts):
        found, remaining = [], Pts.copy()
        for _ in range(max_planes):
            if len(remaining) < min_pts:
                break
            best = None
            for _ in range(400):
                p, q2 = remaining[rng.choice(len(remaining), 2, replace=False)]
                d = q2[:2] - p[:2]
                if np.linalg.norm(d) < 5 * thr:
                    continue
                nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
                inl = np.abs((remaining[:, :2] - p[:2]) @ nrm) < thr
                if best is None or inl.sum() > best[0].sum():
                    best = (inl, nrm, p)
            if best is None:
                break
            inl, nrm, p = best
            tdir = np.array([nrm[1], -nrm[0]])
            # collinear points from unrelated walls: split into contiguous segments, keep the densest
            tt_all = (remaining[inl][:, :2] - p[:2]) @ tdir
            o = np.argsort(tt_all)
            segs = np.split(o, np.nonzero(np.diff(tt_all[o]) > 2.0)[0] + 1)
            inl_idx = np.nonzero(inl)[0][max(segs, key=len)]
            pts = remaining[inl_idx]
            keep_mask = np.ones(len(remaining), bool)
            keep_mask[inl_idx] = False
            remaining = remaining[keep_mask]
            if len(pts) < min_pts:
                continue
            tt = (pts[:, :2] - p[:2]) @ tdir
            t0, t1 = np.percentile(tt, [5, 95])
            z0, z1 = np.percentile(pts[:, 2], [10, 90])
            base = p[:2] + nrm * float(np.median((pts[:, :2] - p[:2]) @ nrm))
            if np.median((C[:, :2] - base) @ nrm) < 0:  # face the cameras
                nrm = -nrm
            L, Hh = t1 - t0, z1 - z0
            if L < 6 * thr or Hh < 3.0 or L > 80:
                continue
            found.append(dict(base=base, tdir=tdir, nrm=nrm, t0=t0, t1=t1, z0=z0, z1=z1, n=len(pts), thr=thr))
        return found

    facades = []
    win, step_w = 40, 20
    for s0 in range(0, T, step_w):
        sel = (Ffi >= s0) & (Ffi < s0 + win)
        if sel.sum() < (3000 if a.source == "pi4" else 60):
            continue
        Pw_ = F[sel]
        Pw_ = Pw_[rng.choice(len(Pw_), min(len(Pw_), 60000), replace=False)]
        thr_w = 0.02 * float(np.median(Fd[sel])) if a.source == "pi4" else 0.35
        for fc in ransac_facades(Pw_, thr_w, 4, max(25, int(0.05 * len(Pw_)))):
            dup = None
            for k, e in enumerate(facades):
                if abs(fc["nrm"] @ e["nrm"]) > 0.95 and abs((fc["base"] - e["base"]) @ e["nrm"]) < 3 * max(
                        fc["thr"], e["thr"]):
                    a0, a1 = sorted([(fc["base"] + fc["tdir"] * fc["t0"] - e["base"]) @ e["tdir"],
                                     (fc["base"] + fc["tdir"] * fc["t1"] - e["base"]) @ e["tdir"]])
                    if min(a1, e["t1"]) - max(a0, e["t0"]) > 0:
                        dup = k
                        break
            if dup is None:
                facades.append(fc)
            elif fc["n"] > facades[dup]["n"]:
                facades[dup] = fc
    print(f"facades found (windowed, de-duplicated): {len(facades)}")

    panels, lines = [], []
    for fc in facades:
        base, tdir, nrm, t0, t1, z0, z1, thr = (fc[k] for k in ("base", "tdir", "nrm", "t0", "t1", "z0", "z1", "thr"))
        L, Hh = t1 - t0, z1 - z0
        lines.append((base, tdir, t0, t1, fc["n"]))
        rows = 2 if Hh > 0.3 * L else 1
        cols = int(np.clip(round(L / max(Hh / rows, 1e-6) * 0.8), 2, 6))
        gap = 0.12
        pw, ph = L / cols, Hh / rows
        for ci in range(cols):
            for ri in range(rows):
                a0 = t0 + (ci + gap / 2) * pw
                a1 = t0 + (ci + 1 - gap / 2) * pw
                b0 = z0 + (ri + gap / 2) * ph
                b1 = z0 + (ri + 1 - gap / 2) * ph
                c2d = lambda tv: base + tdir * tv + nrm * (0.6 * thr)  # noqa: E731 (slightly off the wall)
                quad = [np.r_[c2d(a0), b0], np.r_[c2d(a1), b0], np.r_[c2d(a1), b1], np.r_[c2d(a0), b1]]
                panels.append(quad)
        print(f"facade: {fc['n']} pts, length {L:.1f}, height {Hh:.1f}, {cols}x{rows} panels")

    # back to Blender world
    Rinv = Rz.T
    floor_verts = verts @ Rinv.T
    panels = np.array(panels).reshape(-1, 4, 3) @ Rinv.T if panels else np.zeros((0, 4, 3))

    # Validate every panel against the footage: where a frame sees the panel, Pi4's depth at those
    # pixels must agree with the panel's depth. Panels fitted to reflections / glass float in front of
    # sky or far scenery and fail.
    if len(panels):
        inv_anchor = np.linalg.inv(anchor)
        keep = []
        for quad in panels:
            ctr = quad.mean(0)
            samples = np.array([ctr] + [ctr + 0.6 * (c - ctr) for c in quad])  # centre + inner corners
            Pcv = samples @ inv_anchor[:3, :3].T + inv_anchor[:3, 3]
            agree = seen = 0
            for i in range(0, T, 3):
                w2c = np.linalg.inv(c2w[i])
                Xc = Pcv @ w2c[:3, :3].T + w2c[:3, 3]
                if (Xc[:, 2] <= 0.1).any():
                    continue
                xq, yq = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
                q2 = xq * xq + yq * yq
                dq = 1 + k1 * q2 + k2 * q2 * q2
                uq, vq = f * xq * dq + cx, f * yq * dq + cy
                if not ((uq >= 0) & (uq < W) & (vq >= 0) & (vq < H)).all():
                    continue
                al = cv2.imread(os.path.join(mdir, f"alpha_{idx[i] + 1:04d}.png"), 0)
                ui, vi = uq.astype(int), vq.astype(int)
                if al is not None and (al[vi, ui] > 60).any():
                    continue  # behind the rider this frame: no evidence either way
                D = cv2.resize(depth_all[i], (W, H), interpolation=cv2.INTER_LINEAR)[vi, ui]
                seen += 1
                agree += int(np.mean(np.abs(np.log(D / Xc[:, 2])) < np.log(1.25)) >= 0.6)
            ok = seen >= 5 and agree / seen >= 0.5
            keep.append(ok)
        keep = np.array(keep, bool)
        print(f"panel validation: kept {keep.sum()}/{len(keep)}")
        panels = panels[keep]
    floor_uv = np.stack([(gx + 0.5) / nx, (gy + 0.5) / ny], 1)
    np.savez(os.path.join(a.job, "geometry.npz"), floor_verts=floor_verts.astype(np.float32),
             floor_faces=faces.astype(np.int32), panels=panels.astype(np.float32), up=up, cell=cell,
             floor_uv=floor_uv.astype(np.float32), hgrid=hgrid.astype(np.float32), grid_x0=x0, grid_y0=y0,
             Rz=Rz, anchor=anchor)
    print(f"floor: {len(floor_verts)} verts, {len(faces)} quads; panels: {len(panels)}")

    # debug top view
    allp = np.concatenate([G[:, :2], F[:, :2], C[:, :2]])
    lo, hi = np.percentile(allp, 1, 0), np.percentile(allp, 99, 0)
    S = 900 / max(hi - lo)
    img = np.full((int((hi - lo)[1] * S) + 20, int((hi - lo)[0] * S) + 20, 3), 30, np.uint8)
    to = lambda P: ((P[:, :2] - lo) * S + 10).astype(int)  # noqa: E731
    hz = (verts[:, 2] - np.percentile(verts[:, 2], 2)) / max(np.ptp(verts[:, 2]), 1e-6)
    for (x, y), h in zip(to(verts), hz):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            img[y, x] = cv2.applyColorMap(np.uint8([[int(255 * np.clip(h, 0, 1))]]), cv2.COLORMAP_VIRIDIS)[0, 0]
    for x, y in to(F[::50]):
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            img[y, x] = (90, 90, 255)
    for base, tdir, t0, t1, _ in lines:
        p0, p1 = to(np.array([base + tdir * t0, base + tdir * t1]))
        cv2.line(img, tuple(p0), tuple(p1), (0, 215, 255), 3)
    cv2.polylines(img, [to(C).reshape(-1, 1, 2)], False, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(a.job, "geometry_debug.png"), cv2.flip(img, 0))


if __name__ == "__main__":
    main()
