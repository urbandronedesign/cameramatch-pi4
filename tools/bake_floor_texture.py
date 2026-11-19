"""Bake the video onto the floor: every texel takes its colour from the best frame that saw it.

python /tools/bake_floor_texture.py /work/<job> [--tpc 3] [--frame_step 1]
texel -> 3D (heightfield, bilinear) -> each frame: refined camera + Polynomial lens -> plate pixel.
Rejected: outside the image, behind the rider (fused matte), occluded (Pi4 depth much nearer than the
texel). Best = nearest camera (highest resolution) with a mild preference for the image centre.
Writes /work/<job>/floor_texture.png (+ floor_texture_coverage.png).
"""
import argparse
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--tpc", type=int, default=3, help="texels per grid cell")
    ap.add_argument("--frame_step", type=int, default=1)
    a = ap.parse_args()
    g = np.load(os.path.join(a.job, "geometry.npz"))
    r = np.load(os.path.join(a.job, "result.npz"))
    with np.load(os.path.join(a.job, "pi4_cache.npz")) as z:
        depth_all, src = z["depth"].astype(np.float32), str(z["src"])
    hg = g["hgrid"].astype(np.float64)
    ny, nx = hg.shape
    cell, x0, y0 = float(g["cell"]), float(g["grid_x0"]), float(g["grid_y0"])
    Rz, anchor = g["Rz"], g["anchor"]
    c2w = r["c2w_refined"]
    W, H = (int(v) for v in r["plate_size"])
    f, k1, k2 = float(r["f_px"]), float(r["k1"]), float(r["k2"])
    cx, cy = (float(v) for v in r["principal_px"])
    idx = r["frame_indices"].astype(int)

    # texel grid: texel centres in grid units (texture v grows with grid y; image row 0 = top -> flip later)
    t = a.tpc
    ty, tx = np.mgrid[0:ny * t, 0:nx * t].astype(np.float64)
    gxf, gyf = (tx + 0.5) / t - 0.5, (ty + 0.5) / t - 0.5          # in cell-centre coordinates
    hpad = hg.copy()
    # bilinear height (nan-aware: fall back to nearest cell)
    ix0, iy0 = np.clip(np.floor(gxf).astype(int), 0, nx - 1), np.clip(np.floor(gyf).astype(int), 0, ny - 1)
    ix1, iy1 = np.clip(ix0 + 1, 0, nx - 1), np.clip(iy0 + 1, 0, ny - 1)
    fx_, fy_ = np.clip(gxf - ix0, 0, 1), np.clip(gyf - iy0, 0, 1)
    h00, h10, h01, h11 = hpad[iy0, ix0], hpad[iy0, ix1], hpad[iy1, ix0], hpad[iy1, ix1]
    hb = (h00 * (1 - fx_) * (1 - fy_) + h10 * fx_ * (1 - fy_) + h01 * (1 - fx_) * fy_ + h11 * fx_ * fy_)
    hn = hpad[np.clip(np.round(gyf).astype(int), 0, ny - 1), np.clip(np.round(gxf).astype(int), 0, nx - 1)]
    hz = np.where(np.isnan(hb), hn, hb)
    valid = ~np.isnan(hz)
    Pg = np.stack([x0 + (gxf + 0.5) * cell, y0 + (gyf + 0.5) * cell, np.nan_to_num(hz)], -1)[valid]
    Pbl = Pg @ Rz                                  # gravity frame -> Blender world (Rz is orthonormal)
    inv_anchor = np.linalg.inv(anchor)
    Pcv = Pbl @ inv_anchor[:3, :3].T + inv_anchor[:3, 3]
    print(f"texture {nx * t}x{ny * t}, texels on floor {valid.sum()}")

    best = np.full(len(Pcv), np.inf)
    col = np.zeros((len(Pcv), 3), np.float32)
    cap = cv2.VideoCapture(os.path.join(a.job, src))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    mdir = os.path.join(a.job, "matte", "fused")
    for i in range(0, len(c2w), a.frame_step):
        w2c = np.linalg.inv(c2w[i])
        Xc = Pcv @ w2c[:3, :3].T + w2c[:3, 3]
        z = Xc[:, 2]
        front = z > 0.2
        xq, yq = Xc[:, 0] / np.where(front, z, 1), Xc[:, 1] / np.where(front, z, 1)
        q2 = xq * xq + yq * yq
        dq = 1 + k1 * q2 + k2 * q2 * q2
        u, v = f * xq * dq + cx, f * yq * dq + cy
        m = front & (u >= 1) & (u < W - 2) & (v >= 1) & (v < H - 2) & (q2 < 1.2)
        if not m.any():
            continue
        ui, vi = u[m].round().astype(int), v[m].round().astype(int)
        al = cv2.imread(os.path.join(mdir, f"alpha_{idx[i] + 1:04d}.png"), 0)
        ok = np.ones(m.sum(), bool) if al is None else cv2.dilate(al, np.ones((7, 7), np.uint8))[vi, ui] < 40
        D = cv2.resize(depth_all[i], (W, H), interpolation=cv2.INTER_LINEAR)[vi, ui]
        ok &= D > 0.75 * z[m]                       # something much nearer => texel occluded this frame
        score = z[m] * (1 + 0.6 * q2[m])            # nearer = sharper; prefer the image centre
        better = ok & (score < best[m])
        sel = np.nonzero(m)[0][better]
        best[sel] = score[better]
        img = frames[idx[i]].astype(np.float32)
        us, vs = u[sel], v[sel]
        u0, v0 = np.floor(us).astype(int), np.floor(vs).astype(int)
        du, dv = (us - u0)[:, None], (vs - v0)[:, None]
        col[sel] = (img[v0, u0] * (1 - du) * (1 - dv) + img[v0, u0 + 1] * du * (1 - dv)
                    + img[v0 + 1, u0] * (1 - du) * dv + img[v0 + 1, u0 + 1] * du * dv)
    cov = np.isfinite(best)
    print(f"coverage {cov.mean() * 100:.1f}% of floor texels")
    tex = np.full((ny * t, nx * t, 3), 60, np.uint8)
    flat = tex.reshape(-1, 3)
    vidx = np.nonzero(valid.ravel())[0]
    flat[vidx[cov]] = col[cov].clip(0, 255).astype(np.uint8)
    # fill tiny gaps
    holes = np.zeros(ny * t * nx * t, np.uint8)
    holes[vidx[~cov]] = 255
    tex = cv2.inpaint(tex, holes.reshape(ny * t, nx * t), 3, cv2.INPAINT_TELEA)
    cv2.imwrite(os.path.join(a.job, "floor_texture.png"), cv2.flip(tex, 0))  # image row 0 = top = max v
    cv2.imwrite(os.path.join(a.job, "floor_texture_holes.png"), cv2.flip(holes.reshape(ny * t, nx * t), 0))
    print("wrote floor_texture.png")


if __name__ == "__main__":
    main()
