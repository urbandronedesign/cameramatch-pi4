"""Equirectangular environment map built from the plate (for CG reflections / lighting).

python /tools/build_plate_envmap.py /work/<job> [--width 2048] [--frame_step 3]
1. Every frame's pixels -> undistorted rays (refined camera + Polynomial lens) -> Blender world directions
   -> accumulated into an equirect panorama (foreground matte excluded).
2. Unseen directions behind the camera: mirror of the front (reflection across the plane orthogonal to the
   mean viewing direction) - the classic plate-to-environment trick.
3. Remaining holes (zenith/nadir) inpainted at low res and blurred.
Blender equirect convention: u = -atan2(y, x)/2pi + 0.5, v = atan2(z, hypot(x, y))/pi + 0.5.
Writes /work/<job>/plate_env.png (+ plate_env_coverage.png).
"""
import argparse
import os

import cv2
import numpy as np

CV_TO_BL_WORLD = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def dir_to_uv(d):
    u = -np.arctan2(d[..., 1], d[..., 0]) / (2 * np.pi) + 0.5
    v = np.arctan2(d[..., 2], np.hypot(d[..., 0], d[..., 1])) / np.pi + 0.5
    return u, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--frame_step", type=int, default=3)
    a = ap.parse_args()
    r = np.load(os.path.join(a.job, "result.npz"))
    with np.load(os.path.join(a.job, "pi4_cache.npz")) as z:
        src = str(z["src"])
    c2w = r["c2w_refined"] if "c2w_refined" in r else r["c2w"]
    W, H = (int(v) for v in r["plate_size"])
    f, k1, k2 = float(r["f_px"]), float(r["k1"]), float(r["k2"])
    cx, cy = (float(v) for v in r["principal_px"])
    idx = r["frame_indices"].astype(int)
    anchor = CV_TO_BL_WORLD @ np.linalg.inv(c2w[0])
    EW, EH = a.width, a.width // 2

    s = 2
    vv, uu = np.mgrid[0:H:s, 0:W:s].astype(np.float64)
    xd, yd = (uu - cx) / f, (vv - cy) / f
    xn, yn = xd.copy(), yd.copy()
    for _ in range(30):
        q = xn * xn + yn * yn
        dq = 1 + k1 * q + k2 * q * q
        xn, yn = xd / dq, yd / dq
    rays = np.stack([xn, yn, np.ones_like(xn)], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    cap = cv2.VideoCapture(os.path.join(a.job, src))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    acc = np.zeros((EH, EW, 3), np.float64)
    cnt = np.zeros((EH, EW), np.float64)
    fwd = np.zeros(3)
    mdir = os.path.join(a.job, "matte", "fused")
    for i in range(0, len(c2w), a.frame_step):
        R = (anchor @ c2w[i])[:3, :3]
        fwd += R @ np.array([0, 0, 1.0])
        d = rays @ R.T
        u, v = dir_to_uv(d)
        px = np.clip((u * EW).astype(int), 0, EW - 1)
        py = np.clip(((1 - v) * EH).astype(int), 0, EH - 1)
        img = frames[idx[i]][::s, ::s].astype(np.float64)
        al = cv2.imread(os.path.join(mdir, f"alpha_{idx[i] + 1:04d}.png"), 0)
        ok = (cv2.dilate(al, np.ones((9, 9), np.uint8))[::s, ::s] < 30) if al is not None else np.ones(px.shape, bool)
        np.add.at(acc, (py[ok], px[ok]), img[ok])
        np.add.at(cnt, (py[ok], px[ok]), 1)
    seen = cnt > 0
    pano = np.zeros((EH, EW, 3), np.float32)
    pano[seen] = (acc[seen] / cnt[seen, None]).astype(np.float32)
    # close pinholes inside the seen region
    k = np.ones((5, 5), np.uint8)
    seen_c = cv2.morphologyEx(seen.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    holes = (seen_c & ~seen).astype(np.uint8) * 255
    pano = cv2.inpaint(np.clip(pano, 0, 255).astype(np.uint8), holes, 3, cv2.INPAINT_TELEA).astype(np.float32)
    seen = seen_c
    print(f"plate covers {seen.mean() * 100:.1f}% of the sphere")

    if seen.mean() < 0.10:
        # Narrow lens: a true projection covers almost nothing. Use the classic stretched-plate environment:
        # actor-free temporal median plate over the front hemisphere (+-90 deg az, +-45 deg el), mirrored
        # behind; beyond +-45 deg elevation, the top/bottom rows fade into blurred sky / ground colour.
        stack, mstack = [], []
        for i in range(0, len(c2w), max(1, len(c2w) // 24)):
            stack.append(cv2.resize(frames[idx[i]], (W // 2, H // 2), interpolation=cv2.INTER_AREA))
            al = cv2.imread(os.path.join(mdir, f"alpha_{idx[i] + 1:04d}.png"), 0)
            al = cv2.resize(al, (W // 2, H // 2)) if al is not None else np.zeros((H // 2, W // 2), np.uint8)
            mstack.append(cv2.dilate(al, np.ones((15, 15), np.uint8)) > 30)
        st_ = np.array(stack, np.float32)
        ms_ = np.array(mstack)
        clean = np.median(st_, axis=0)
        # actors that stay put are never seen past: inpaint everything they occupy in >15% of frames
        actor = ms_.mean(0) > 0.15
        actor = cv2.dilate(actor.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
        small = cv2.resize(clean.astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        sm_m = cv2.resize(actor.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        fill = cv2.inpaint(small, sm_m * 255, 15, cv2.INPAINT_TELEA)
        fill = cv2.GaussianBlur(cv2.resize(fill, (clean.shape[1], clean.shape[0]), interpolation=cv2.INTER_CUBIC),
                                (0, 0), 6).astype(np.float32)
        a_ = cv2.GaussianBlur(actor.astype(np.float32), (0, 0), 8)[..., None]
        clean = (clean * (1 - a_) + fill * a_).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(a.job, "plate_clean.png"), clean)
        fwd /= np.linalg.norm(fwd) if np.linalg.norm(fwd) > 0 else 1
        fh = fwd - np.array([0, 0, fwd[2]])
        fh /= np.linalg.norm(fh)
        right = np.cross(fh, [0, 0, 1.0])
        ey, ex = np.mgrid[0:EH, 0:EW].astype(np.float64)
        lon = -((ex + 0.5) / EW - 0.5) * 2 * np.pi
        lat = (0.5 - (ey + 0.5) / EH) * np.pi
        D = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
        az = np.arctan2(D @ right, D @ fh)                 # 0 = forward, +right
        az = np.where(np.abs(az) > np.pi / 2, np.sign(az) * np.pi - az, az)  # mirror behind
        el = np.arcsin(np.clip(D[..., 2], -1, 1))
        hh, ww = clean.shape[:2]
        mx = ((az / np.pi + 0.5) * ww).astype(np.float32)
        my = ((0.5 - np.clip(el, -np.pi / 4, np.pi / 4) / (np.pi / 2)) * hh).astype(np.float32)
        pano = cv2.remap(clean, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        lum = clean.astype(np.float32).mean(2)
        upper = clean[: hh // 2].reshape(-1, 3).astype(np.float32)
        ul = lum[: hh // 2].ravel()
        top = upper[ul >= np.percentile(ul, 92)].mean(0)    # overcast sky seen through the trees
        bot = cv2.GaussianBlur(clean[-max(2, hh // 12):].astype(np.float32), (0, 0), 25).mean((0, 1))
        # Fade early: stretched plate edges smear into streaks that mirror balls show on their lower half.
        deg = np.degrees(el)
        # below the horizon keep the plate's ground content, increasingly softened, before the ground colour
        soft = cv2.GaussianBlur(pano, (0, 0), 18)
        s_bl = np.clip((-deg - 5) / 25, 0, 1)[..., None]
        pano = pano * (1 - s_bl) + soft * s_bl
        t_top = np.clip((deg - 30) / 20, 0, 1)
        t_bot = np.clip((-deg - 25) / 35, 0, 1)
        t = np.where(el > 0, t_top, t_bot)[..., None]
        t = t * t * (3 - 2 * t)
        fade = np.where((el > 0)[..., None], top, bot)
        pano = pano * (1 - t) + fade * t
        pano = cv2.GaussianBlur(pano, (0, 0), 2.5)
        cv2.imwrite(os.path.join(a.job, "plate_env.png"), np.clip(pano, 0, 255).astype(np.uint8))
        print("narrow lens -> stretched-plate environment written (plate_env.png, plate_clean.png)")
        return

    # mirror the front into the unseen back: d' = d - 2 (d.f) f  (only where the front is seen)
    fwd /= np.linalg.norm(fwd)
    fwd_h = fwd - np.array([0, 0, fwd[2]])
    fwd_h /= np.linalg.norm(fwd_h)
    ey, ex = np.mgrid[0:EH, 0:EW].astype(np.float64)
    uu_, vv_ = (ex + 0.5) / EW, 1 - (ey + 0.5) / EH
    lon = -(uu_ - 0.5) * 2 * np.pi
    lat = (vv_ - 0.5) * np.pi
    D = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
    Dm = D - 2 * (D @ fwd_h)[..., None] * fwd_h
    mu, mv = dir_to_uv(Dm)
    mapx = (mu * EW - 0.5).astype(np.float32)
    mapy = ((1 - mv) * EH - 0.5).astype(np.float32)
    mirrored = cv2.remap(pano, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    mseen = cv2.remap(seen.astype(np.float32), mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP) > 0.5
    fill = ~seen & mseen
    pano[fill] = cv2.GaussianBlur(mirrored, (0, 0), 2.0)[fill]
    have = seen | fill
    print(f"after mirror fill: {have.mean() * 100:.1f}%")

    # zenith / nadir: inpaint at low res from the nearest seen colours, then blur
    sm = cv2.resize(pano, (256, 128), interpolation=cv2.INTER_AREA)
    hm = cv2.resize(have.astype(np.float32), (256, 128), interpolation=cv2.INTER_AREA) < 0.5
    sm = cv2.inpaint(np.clip(sm, 0, 255).astype(np.uint8), hm.astype(np.uint8) * 255, 12, cv2.INPAINT_TELEA)
    sm = cv2.GaussianBlur(sm.astype(np.float32), (0, 0), 6)
    big = cv2.resize(sm, (EW, EH), interpolation=cv2.INTER_CUBIC)
    wgt = cv2.GaussianBlur(have.astype(np.float32), (0, 0), 12)[..., None]
    out = pano * wgt + big * (1 - wgt)
    cv2.imwrite(os.path.join(a.job, "plate_env.png"), np.clip(out, 0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join(a.job, "plate_env_coverage.png"),
                (np.stack([seen, fill, ~have], -1) * 255).astype(np.uint8))
    print("wrote plate_env.png")


if __name__ == "__main__":
    main()
