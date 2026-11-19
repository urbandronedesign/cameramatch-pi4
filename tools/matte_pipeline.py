"""Foreground matte refinement for a CameraMatch job: Pi4 mask -> SAM 2.1 -> MatAnyone 2.

docker compose --profile tools run --rm matte python /tools/matte_pipeline.py /work/<job> [--methods ...]

Writes /work/<job>/matte/<method>/alpha_####.png (scene-frame numbered, 8-bit alpha, plate res),
metrics.json and crops.jpg. Methods:
  pi4                 Pi4 motion mask (as used so far: >0.5, dilate 5px, blur 5)
  sam2                SAM 2.1 video masks, anchored every --anchor frames by SAM-refined Pi4 prompts
  matanyone2          MatAnyone 2 (HF PeiqingYang/MatAnyone2) seeded once with SAM-refined frame-0 mask
  matanyone2_chunked  MatAnyone 2 re-seeded from SAM 2 every --chunk frames, crossfaded over --overlap
MatAnyone 2 is NTU S-Lab License 1.0 (non-commercial); SAM 2.1 is Apache-2.0.
"""
import argparse
import json
import os
import time
import urllib.request

import cv2
import numpy as np
import torch

SAM2_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


LOG_HOOK = None        # set by matte_server: callable(str) -> per-job log
PROGRESS_HOOK = None   # set by matte_server: callable(stage, frac, msg)


def log(*a):
    line = " ".join(str(x) for x in a)
    print("[matte]", line, flush=True)
    if LOG_HOOK:
        LOG_HOOK(line)


def prog(stage, frac, msg=None):
    if PROGRESS_HOOK:
        PROGRESS_HOOK(stage, frac, msg)


# --------------------------------------------------------------------------- io
def load_frames(src, indices):
    wanted = set(int(i) for i in indices)
    out = {}
    if os.path.isdir(src):
        files = sorted(f for f in os.listdir(src) if os.path.splitext(f)[1].lower() in IMG_EXT)
        for i in sorted(wanted):
            out[i] = cv2.cvtColor(cv2.imread(os.path.join(src, files[i])), cv2.COLOR_BGR2RGB)
    else:
        cap = cv2.VideoCapture(src)
        i = 0
        while i <= max(wanted):
            ok, f = cap.read()
            if not ok:
                break
            if i in wanted:
                out[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            i += 1
        cap.release()
    return np.stack([out[int(i)] for i in indices])


# --------------------------------------------------------------------------- SAM 2.1
def sam2_paths():
    ck = "/cache/sam2/sam2.1_hiera_large.pt"
    if not os.path.exists(ck):
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        log("downloading SAM 2.1 large ...")
        urllib.request.urlretrieve(SAM2_URL, ck)
    return SAM2_CFG, ck


def prompts_from_mask(m, rng):
    """Per connected component: box, interior positive points, near-outside negative points."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    H, W = m.shape
    comps = []
    for c in range(1, n):
        x, y, w, h, area = stats[c]
        if area < 0.003 * H * W:
            continue
        comp = (lab == c).astype(np.uint8)
        dt = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        pos = []
        d = dt.copy()
        for _ in range(4):
            yy, xx = np.unravel_index(np.argmax(d), d.shape)
            if d[yy, xx] < 2:
                break
            pos.append((xx, yy))
            cv2.circle(d, (int(xx), int(yy)), int(max(6, 0.6 * dt[yy, xx])), 0, -1)
        ring = cv2.dilate(comp, np.ones((31, 31), np.uint8)) - cv2.dilate(comp, np.ones((13, 13), np.uint8))
        ring[m > 0] = 0
        ry, rx = np.nonzero(ring)
        neg = [(rx[k], ry[k]) for k in rng.choice(len(rx), min(4, len(rx)), replace=False)] if len(rx) else []
        pad = 0.03 * max(W, H)
        box = np.array([max(0, x - pad), max(0, y - pad), min(W - 1, x + w + pad), min(H - 1, y + h + pad)])
        comps.append((box, pos, neg))
    return comps


def sam_refine_frame(pred, img, pi4_mask, rng):
    pred.set_image(img)
    out = np.zeros(pi4_mask.shape, bool)
    for box, pos, neg in prompts_from_mask(pi4_mask, rng):
        pts = np.array(pos + neg, np.float32) if (pos or neg) else None
        lbl = np.array([1] * len(pos) + [0] * len(neg), np.int32) if pts is not None else None
        masks, scores, _ = pred.predict(point_coords=pts, point_labels=lbl, box=box, multimask_output=False)
        out |= masks[0] > 0
    return out


# --------------------------------------------------------------------------- MatAnyone 2
_MA = {}


def matanyone_model():
    if "m" not in _MA:
        from matanyone2 import MatAnyone2
        m = MatAnyone2.from_pretrained("PeiqingYang/MatAnyone2").cuda().eval()
        _MA["m"] = m
    return _MA["m"]


@torch.inference_mode()
def matanyone_run(frames, seed_mask, n_warmup=10, r=3):
    """frames: [T,H,W,3] uint8 RGB; seed_mask: HxW bool for frames[0]. Returns [T,H,W] float alpha."""
    from matanyone2.inference.inference_core import InferenceCore
    from matanyone2.utils.inference_utils import gen_dilate, gen_erosion

    model = matanyone_model()
    proc = InferenceCore(model, cfg=model.cfg)
    m = (seed_mask.astype(np.uint8) * 255)
    if r > 0:  # same unknown-band trick as upstream (scaled down for small plates)
        m = gen_erosion(gen_dilate(m, r, r), r, r)
    mask = torch.from_numpy(m).float().cuda()
    vf = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    vf = torch.cat([vf[:1].repeat(n_warmup, 1, 1, 1), vf], 0)
    alphas = []
    with torch.amp.autocast("cuda"):
        for ti in range(vf.shape[0]):
            image = (vf[ti] / 255.0).cuda()
            if ti == 0:
                proc.step(image, mask, objects=[1])
                prob = proc.step(image, first_frame_pred=True)
            elif ti <= n_warmup:
                prob = proc.step(image, first_frame_pred=True)
            else:
                prob = proc.step(image)
            a = proc.output_prob_to_mask(prob).float().cpu().numpy()
            if ti >= n_warmup:
                alphas.append(a)
    return np.clip(np.stack(alphas), 0, 1)


# --------------------------------------------------------------------------- metrics
def metrics(alpha, frames):
    T = len(alpha)
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    H, W = gray[0].shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    edge_d, edge_in, flick = [], [], []
    for t in range(T):
        e = cv2.Canny(gray[t], 50, 150)
        dist = cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2, 3)
        b = (alpha[t] > 0.5).astype(np.uint8)
        cont = cv2.morphologyEx(b, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        if cont.any():
            edge_d.append(np.median(dist[cont]))
            edge_in.append(np.mean(dist[cont] <= 1.5))
        if t > 0:
            flow = dis.calc(gray[t], gray[t - 1], None)  # t -> t-1
            warped = cv2.remap(alpha[t - 1].astype(np.float32), gx + flow[..., 0], gy + flow[..., 1],
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            flick.append(np.mean(np.abs(alpha[t] - warped)) * 1000)
    soft = np.mean((alpha > 0.05) & (alpha < 0.95)) * 100
    fl = [((alpha[t] > 0.5) & ~border_connected(alpha[t] > 0.5)).sum() / max((alpha[t] > 0.5).sum(), 1)
          for t in range(T)]
    return dict(floating_area_pct=float(np.mean(fl) * 100),edge_dist_px=float(np.mean(edge_d)), edge_within_1p5px=float(np.mean(edge_in)),
                flicker_x1000=float(np.mean(flick)), soft_pixels_pct=float(soft),
                fg_area_pct=float(np.mean(alpha) * 100))


def border_connected(m, min_frac=0.002):
    """Keep components touching the image border (camera-attached rig: bike, arms, visor);
    drop floating blobs (reflections, distant movers)."""
    m = m.astype(np.uint8)
    H, W = m.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    for c in range(1, n):
        x, y, w, h, area = stats[c]
        touches = x == 0 or y == 0 or x + w >= W or y + h >= H
        keep[c] = touches and area >= min_frac * H * W
    return keep[lab]


def supported(m, support, frac=0.2, min_px=30):
    """Keep components of m with at least `frac` of their area inside `support` (dilated Pi4 motion mask).
    General rule for people / objects in a scene: a railing may cut a person into pieces that do not touch
    the frame border, but every real piece is backed by Pi4's motion mask; spurious blobs are not."""
    m = m.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool)
    inside = np.bincount(lab.ravel(), weights=support.ravel().astype(np.float64), minlength=n)
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    keep = (inside / np.maximum(area, 1) >= frac) & (area >= min_px)
    keep[0] = False
    return keep[lab]


def clean_alpha(al, mode="rig", support=None):
    """Zero soft alpha belonging to unsupported components (rig: not border-connected; scene: not backed by
    the Pi4 motion mask)."""
    out = al.copy()
    k = np.ones((7, 7), np.uint8)
    for t in range(len(al)):
        b = al[t] > 0.5
        keepm = border_connected(b) if mode == "rig" else supported(b, support[t])
        drop = b & ~keepm
        if drop.any():
            out[t][cv2.dilate(drop.astype(np.uint8), k) > 0] = 0
    return out


def iou(a, b):
    a, b = a > 0.5, b > 0.5
    return float((a & b).sum() / max((a | b).sum(), 1))


# --------------------------------------------------------------------------- main
INV_SCALE = 4.0  # occlusion export: 16-bit PNG value = (1/depth) / INV_SCALE


def run(job, methods="fused", anchor=30, chunk=30, overlap=10, no_clean=False, write=None, crops=True,
        occlusion=False, mode="auto"):
    """Compute mattes for a job. methods: comma string or list; "fused" pulls in sam2 + matanyone2_seeded.
    write: methods whose alpha PNGs and metrics are written (default: all computed). Returns metrics."""
    from types import SimpleNamespace
    a = SimpleNamespace(job=job, anchor=anchor, chunk=chunk, overlap=overlap, no_clean=no_clean)
    methods = methods.split(",") if isinstance(methods, str) else list(methods)
    if "fused" in methods:
        methods = list(dict.fromkeys(["sam2", "matanyone2_seeded"] + methods))
    write = set(methods if write is None else write)
    rng = np.random.default_rng(0)

    # Materialise once: NpzFile is lazy and re-reads the whole array on every cache["key"] access
    # (that was ~110 s per run over the Windows bind mount).
    with np.load(os.path.join(a.job, "pi4_cache.npz")) as z:
        cache = {k: z[k] for k in z.files}
    res = np.load(os.path.join(a.job, "result.npz"))
    idx = res["frame_indices"].astype(int)
    prog("load", 0.2, "decoding frames")
    frames = load_frames(os.path.join(a.job, str(cache["src"])), idx)
    T, H, W, _ = frames.shape
    log(f"{T} frames {W}x{H}")
    prog("load", 1.0, f"{T} frames {W}x{H}")
    dyn = np.stack([cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR) for d in cache["dyn"]]) > 127
    support = np.stack([cv2.dilate(d.astype(np.uint8), np.ones((15, 15), np.uint8)) for d in dyn]) > 0
    if mode == "auto":
        # Camera-attached rig (POV bike, car hood) = foreground very close to the lens vs the scene.
        ratios = []
        for t in range(0, T, max(1, T // 20)):
            dm = cache["depth"][t].astype(np.float32)
            fg = cv2.resize(dyn[t].astype(np.uint8), (dm.shape[1], dm.shape[0])) > 0
            if fg.mean() > 0.01 and (~fg).any():
                ratios.append(np.median(dm[fg]) / max(np.median(dm[~fg]), 1e-6))
        ratio = float(np.median(ratios)) if ratios else 1.0
        mode = "rig" if ratio < 0.12 else "scene"
        log(f"matte mode auto -> {mode} (foreground/scene depth ratio {ratio:.2f})")
    a.mode = mode
    out_root = os.path.join(a.job, "matte")
    os.makedirs(out_root, exist_ok=True)
    alphas, times = {}, {}

    if "pi4" in methods:
        k = np.ones((5, 5), np.uint8)
        alphas["pi4"] = np.stack([cv2.GaussianBlur(cv2.dilate(m.astype(np.uint8) * 255, k), (5, 5), 0)
                                  for m in dyn]).astype(np.float32) / 255

    need_sam = any(m.startswith(("sam2", "matanyone2")) for m in methods)
    if need_sam:
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        cfg, ck = sam2_paths()
        t0 = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            ipred = SAM2ImagePredictor(build_sam2(cfg, ck, device="cuda"))
            anchors = list(range(0, T, a.anchor))
            seeds = {}
            for n_a, k in enumerate(anchors):
                prog("sam_seeds", n_a / len(anchors), f"SAM 2.1 refining Pi4 mask, keyframe {n_a + 1}/{len(anchors)}")
                seeds[k] = sam_refine_frame(ipred, frames[k], dyn[k], rng)
            log("SAM refine of Pi4 seeds, IoU vs Pi4:", [round(iou(seeds[k], dyn[k]), 3) for k in anchors])
            if not a.no_clean:
                seeds = {k: (border_connected(s) if a.mode == "rig" else supported(s, support[k]))
                         for k, s in seeds.items()}
            fdir = os.path.join(out_root, "jpg")
            os.makedirs(fdir, exist_ok=True)
            for i, f in enumerate(frames):
                cv2.imwrite(os.path.join(fdir, f"{i:05d}.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 98])
            vpred = build_sam2_video_predictor(cfg, ck, device="cuda")
            st = vpred.init_state(fdir)
            for k in anchors:
                vpred.add_new_mask(st, k, 1, torch.from_numpy(seeds[k]))
            sam = np.zeros((T, H, W), np.float32)
            for fi, _, logits in vpred.propagate_in_video(st):
                if fi % 5 == 0:
                    prog("sam_video", fi / T, f"SAM 2.1 video propagation frame {fi}/{T}")
                sam[fi] = (logits[0, 0] > 0).float().cpu().numpy()
        times["sam2"] = time.time() - t0
        alphas["sam2"] = sam
        del vpred, ipred
        torch.cuda.empty_cache()

    if "matanyone2" in methods:
        t0 = time.time()
        alphas["matanyone2"] = matanyone_run(frames, seeds[0])
        times["matanyone2"] = time.time() - t0

    if "matanyone2_chunked" in methods or "matanyone2_seeded" in methods:
        t0 = time.time()
        acc = np.zeros((T, H, W), np.float32)
        wsum = np.zeros((T, 1, 1), np.float32)
        start = 0
        while start < T:
            end = min(T, start + a.chunk + a.overlap)
            seed = seeds.get(start, alphas["sam2"][start] > 0.5)
            prog("matanyone", start / T, f"MatAnyone 2 chunk {start // a.chunk + 1}/{(T + a.chunk - 1) // a.chunk} "
                                         f"(frames {start}-{end})")
            al = matanyone_run(frames[start:end], seed)
            w = np.ones(end - start, np.float32)
            if start > 0:  # ramp in over the overlap with the previous chunk
                n = min(a.overlap, end - start)
                w[:n] = np.linspace(0, 1, n + 2)[1:-1]
            if end < T:
                n = min(a.overlap, end - start)
                w[-n:] = np.minimum(w[-n:], np.linspace(1, 0, n + 2)[1:-1])
            acc[start:end] += al * w[:, None, None]
            wsum[start:end, 0, 0] += w
            start += a.chunk
        chunked = acc / np.maximum(wsum, 1e-6)
        times["matanyone2_chunked"] = time.time() - t0
        if "matanyone2_chunked" in methods:
            alphas["matanyone2_chunked"] = chunked
        if "matanyone2_seeded" in methods:  # clean seeds (above) + floating-blob removal
            alphas["matanyone2_seeded"] = clean_alpha(chunked, a.mode, support)
            times["matanyone2_seeded"] = times["matanyone2_chunked"]

    if "fused" in methods:
        # MatAnyone 2 soft alpha where it has the part; SAM 2 structure (eroded 1px, feathered) fills
        # MatAnyone dropouts on thin/low-contrast parts. Both border-filtered (camera-attached rig).
        ma = alphas.get("matanyone2_seeded")
        if ma is None:
            raise SystemExit("fused needs matanyone2_seeded")
        # SAM-only regions are accepted only if they touch MatAnyone's matte (a dropout on an existing
        # part) or are near the camera per Pi4 depth (camera-attached rig) - distant reflections fail both.
        k = np.ones((3, 3), np.uint8)
        depth = np.stack([cv2.resize(d.astype(np.float32), (W, H)) for d in cache["depth"]])
        fused, n_rej = [], 0
        for t in range(T):
            if t % 10 == 0:
                prog("fuse", t / T, f"fusing MatAnyone + SAM frame {t}/{T}")
            s = alphas["sam2"][t] > 0.5
            s = border_connected(s) if a.mode == "rig" else supported(s, support[t])
            m_fg = ma[t] > 0.5
            only = (s & ~m_fg).astype(np.uint8)
            near_ma = cv2.dilate(m_fg.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            n, lab = cv2.connectedComponents(only, connectivity=8)
            if n > 1:  # vectorised per-component tests (thin edge slivers give hundreds of components)
                from scipy import ndimage
                attached = np.zeros(n, bool)
                attached[np.unique(lab[near_ma & (lab > 0)])] = True
                meds = np.asarray(ndimage.median(depth[t], lab, index=np.arange(1, n)))
                keep = np.ones(n, bool)
                if a.mode == "rig":
                    keep[1:] = attached[1:] | (meds < 0.3 * float(np.median(depth[t])))
                else:  # scene: SAM fills dropouts attached to the matte or backed by Pi4's motion mask
                    backed = np.bincount(lab.ravel(), weights=support[t].ravel().astype(np.float64), minlength=n)
                    area = np.bincount(lab.ravel(), minlength=n).astype(np.float64)
                    keep[1:] = attached[1:] | (backed[1:] / np.maximum(area[1:], 1) >= 0.5)
                s[(lab > 0) & ~keep[lab]] = False
                n_rej += int((~keep[1:]).sum())
            fused.append(np.maximum(ma[t], cv2.GaussianBlur(cv2.erode(s.astype(np.float32), k), (3, 3), 0)))
        log(f"fused: rejected {n_rej} far, detached SAM-only regions")
        alphas["fused"] = np.stack(fused)
        times["fused"] = times["matanyone2_seeded"] + times["sam2"]

    # --- write + score
    report = {}
    ref = alphas.get("sam2")
    prog("export", 0.1, "writing mattes + quality metrics")
    for name, al in alphas.items():
        if name not in write:
            continue
        d = os.path.join(out_root, name)
        os.makedirs(d, exist_ok=True)
        for i, fi in enumerate(idx):
            cv2.imwrite(os.path.join(d, f"alpha_{fi + 1:04d}.png"), (al[i] * 255).round().astype(np.uint8))
        m = metrics(al, frames)
        m["iou_vs_pi4"] = float(np.mean([iou(al[i], dyn[i]) for i in range(T)]))
        if ref is not None:
            ious = [iou(al[i], ref[i]) for i in range(T)]
            m["iou_vs_sam2_mean"], m["iou_vs_sam2_min"] = float(np.mean(ious)), float(np.min(ious))
        m["seconds"] = round(times.get(name, 0.0), 1)
        report[name] = m
        log(name, json.dumps({k: round(v, 3) for k, v in m.items()}))
    json.dump(report, open(os.path.join(out_root, "metrics.json"), "w"), indent=2)

    prog("export", 0.6, "writing occlusion maps")
    if occlusion and "fused" in alphas:  # inputs for the occlusion-aware composite
        occ = os.path.join(a.job, "occlusion")
        os.makedirs(occ, exist_ok=True)
        for i, fi in enumerate(idx):
            dd = cv2.resize(cache["depth"][i].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            inv = np.where(dd > 1e-6, 1.0 / np.maximum(dd, 1e-6), 0.0)
            cv2.imwrite(os.path.join(occ, f"invdepth_{fi + 1:04d}.png"),
                        np.clip(inv / INV_SCALE * 65535, 0, 65535).astype(np.uint16))
            cv2.imwrite(os.path.join(occ, f"fgmask_{fi + 1:04d}.png"),
                        (alphas["fused"][i] * 255).round().astype(np.uint8))
    if not crops:
        log("done ->", out_root)
        return report

    # --- visual crops: tint the matte over the plate around the busiest boundary region
    picks = [int(T * p) for p in (0.1, 0.35, 0.58, 0.85)]
    rows = []
    for name, al in alphas.items():
        if name not in write:
            continue
        tiles = []
        for t in picks:
            f = frames[t].astype(np.float32)
            tint = f * (1 - 0.55 * al[t][..., None]) + np.array([255, 40, 40]) * 0.55 * al[t][..., None]
            y0, x0 = int(0.45 * H), int(0.05 * W)
            crop = tint[y0:y0 + int(0.4 * H), x0:x0 + int(0.9 * W)]
            crop = cv2.resize(crop.astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            cv2.putText(crop, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            tiles.append(crop)
        rows.append(np.hstack(tiles))
    cv2.imwrite(os.path.join(out_root, "crops.jpg"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    log("done ->", out_root)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--methods", default="pi4,sam2,matanyone2,matanyone2_seeded,fused")
    ap.add_argument("--anchor", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--no_clean", action="store_true", help="keep floating components in SAM seeds")
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--occlusion", action="store_true", help="also export occlusion/invdepth + fgmask")
    a = ap.parse_args()
    run(a.job, a.methods, a.anchor, a.chunk, a.overlap, a.no_clean, occlusion=a.occlusion)


if __name__ == "__main__":
    main()
