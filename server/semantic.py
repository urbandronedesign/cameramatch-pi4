"""Semantic masks of non-rigid / non-static scene regions (trees, water, sky...) for the camera tracker.

OneFormer (ADE20K, Swin-Tiny; MIT license, shi-labs/oneformer_ade20k_swin_tiny via Hugging Face transformers).
Run on keyframes, each frame takes the nearest keyframe's mask (these regions move slowly in the image),
dilated a little so tracks on the boundary (leaf edges against the sky) are excluded too.
"""
import time

import cv2
import numpy as np
import torch

MODEL = "shi-labs/oneformer_ade20k_swin_tiny"
# ADE20K names (first synonym) treated as non-rigid / not static
DEFAULT_CLASSES = ("tree", "plant", "palm", "flower", "water", "sea", "river", "waterfall", "fountain", "sky", "flag")
_cache = {}


def _model(device):
    if "m" not in _cache:
        from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
        proc = OneFormerProcessor.from_pretrained(MODEL)
        m = OneFormerForUniversalSegmentation.from_pretrained(MODEL).to(device).eval()
        if device == "cuda":
            m = m.half()
        _cache["m"] = (proc, m)
    return _cache["m"]


def masks(src, frame_indices, model_size, classes=DEFAULT_CLASSES, step=5, short_side=512, dilate_px=2,
          log=print, progress=None):
    """src: video / image folder; frame_indices: the frames the solve uses (in order).
    Returns (mask uint8 [T, Hm, Wm] 0/1 at Pi4 model resolution, {class: image fraction})."""
    from refine import iter_frames
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    proc, m = _model(dev)
    lab = m.config.id2label
    want = {int(k) for k, v in lab.items() if v.split(",")[0].split(";")[0].strip() in classes}
    Wm, Hm = model_size
    fidx = [int(i) for i in frame_indices]
    T = len(fidx)
    keys = list(range(0, T, step))
    if keys[-1] != T - 1:
        keys.append(T - 1)
    want_src = {fidx[k]: k for k in keys}
    kmask, frac = {}, {}
    n = 0
    for src_i, fr in iter_frames(src, [fidx[k] for k in keys]):
        k = want_src[int(src_i)]
        n += 1
        h, w = fr.shape[:2]
        sc = short_side / min(h, w)
        img = cv2.cvtColor(cv2.resize(fr, (int(round(w * sc)), int(round(h * sc))), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        inp = proc(images=img, task_inputs=["semantic"], return_tensors="pt")
        inp = {a: (b.to(dev).half() if (dev == "cuda" and b.dtype == torch.float32) else b.to(dev))
               for a, b in inp.items()}
        with torch.no_grad():
            out = m(**inp)
        seg = proc.post_process_semantic_segmentation(out, target_sizes=[img.shape[:2]])[0].cpu().numpy()
        mk = np.isin(seg, list(want)).astype(np.uint8)
        for c in np.unique(seg[mk > 0]):
            name = lab[int(c)].split(",")[0]
            frac[name] = frac.get(name, 0.0) + float((seg == c).mean()) / len(keys)
        kmask[k] = cv2.resize(mk, (Wm, Hm), interpolation=cv2.INTER_NEAREST)
        if progress:
            progress(n / len(keys), f"semantic masks {n}/{len(keys)}")
    if dilate_px > 0:
        ker = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
        kmask = {k: cv2.dilate(v, ker) for k, v in kmask.items()}
    karr = np.array(keys)
    out = np.stack([kmask[int(karr[np.argmin(np.abs(karr - i))])] for i in range(T)])
    log(f"[semantic] {len(keys)} keyframes in {time.time() - t0:.1f}s, masked {100 * out.mean():.1f}% of the image: "
        + ", ".join(f"{k} {100 * v:.0f}%" for k, v in sorted(frac.items(), key=lambda x: -x[1]) if v > 0.005))
    return out, frac
