# SPDX-License-Identifier: GPL-3.0-or-later
"""Plain-language layer: footage presets, the solve quality grade with one-click fixes, friendly errors."""
import json
import os

# ----------------------------------------------------------------------------------------------- presets
# Each preset sets the solve options together. AUTO: zoom detected from Pi4's focal, rolling shutter on for
# portrait (phone) footage, everything else at the defaults.
PRESETS = {
    "AUTO": ("Auto (recommended)", "Detects zoom and phone (portrait) footage by itself",
             dict(zoom_mode="AUTO", rs_mode="AUTO", solve_distortion=True, lock_focal=False, ignore_nonrigid=True,
                  max_tracks=600)),
    "PHONE": ("Phone, handheld", "Phone or action cam carried by hand: rolling shutter on",
              dict(zoom_mode="OFF", rs_mode="ON", solve_distortion=True, lock_focal=False, ignore_nonrigid=True,
                   max_tracks=600)),
    "DRONE": ("Drone", "Drone flight: rolling shutter on, trees and water ignored, more tracks",
              dict(zoom_mode="OFF", rs_mode="ON", solve_distortion=True, lock_focal=False, ignore_nonrigid=True,
                   max_tracks=900)),
    "TRIPOD": ("Tripod / slow pan", "Camera mostly turns in place: focal locked to the model, no lens distortion "
               "(they cannot be measured without camera travel)",
               dict(zoom_mode="OFF", rs_mode="OFF", solve_distortion=False, lock_focal=True, ignore_nonrigid=True,
                    max_tracks=600)),
    "ZOOM": ("Zoom lens", "The focal length changes during the shot: one focal per frame",
             dict(zoom_mode="ON", rs_mode="OFF", solve_distortion=False, lock_focal=False, ignore_nonrigid=True,
                  max_tracks=600)),
    "CINEMA": ("Cinema camera", "Rectilinear lens, global or fast shutter, smooth move",
               dict(zoom_mode="OFF", rs_mode="OFF", solve_distortion=False, lock_focal=False, ignore_nonrigid=True,
                    max_tracks=600)),
    "NIGHT": ("Night / low light", "Noisy, dark footage: more tracks, robust settings",
              dict(zoom_mode="AUTO", rs_mode="AUTO", solve_distortion=False, lock_focal=False, ignore_nonrigid=True,
                   max_tracks=1000)),
}
PRESET_ITEMS = [(k, v[0], v[1]) for k, v in PRESETS.items()]


def apply_preset(s, key):
    for k, v in PRESETS[key][2].items():
        setattr(s, k, v)


# ----------------------------------------------------------------------------------------------- grade
REGION = [["top left", "top", "top right"], ["left", "centre", "right"],
          ["bottom left", "bottom", "bottom right"]]


def load_stats(job_dir):
    """Refinement stats of a job: refine_stats.json (written by every solve / re-solve), else result.json."""
    for name, key in (("refine_stats.json", None), ("result.json", "refine")):
        p = os.path.join(job_dir, name)
        if os.path.isfile(p):
            try:
                with open(p) as fh:
                    d = json.load(fh)
                d = d.get(key) if key else d
                if d:
                    return d
            except (OSError, ValueError):
                pass
    return None


def load_check(job_dir):
    p = os.path.join(job_dir, "check", "metrics.json")
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def grade(stats, settings, check=None, clip_size=None):
    """-> (level 'GOOD'|'FAIR'|'BAD'|'NONE', headline, [(text, fix_id or None), ...])"""
    if not stats:
        return "NONE", "Not solved yet", []
    med, p90 = stats["final"]["median"], stats["final"]["p90"]
    level = "GOOD" if (med < 0.5 and p90 < 1.5) else ("FAIR" if (med < 1.0 and p90 < 3.0) else "BAD")
    words = {"GOOD": "Good: CG will stick to this shot",
             "FAIR": "Fair: usable, small sliding is possible",
             "BAD": "Poor: CG will slide, try the suggestions"}
    tips = []
    dg = stats.get("diag") or {}
    tpf = dg.get("tracks_per_frame")
    if tpf is not None and tpf < 40:
        level = "BAD" if tpf < 15 else ("FAIR" if level == "GOOD" else level)
        tips.append((f"Only {tpf:.0f} features tracked per frame: the scene has little texture or is very "
                     "dark/blurred. Try the Night preset or a higher Model Resolution.", None))
    if check and check.get("stick_med_px") is not None and check.get("stick_samples", 0) > 50:
        st = check["stick_med_px"]
        if st > 3.0:
            level = "BAD"
            tips.append((f"The ground grid slides {st:.1f} px on the ground in the overlay check.", None))
        elif st > 1.5 and level == "GOOD":
            level = "FAIR"
    # where it struggles: the grid cell with the most rejected (moving) features
    rej, n = dg.get("grid_reject"), dg.get("grid_n")
    if rej and n:
        cells = [(rej[c], c) for c in range(9) if n[c] > 0.03 * sum(n)]
        if cells:
            r, c = max(cells)
            if r > 0.2:
                where = REGION[c // 3][c % 3]
                if not settings.ignore_nonrigid:
                    tips.append((f"Many features in the {where} of the frame move (trees, water, sky?).",
                                 "ignore_nonrigid"))
                else:
                    tips.append((f"Something in the {where} of the frame moves (people, a screen, a reflection, "
                                 "a vehicle). Paint an Ignore Mask over it.", "mask"))
    W, H = clip_size or stats.get("plate", [1, 0])
    rs_on = abs(float(stats.get("rolling_shutter", 0.0) or 0.0)) > 0 or settings.rs_mode == "ON"
    if H > W and not rs_on:
        tips.append(("Portrait video is usually a phone: turn on Rolling Shutter.", "rs"))
    za = stats.get("zoom_auto")
    if za and za.get("zoom"):
        tips.append((f"Zoom detected (focal changes {za['focal_range'] * 100:.0f}%): the camera's focal is "
                     "animated.", None))
    f, f0 = stats.get("f_px"), stats.get("f_px_pi4")
    if level != "GOOD" and f and f0 and not settings.lock_focal and (f / f0 > 1.35 or f / f0 < 0.74):
        tips.append((f"The solved focal is {abs(f / f0 - 1) * 100:.0f}% away from the model's estimate. If the "
                     "camera mostly turns without travelling, lock the focal.", "lock_focal"))
    if level == "BAD" and settings.solve_distortion:
        tips.append(("Cinema or rectilinear lens? Turn off Lens Distortion.", "no_distortion"))
    bad = dg.get("bad_frames") or []
    if bad and len(bad) >= 3:
        tips.append((f"Frames {bad[0] + 1}-{bad[-1] + 1} track worse than the rest (fast motion or motion blur).",
                     None))
    return level, f"{words[level]} ({med:.2f} px)", tips


# ----------------------------------------------------------------------------------------------- errors
def friendly(err):
    """-> (plain message, needs_engine)"""
    e = str(err or "")
    low = e.lower()
    if any(k in low for k in ("connection refused", "10061", "urlopen error", "remote end closed",
                              "connection reset", "timed out", "matte-server running")):
        return "The engine is not running. Click Start Engine (Pi4 tab, top).", True
    if "cancel" in low:
        return "Cancelled.", False
    if "out of memory" in low:
        return ("The GPU ran out of memory. Raise Frame Stride, lower Max Frames or Model Resolution "
                "(Advanced: Solve), or turn off Detect Moving Objects."), False
    if "model loading" in low or "model starting" in low:
        return "The engine is still loading the Pi4 model (first start downloads it). Try again in a minute.", False
    if "unknown job" in low or "pi4_cache" in low:
        return "This solve's files are missing from the work folder. Solve the clip again.", False
    if "no ground plane" in low:
        return "No floor found below the camera, so the overlay check cannot draw its grid.", False
    if "cannot read plate" in low or "no plate" in low:
        return "The engine cannot read the footage copy. Re-solve the clip (Solve Camera).", False
    if "no r_" in low and "exr" in low:
        return "No rendered frames were found. Render again.", False
    first = e.strip().splitlines()[0] if e.strip() else "unknown error"
    return first[:200] + " (details: Show Log)", False
