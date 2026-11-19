"""Walkthrough video + GIF from the guided GUI test's screenshots (tests/gui_guided.py), with captions.

python /tools/make_walkthrough.py <shots_dir> <out.mp4> [--gif out.gif] [--comp comp.mp4] [--seconds 6]
Each screenshot is held `seconds` with a caption bar (the step it shows); the optional comp clip plays at
the end. The GIF is a 960 px, lower frame-rate version for the README.
"""
import argparse
import glob
import os
import subprocess

import cv2
import numpy as np

CAPTIONS = {
    "empty_scene": ("Start", "Sidebar (N) > Pi4. The engine light is green: Docker and the engine run."),
    "footage_loaded": ("1. Load Footage", "A video or an image sequence. Pick the footage type (Auto detects it)."),
    "solving": ("2. Solve Camera", "Progress, time left, and Cancel. About a minute for 120 frames."),
    "solved_grade": ("3. Check the Solve", "A grade in plain words, suggestions with one-click fixes."),
    "overlay_check": ("3. Check Overlay", "A grid drawn on the ground in every frame: it must stay glued to it."),
    "set_built": ("4. Build Set + Lighting", "Floor shadow catcher, walls, and the footage as the light."),
    "characters": ("5. Add Characters", "Import a character, Walk to Cursor or along a path: feet stay planted."),
    "preview_done": ("6. Render", "Preview (half size, about 1 s a frame), then Final Render. Composited for you."),
}


def caption(img, title, text, W):
    h, w = img.shape[:2]
    img = cv2.resize(img, (W, int(h * W / w)), interpolation=cv2.INTER_AREA)
    bar = np.full((int(W * 0.06), W, 3), 24, np.uint8)
    s = W / 1920
    cv2.putText(bar, title, (int(24 * s), int(bar.shape[0] * 0.68)), cv2.FONT_HERSHEY_DUPLEX, 1.25 * s,
                (90, 200, 255), max(1, int(2 * s)), cv2.LINE_AA)
    cv2.putText(bar, text, (int(520 * s), int(bar.shape[0] * 0.66)), cv2.FONT_HERSHEY_SIMPLEX, 0.95 * s,
                (235, 235, 235), max(1, int(2 * s)), cv2.LINE_AA)
    out = np.vstack([bar, img])
    return out[: out.shape[0] // 2 * 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shots")
    ap.add_argument("out")
    ap.add_argument("--gif", default="")
    ap.add_argument("--comp", default="")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--width", type=int, default=1920)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.shots, "*.png")))
    tmp = os.path.splitext(a.out)[0] + "_frames"
    os.makedirs(tmp, exist_ok=True)
    fps, n = 25, 0
    size = None
    for f in files:
        key = os.path.basename(f)[3:-4]
        title, text = CAPTIONS.get(key, (key, ""))
        fr = caption(cv2.imread(f), title, text, a.width)
        size = size or fr.shape[:2]
        fr = cv2.resize(fr, (size[1], size[0]))
        for _ in range(int(a.seconds * fps)):
            cv2.imwrite(os.path.join(tmp, f"f_{n:05d}.png"), fr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            n += 1
    if a.comp:
        cap = cv2.VideoCapture(a.comp)
        while True:
            ok, img = cap.read()
            if not ok:
                break
            fr = caption(img, "Result", "The composited shot (preview quality).", size[1])
            fr = cv2.resize(fr, (size[1], size[0]))
            cv2.imwrite(os.path.join(tmp, f"f_{n:05d}.png"), fr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            n += 1
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", os.path.join(tmp, "f_%05d.png"),
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", a.out], check=True)
    if a.gif:
        pal = os.path.join(tmp, "pal.png")
        vf = "fps=5,scale=960:-1:flags=lanczos"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.out, "-vf", vf + ",palettegen=max_colors=128",
                        pal], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.out, "-i", pal, "-lavfi",
                        vf + "[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4", a.gif], check=True)
    for f in glob.glob(os.path.join(tmp, "*")):
        os.remove(f)
    os.rmdir(tmp)
    print(f"wrote {a.out} ({n} frames, {n / fps:.0f} s)" + (f" and {a.gif}" if a.gif else ""))


if __name__ == "__main__":
    main()
