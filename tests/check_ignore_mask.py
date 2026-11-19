"""Rasterise a programmatic Mask and check where it lands (mask-space coordinate convention).

blender -b --factory-startup --python tests/check_ignore_mask.py -- <video> <out_job_dir>
Mask: a rectangle with corners (0,1)-(1,0.5) in mask space. Expected: the top half of the frame is white.
(Animated / parented masks are evaluated by Blender's compositor Mask node, the same code as the editor.)
"""
import importlib
import json
import os
import sys

import addon_utils
import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
video, job = argv[0], argv[1]
addon_utils.enable("bl_ext.user_default.cameramatch_pi4", default_set=True)
im = importlib.import_module("bl_ext.user_default.cameramatch_pi4.ignore_mask")
sc = bpy.context.scene
clip = bpy.data.movieclips.load(video)
sc.frame_start, sc.frame_end = clip.frame_start, clip.frame_start + clip.frame_duration - 1
mask = bpy.data.masks.new("ignore_test")
layer = mask.layers.new()
sp = layer.splines.new()
sp.use_cyclic = True
sp.points.add(3)
for p, c in zip(sp.points, [(0.0, 1.0), (1.0, 1.0), (1.0, 0.5), (0.0, 0.5)]):
    p.co = c
    p.handle_type = "VECTOR"
os.makedirs(job, exist_ok=True)
im.rasterize(mask, clip, sc, job)
meta = json.load(open(os.path.join(job, "ignore_mask.json")))
n = len([f for f in os.listdir(os.path.join(job, "ignore_mask")) if f.endswith(".png")])
img = bpy.data.images.load(os.path.join(job, meta["pattern"].format(meta["first_scene_frame"])))
w, h = img.size
a = np.array(img.pixels[:], np.float32).reshape(h, w, -1)[::-1, :, 0]     # row 0 = top
top_frac = (a > 0.5).mean(1)
print(f"[mask] {n} frames, {w}x{h}; white fraction top quarter {top_frac[: h // 4].mean():.2f}, "
      f"rows 40-48% {top_frac[int(.40 * h):int(.48 * h)].mean():.2f}, rows 52-60% {top_frac[int(.52 * h):int(.60 * h)].mean():.2f}, "
      f"bottom quarter {top_frac[3 * h // 4:].mean():.2f}; boundary row {np.argmax(top_frac < 0.5) / h:.3f} of the height",
      flush=True)
