"""blender -b scene.blend --python tests/render_frames.py -- <out_dir> <frames 'a,b,c' or 'all'> [samples]"""
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
out, frames = os.path.abspath(argv[0]), argv[1]
sc = bpy.context.scene
if len(argv) > 2:
    sc.cycles.samples = int(argv[2])
os.makedirs(out, exist_ok=True)
cp = bpy.context.preferences.addons["cycles"].preferences
cp.compute_device_type = "OPTIX"
cp.get_devices()
for d in cp.devices:
    d.use = d.type == "OPTIX"
fl = list(range(sc.frame_start, sc.frame_end + 1)) if frames == "all" else [int(x) for x in frames.split(",")]
# warm-up: first movie-texture access in a background session can come back black
sc.frame_set(fl[0])
s0 = sc.cycles.samples
sc.cycles.samples = 1
sc.render.filepath = os.path.join(out, "_warmup")
bpy.ops.render.render(write_still=True)
sc.cycles.samples = s0
for fr in fl:
    p = os.path.join(out, f"r_{fr:04d}.exr")
    if os.path.exists(p) and frames == "all":
        continue
    sc.frame_set(fr)
    sc.render.filepath = p
    bpy.ops.render.render(write_still=True)
    print("RENDERED", fr, flush=True)
