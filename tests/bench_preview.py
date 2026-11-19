"""Time preview-render settings on a built scene (headless).

blender -b <scene.blend> --python tests/bench_preview.py -- <out_dir> [frames]
Renders `frames` frames at 50 % / 24 samples with: motion blur on/off x persistent data on/off.
"""
import importlib
import os
import sys
import time

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
out = os.path.abspath(argv[0])
nfr = int(argv[1]) if len(argv) > 1 else 8
cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
sc = bpy.context.scene
print("[bench] hidden:", cm.hide_camera_blockers(sc, print))
f0 = sc.frame_start + 20
for mb in (True, False):
    for pers in (False, True):
        d = os.path.join(out, f"mb{int(mb)}_p{int(pers)}")
        cm.set_builder.setup_render(sc, d, 24, mb, log=lambda *a: None, percentage=50)
        sc.render.use_persistent_data = pers
        sc.frame_start, sc.frame_end = f0, f0 + nfr - 1
        bpy.ops.render.render(animation=True)          # warm-up excluded: first frame timed separately below
        t = time.time()
        sc.frame_start, sc.frame_end = f0 + nfr, f0 + 2 * nfr - 1
        bpy.ops.render.render(animation=True)
        print(f"[bench] motion_blur={mb} persistent={pers}: {(time.time() - t) / nfr:.2f} s/frame", flush=True)
