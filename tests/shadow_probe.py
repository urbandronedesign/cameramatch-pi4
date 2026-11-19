"""Where can the spheres' shadows land? Render shadow-catcher alpha for several sun directions at one frame.

blender -b spheres.blend --python tests/shadow_probe.py -- <out_dir> <frame>
"""
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
out, frame = os.path.abspath(argv[0]), int(argv[1])
os.makedirs(out, exist_ok=True)
sc = bpy.context.scene
sc.frame_set(frame)
cam = sc.camera
sc.render.engine = "CYCLES"
sc.cycles.samples = 32
sc.cycles.use_denoising = True
sc.render.film_transparent = True
sc.render.use_compositing = False
sc.render.resolution_percentage = 50
sc.render.image_settings.color_mode = "RGBA"
cp = bpy.context.preferences.addons["cycles"].preferences
cp.compute_device_type = "OPTIX"
cp.get_devices()
for d in cp.devices:
    d.use = d.type == "OPTIX"
sc.cycles.device = "GPU"
for o in bpy.data.objects:
    if o.name.startswith("Chrome_sphere"):
        o.visible_camera = False  # show only the shadows they cast
sun = bpy.data.objects["Sun"]
fwd = -cam.matrix_world.col[2].xyz
fwd.z = 0
fwd.normalize()
right = fwd.cross(Vector((0, 0, 1)))
dirs = {"front_high": fwd * 0.4 + Vector((0, 0, -1)), "front_left_low": fwd * 0.8 + right * 0.6 + Vector((0, 0, -0.5)),
        "side_right": -right + Vector((0, 0, -0.6)), "back_low": -fwd * 0.9 + Vector((0, 0, -0.45)),
        "top": Vector((0, 0, -1))}
for name, d in dirs.items():
    sun.rotation_euler = d.normalized().to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = os.path.join(out, f"shadow_{name}.png")
    bpy.ops.render.render(write_still=True)
print("PROBED", list(dirs))
