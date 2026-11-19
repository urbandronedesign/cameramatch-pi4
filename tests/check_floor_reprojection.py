"""Geometry check: project frame (f - LAG) onto the floor, view from frame f, compare to plate f.

blender -b scene.blend --python tests/check_floor_reprojection.py -- <out_dir> [lag]
"""
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
out = os.path.abspath(argv[0])
lag = int(argv[1]) if len(argv) > 1 else 8
os.makedirs(out, exist_ok=True)
sc = bpy.context.scene
cam = sc.camera
floor = bpy.data.objects["Pi4_floor"]
floor.visible_camera = True
for o in bpy.data.objects:  # isolate the floor: nothing else may cover pixels
    if o is not floor and o.type not in {"CAMERA", "EMPTY"}:
        o.hide_render = True
if floor.data.materials and "Texture Coordinate" not in floor.data.materials[0].node_tree.nodes:
    floor.data.materials[0] = bpy.data.materials["Pi4_projector"]
sc.world = bpy.data.worlds.new("black_check")
sc.world.color = (0, 0, 0)

proj = bpy.data.objects.new("lag_projector", None)
sc.collection.objects.link(proj)
for f in range(sc.frame_start, sc.frame_end + 1):
    sc.frame_set(max(sc.frame_start, f - lag))
    M = cam.matrix_world.copy()
    proj.matrix_world = M
    proj.keyframe_insert("location", frame=f)
    proj.rotation_mode = "QUATERNION"
    proj.keyframe_insert("rotation_quaternion", frame=f)
nt = floor.data.materials[0].node_tree
nt.nodes["Texture Coordinate"].object = proj
nt.nodes["Image Texture"].image_user.frame_offset = -lag

sc.render.engine = "BLENDER_EEVEE"
sc.render.film_transparent = True            # alpha marks exactly where the floor rendered
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGBA"
sc.render.use_compositing = False
sc.view_settings.view_transform = "Standard"
for f in (40, 100, 150, 200, 260):
    sc.frame_set(f)
    sc.render.filepath = os.path.join(out, f"lag_{f:04d}.png")
    bpy.ops.render.render(write_still=True)
print("DONE")
