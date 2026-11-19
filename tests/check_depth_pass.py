"""Is Cycles' Depth pass planar or radial? What does it hold at anti-aliased edges?
blender -b --factory-startup --python tests/check_depth_pass.py -- <out_dir>"""
import os
import sys

import bpy
import numpy as np

out = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
os.makedirs(out, exist_ok=True)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = 16
sc.cycles.use_denoising = False
sc.render.resolution_x, sc.render.resolution_y = 200, 100
sc.render.film_transparent = True
cam = bpy.data.objects.new("c", bpy.data.cameras.new("c"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.lens, cam.data.sensor_width = 18, 36  # wide: corner angle ~ 51 deg -> radial/planar ~1.6
bpy.ops.mesh.primitive_plane_add(size=6, location=(-0.0, 0, -10))  # covers part of the frame only
sc.view_layers[0].use_pass_z = True
ng = bpy.data.node_groups.new("c", "CompositorNodeTree")
sc.compositing_node_group = ng
ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
rl = ng.nodes.new("CompositorNodeRLayers")
o = ng.nodes.new("NodeGroupOutput")
ng.links.new(rl.outputs["Depth"], o.inputs[0])
sc.render.use_compositing = True
sc.render.image_settings.file_format = "OPEN_EXR"
sc.render.image_settings.color_depth = "32"
sc.render.filepath = os.path.join(out, "depth.exr")
bpy.ops.render.render(write_still=True)
img = bpy.data.images.load(os.path.join(out, "depth.exr"))
px = np.array(img.pixels[:]).reshape(100, 200, 4)[::-1, :, 0]  # top row first
print("DEPTH centre", px[50, 100], "near plane edge row50:", np.round(px[50, 60:75], 3))
x = np.arange(200) + 0.5 - 100
fpx = 18 / 36 * 200
print("planar expected 10 everywhere on plane; radial at col 70 would be",
      round(10 * np.sqrt(1 + (x[70] / fpx) ** 2), 3), "measured", px[50, 70])
print("background value", px[2, 2])
