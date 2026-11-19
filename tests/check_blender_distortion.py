"""Check Blender's Polynomial distortion matches refine.py's model, using the GT plate.

blender -b --factory-startup --python tests/check_blender_distortion.py -- <gt_dir> <frame>
Undistorts the distorted plate frame with Blender's compositor (Movie Distortion, UNDISTORT)
using the GT k1/k2/focal, and writes it next to the centre crop of the clean overscan render.
"""
import json
import os
import sys

import bpy

gt_dir, frame = sys.argv[sys.argv.index("--") + 1:][:2]
gt = json.load(open(os.path.join(gt_dir, "gt.json")))
W, H = gt["plate"]
scene = bpy.context.scene

clip = bpy.data.movieclips.load(os.path.join(gt_dir, "plate", f"{int(frame):06d}.png"))
tc = clip.tracking.camera
tc.sensor_width = 36.0
tc.units = "PIXELS"
tc.focal_length_pixels = gt["f_px"]
tc.distortion_model = "POLYNOMIAL"
tc.k1, tc.k2, tc.k3 = gt["k1"], gt["k2"], 0.0
tc.principal_point = (0.0, 0.0)
print("CAM", tc.focal_length, tc.focal_length_pixels, list(tc.principal_point), list(tc.principal_point_pixels))

img = bpy.data.images.load(os.path.join(gt_dir, "plate", f"{int(frame):06d}.png"))
ng = bpy.data.node_groups.new("comp", "CompositorNodeTree")
scene.compositing_node_group = ng
ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
n_img = ng.nodes.new("CompositorNodeImage")
n_img.image = img
n_dist = ng.nodes.new("CompositorNodeMovieDistortion")
n_dist.clip = clip
try:
    n_dist.distortion_type = "UNDISTORT"
except AttributeError:
    n_dist.inputs["Type"].default_value = "Undistort"
n_out = ng.nodes.new("NodeGroupOutput")
ng.links.new(n_img.outputs["Image"], n_dist.inputs["Image"])
ng.links.new(n_dist.outputs["Image"], n_out.inputs[0])
scene.render.resolution_x, scene.render.resolution_y = W, H
scene.render.resolution_percentage = 100
scene.render.use_compositing = True
scene.render.filepath = os.path.join(gt_dir, f"blender_undist_{int(frame):06d}.png")
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.render.render(write_still=True)
print("WROTE", scene.render.filepath)

