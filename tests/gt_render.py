"""Render a synthetic ground-truth plate (undistorted, with overscan) + export GT camera.

blender -b --factory-startup --python tests/gt_render.py -- <out_dir> [frames] [zoom]
zoom > 1: smooth zoom-in over the shot (focal x zoom at the end); per-frame focal in gt.json f_px_frames.
Final plate: 1920x1080, f = 1280 px (24 mm on 36 mm). Rendered with 1.2x overscan at the
same pixel focal so the distortion step (tests/gt_distort.py) has pixels to pull from.
"""
import json
import math
import os
import random
import sys

import bpy
from mathutils import Euler, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
out = os.path.abspath(argv[0])
N = int(argv[1]) if len(argv) > 1 else 120
ZOOM = float(argv[2]) if len(argv) > 2 else 1.0
W, H, F_PX, OVER = 1920, 1080, 1280.0, 1.2
os.makedirs(os.path.join(out, "undist"), exist_ok=True)
random.seed(1)

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
scene = bpy.context.scene


def textured_material(name, scale, hue):
    m = bpy.data.materials.new(name)
    if m.node_tree is None:
        m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out_n = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfDiffuse")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 8
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = scale * 0.7
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["Factor"].default_value = 0.5
    tc = nt.nodes.new("ShaderNodeTexCoord")  # object space -> texture scale in metres
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.05 * hue, 0.04, 0.03, 1)
    ramp.color_ramp.elements[1].color = (0.9, 0.85 * hue, 0.7, 1)
    nt.links.new(noise.outputs["Fac"], mix.inputs[6])
    nt.links.new(vor.outputs["Distance"], mix.inputs[7])
    nt.links.new(mix.outputs[2], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Color"])
    nt.links.new(bsdf.outputs["BSDF"], out_n.inputs["Surface"])
    return m


bpy.ops.mesh.primitive_plane_add(size=200)
bpy.context.object.data.materials.append(textured_material("ground", 3.0, 1.0))
for i in range(60):
    sx, sy, sz = random.uniform(0.5, 3), random.uniform(0.5, 3), random.uniform(0.5, 6)
    x, y = random.uniform(-25, 25), random.uniform(4, 60)
    if abs(x) < 2.5 and y < 25:
        x += 6 * (1 if x >= 0 else -1)
    bpy.ops.mesh.primitive_cube_add(location=(x, y, sz / 2))
    o = bpy.context.object
    o.scale = (sx / 2, sy / 2, sz / 2)
    o.rotation_euler.z = random.uniform(0, math.pi)
    o.data.materials.append(textured_material(f"box{i}", random.uniform(2, 8), random.uniform(0.4, 1.2)))

bpy.ops.object.light_add(type="SUN", rotation=(math.radians(50), 0, math.radians(30)))
bpy.context.object.data.energy = 4
if scene.world is None:
    scene.world = bpy.data.worlds.new("w")
scene.world.color = (0.5, 0.6, 0.8)

cam_data = bpy.data.cameras.new("gt")
cam = bpy.data.objects.new("gt", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
Wr, Hr = int(round(W * OVER / 2) * 2), int(round(H * OVER / 2) * 2)
cam_data.sensor_fit = "HORIZONTAL"
cam_data.sensor_width = 36.0
cam_data.lens = F_PX / Wr * 36.0  # same pixel focal at overscan resolution
scene.render.resolution_x, scene.render.resolution_y = Wr, Hr
scene.render.resolution_percentage = 100
scene.render.engine = "BLENDER_EEVEE"
try:
    scene.eevee.taa_render_samples = 16
except AttributeError:
    pass
scene.render.image_settings.file_format = "PNG"
scene.frame_start, scene.frame_end = 1, N

gt = []
for i in range(N):
    t = i / max(N - 1, 1)
    pos = Vector((1.5 * math.sin(2 * math.pi * t), 0.12 * i, 1.6 + 0.08 * math.sin(9 * t)))
    rot = Euler((math.radians(85 + 3 * math.sin(5 * t)), math.radians(1.5 * math.sin(7 * t)),
                 math.radians(-20 * math.sin(2 * math.pi * t * 0.8))), "XYZ")
    cam.location, cam.rotation_euler = pos, rot
    cam.keyframe_insert("location", frame=i + 1)
    cam.keyframe_insert("rotation_euler", frame=i + 1)
    zf = 1 + (ZOOM - 1) * t * t * (3 - 2 * t)
    cam_data.lens = F_PX * zf / Wr * 36.0
    cam_data.keyframe_insert("lens", frame=i + 1)
f_frames = []
for i in range(N):
    scene.frame_set(i + 1)
    gt.append([list(r) for r in cam.matrix_world])
    f_frames.append(cam_data.lens / 36.0 * Wr)
    scene.render.filepath = os.path.join(out, "undist", f"{i:06d}.png")
    bpy.ops.render.render(write_still=True)

with open(os.path.join(out, "gt.json"), "w") as fh:
    json.dump(dict(c2w_blender=gt, f_px=F_PX, f_px_frames=f_frames, zoom=ZOOM, plate=[W, H], render=[Wr, Hr]), fh)
print("GT DONE", out)
