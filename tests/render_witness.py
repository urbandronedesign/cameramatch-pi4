"""Third-person witness render: reprojected floor + gold panels + solved-camera frustum.

blender -b scene.blend --python tests/render_witness.py -- <out_dir> [res] [samples]
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
out = os.path.abspath(argv[0])
res = int(argv[1]) if len(argv) > 1 else 512
samples = int(argv[2]) if len(argv) > 2 else 32
os.makedirs(out, exist_ok=True)
sc = bpy.context.scene
shot = sc.camera
up = Vector(np.load(os.path.join(sc.cameramatch.last_job, "geometry.npz"))["up"].tolist()).normalized()
F0, F1 = sc.frame_start, sc.frame_end

# ---- follow path (smoothed): behind, above and slightly to the side of the rider
C, Fw = [], []
for f in range(F0, F1 + 1):
    sc.frame_set(f)
    M = shot.matrix_world
    fw = -M.col[2].xyz
    fw = (fw - fw.dot(up) * up).normalized()
    C.append(np.array(M.translation))
    Fw.append(np.array(fw))
C, Fw = np.array(C), np.array(Fw)


def smooth(a, k=14):
    p = np.pad(a, ((k, k), (0, 0)), mode="edge")
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    return np.stack([np.convolve(p[:, j], ker, "valid") for j in range(a.shape[1])], 1)


Cs, Fs = smooth(C), smooth(Fw)
Fs /= np.linalg.norm(Fs, axis=1, keepdims=True)
U = np.array(up)
R = np.cross(Fs, U)
wpos = Cs - 4.0 * Fs + 2.3 * U + 1.3 * R
wtgt = Cs + 7.0 * Fs - 1.3 * U

wd = bpy.data.cameras.new("witness")
wd.lens = 17
wd.clip_start, wd.clip_end = 0.05, 2000
wit = bpy.data.objects.new("witness", wd)
sc.collection.objects.link(wit)
wit.rotation_mode = "QUATERNION"
for i, f in enumerate(range(F0, F1 + 1)):
    d = Vector((wtgt[i] - wpos[i]).tolist())
    wit.location = Vector(wpos[i].tolist())
    wit.rotation_quaternion = d.to_track_quat("-Z", "Y")
    wit.keyframe_insert("location", frame=f)
    wit.keyframe_insert("rotation_quaternion", frame=f)

# ---- solved-camera frustum marker (red wire pyramid parented to the shot camera)
clip = next(c for c in bpy.data.movieclips if not c.name.endswith(("_motion_mask", "_matte")))
W, H = clip.size
L = 0.9
hw = L * (shot.data.sensor_width / 2) / shot.data.lens
hh = hw * H / W
me = bpy.data.meshes.new("frustum")
me.from_pydata([(0, 0, 0), (-hw, -hh, -L), (hw, -hh, -L), (hw, hh, -L), (-hw, hh, -L)],
               [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)], [])
fr = bpy.data.objects.new("solved_camera_frustum", me)
sc.collection.objects.link(fr)
fr.parent = shot
wire = fr.modifiers.new("skin", "SKIN")
for v in me.skin_vertices[0].data:
    v.radius = (0.025, 0.025)
red = bpy.data.materials.new("frustum_red")
if red.node_tree is None:
    red.use_nodes = True
nt = red.node_tree
nt.nodes.clear()
em = nt.nodes.new("ShaderNodeEmission")
em.inputs["Color"].default_value = (1, 0.05, 0.05, 1)
em.inputs["Strength"].default_value = 4
o = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(em.outputs[0], o.inputs[0])
me.materials.append(red)

# ---- render setup
bpy.data.objects["Pi4_floor"].visible_camera = True
if bpy.data.objects.get("Pi4_lowpoly") is not None:  # low-poly scene mesh replaces the heightfield
    bpy.data.objects["Pi4_lowpoly"].visible_camera = True
for ob in bpy.data.objects:
    if ob.name.endswith(("_points", "_tracks")):
        ob.hide_render = True
sc.camera = wit
sc.render.engine = "CYCLES"
cp = bpy.context.preferences.addons["cycles"].preferences
cp.compute_device_type = "OPTIX"
cp.get_devices()
for dv in cp.devices:
    dv.use = dv.type == "OPTIX"
sc.cycles.device = "GPU"
sc.cycles.samples = samples
sc.cycles.use_denoising = True
sc.render.film_transparent = False
sc.render.use_compositing = False
sc.view_settings.view_transform = "AgX"
sc.render.resolution_x = sc.render.resolution_y = res
sc.render.resolution_percentage = 100
sc.render.image_settings.file_format = "PNG"
sc.render.filepath = os.path.join(out, "wit_")
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(os.path.dirname(out), "witness.blend"))
bpy.ops.render.render(animation=True)
print("RENDERED witness")
