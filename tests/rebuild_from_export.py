"""Rebuild a CameraMatch shot from its EXPORT FOLDER ONLY, the way another application would (no add-on code).

blender --factory-startup -b --python tests/rebuild_from_export.py -- <export_dir> <render_dir> <samples> <abc|usd|fbx> <frames...>
Reads camera.json (resolution, fps, range, rolling shutter), imports the Alembic (camera, set, CG), turns the set
meshes into shadow catchers hidden from the camera, lights with lighting/plate_env.png, renders multilayer EXRs
(Combined + Shadow Catcher) of the given frames with a pinhole camera (lens distortion goes on in the comp, through
stmap_redistort.exr).
"""
import glob
import json
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
exp, rdir, samples = os.path.abspath(argv[0]), os.path.abspath(argv[1]), int(argv[2])
fmt = argv[3]
frames = [int(f) for f in argv[4:]]
J = json.load(open(os.path.join(exp, "camera.json")))
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
W, H = J["resolution"]
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
fps = J["fps"]
sc.render.fps, sc.render.fps_base = round(fps), round(fps) / fps
sc.frame_start, sc.frame_end = J["frame_range"]
src = glob.glob(os.path.join(exp, "*." + {"abc": "abc", "usd": "usdc", "fbx": "fbx"}[fmt]))[0]
if fmt == "abc":
    bpy.ops.wm.alembic_import(filepath=src, as_background_job=False)
elif fmt == "usd":
    bpy.ops.wm.usd_import(filepath=src)
else:
    bpy.ops.import_scene.fbx(filepath=src, anim_offset=0.0)
cam = next(o for o in sc.objects if o.type == "CAMERA")
sc.camera = cam
cam.data.sensor_fit = "HORIZONTAL"
cam.data.sensor_width = J["sensor_width_mm"]
print("REBUILD camera", cam.name, "objects", len(sc.objects), flush=True)
grey = bpy.data.materials.new("set_grey")
for o in sc.objects:
    if o.type == "MESH" and o.name.startswith(("CM_floor_catcher", "CM_walls", "CM_street_walls")):
        o.is_shadow_catcher = True
        if not o.name.startswith("CM_floor_catcher"):
            o.visible_camera = False
        o.data.materials.clear()
        o.data.materials.append(grey)
    elif o.type == "MESH" and o.name.endswith("_tracks"):
        o.hide_render = True
    elif o.type == "EMPTY":
        o.hide_render = True
w = bpy.data.worlds.new("env")
sc.world = w
if w.node_tree is None:
    w.use_nodes = True
nt = w.node_tree
nt.nodes.clear()
env = nt.nodes.new("ShaderNodeTexEnvironment")
env.image = bpy.data.images.load(os.path.join(exp, "lighting", "plate_env.png"))
bg = nt.nodes.new("ShaderNodeBackground")
out = nt.nodes.new("ShaderNodeOutputWorld")
nt.links.new(env.outputs["Color"], bg.inputs["Color"])
nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
sc.render.engine = "CYCLES"
try:
    cp = bpy.context.preferences.addons["cycles"].preferences
    cp.compute_device_type = "OPTIX"
    cp.get_devices()
    for d in cp.devices:
        d.use = d.type == "OPTIX"
    sc.cycles.device = "GPU"
except Exception as e:
    print("REBUILD gpu", e)
sc.cycles.samples = samples
sc.cycles.use_denoising = True
sc.render.film_transparent = True
sc.render.use_motion_blur = True
sc.render.motion_blur_shutter = 0.5
rs = J["rolling_shutter"]
if rs["readout_frames"] > 0.05 and rs["axis"] == "rows":          # as an artist would from camera.json
    win = rs["readout_frames"] + 0.5
    sc.render.motion_blur_shutter = win
    sc.cycles.rolling_shutter_type = "TOP"
    sc.cycles.rolling_shutter_duration = 0.5 / win
sc.view_settings.view_transform = "Standard"
sc.view_layers[0].cycles.use_pass_shadow_catcher = True
im = sc.render.image_settings
if hasattr(im, "media_type"):
    im.media_type = "MULTI_LAYER_IMAGE"
im.file_format = "OPEN_EXR_MULTILAYER"
im.color_depth = "16"
os.makedirs(rdir, exist_ok=True)
for f in frames:
    sc.frame_set(f)
    sc.render.filepath = os.path.join(rdir, f"r_{f:04d}")
    bpy.ops.render.render(write_still=True)
print("REBUILD done", len(frames), flush=True)
