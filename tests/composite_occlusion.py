"""Occlusion-aware composite of the CG check cubes over the plate, all in Blender's compositor.

blender -b <with_geometry.blend> --python tests/composite_occlusion.py -- <occlusion_dir> <out_dir> [samples]
- Cycles (GPU) render of the cubes + shadow-catcher planes (film transparent, Depth pass).
- CG RGBA and alpha-premultiplied inverse depth are distorted with the solved lens (Movie Distortion).
- Visibility = soft depth test vs Pi4 inverse depth  x  (1 - Pi4 foreground matte).
- Result alpha-overed on the movie clip.
"""
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
occ, out_dir = os.path.abspath(argv[0]), os.path.abspath(argv[1])
samples = int(argv[2]) if len(argv) > 2 else 64
INV_SCALE, SOFT = 4.0, 0.08            # soft ramp 8%
TOL = float(argv[3]) if len(argv) > 3 else 0.12  # depth tolerance (raise for CG lying on real surfaces)
sc = bpy.context.scene
clip = next(c for c in bpy.data.movieclips if not c.name.endswith("_motion_mask"))
W, H = clip.size

# ---- render setup: Cycles GPU
sc.render.engine = "CYCLES"
cp = bpy.context.preferences.addons["cycles"].preferences
for dev_type in ("OPTIX", "CUDA"):
    try:
        cp.compute_device_type = dev_type
        cp.get_devices()
        if any(d.type == dev_type for d in cp.devices):
            for d in cp.devices:
                d.use = d.type == dev_type
            sc.cycles.device = "GPU"
            print("cycles device", dev_type)
            break
    except TypeError:
        continue
sc.cycles.samples = samples
sc.cycles.use_denoising = True
sc.render.film_transparent = True
sc.view_settings.view_transform = "Standard"
sc.view_layers[0].use_pass_z = True
if sc.world is None:
    sc.world = bpy.data.worlds.new("w")
keep = bool(sc.get("cm_keep_lights"))  # scenes that set up their own lighting (e.g. plate environment)
if not keep:
    sc.world.color = (0.45, 0.47, 0.5)  # overcast ambient
for o in bpy.data.objects:
    if o.name.endswith(("_points", "_tracks")):
        o.hide_render = True
    if o.type == "LIGHT" and not keep:
        o.data.energy = 2.5
        o.data.angle = 0.35  # soft overcast shadows

# ---- shadow catchers under each cube
for cube in [o for o in bpy.data.objects if o.name.startswith("CG_check_")]:
    s = max(cube.dimensions)
    bpy.ops.mesh.primitive_plane_add(size=1)
    p = bpy.context.object
    p.name = cube.name + "_catcher"
    M = cube.matrix_world.normalized().copy()
    M.translation = cube.matrix_world.translation - M.col[2].xyz * (s / 2 - 1e-4)
    p.matrix_world = M
    p.scale = (7 * s, 7 * s, 1)
    p.is_shadow_catcher = True


# ---- compositor
ng = bpy.data.node_groups.new("Pi4 occlusion comp", "CompositorNodeTree")
sc.compositing_node_group = ng
ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
N, L = ng.nodes, ng.links


def math(op, a=None, b=None, clamp=False, loc=(0, 0)):
    n = N.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    n.location = loc
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            L.new(v, n.inputs[i])
    return n.outputs[0]


def menu(node, name, value):
    try:
        node.inputs[name].default_value = value
    except (KeyError, TypeError):
        setattr(node, name.lower().replace(" ", "_"), value.upper().replace(" ", "_"))


def seq_image(prefix, colorspace):
    first = os.path.join(occ, f"{prefix}_{sc.frame_start:04d}.png")
    img = bpy.data.images.load(first)
    img.source = "SEQUENCE"
    img.colorspace_settings.name = colorspace
    n = N.new("CompositorNodeImage")
    n.image = img
    n.frame_duration = sc.frame_end - sc.frame_start + 1
    n.frame_start = sc.frame_start
    n.frame_offset = sc.frame_start - 1
    n.use_auto_refresh = True
    return n


rl = N.new("CompositorNodeRLayers")
mc = N.new("CompositorNodeMovieClip")
mc.clip = clip

# CG colour -> lens distortion
dist_rgb = N.new("CompositorNodeMovieDistortion")
dist_rgb.clip = clip
menu(dist_rgb, "Type", "Distort")
L.new(rl.outputs["Image"], dist_rgb.inputs["Image"])

# premultiplied inverse depth (background 1e10 -> ~0) packed with alpha, then distorted
inv_cg = math("DIVIDE", 1.0, rl.outputs["Depth"])
inv_pm = math("MULTIPLY", inv_cg, rl.outputs["Alpha"])
pack = N.new("CompositorNodeCombineColor")
L.new(inv_pm, pack.inputs["Red"])
L.new(rl.outputs["Alpha"], pack.inputs["Green"])
dist_d = N.new("CompositorNodeMovieDistortion")
dist_d.clip = clip
menu(dist_d, "Type", "Distort")
L.new(pack.outputs["Image"], dist_d.inputs["Image"])
unpack = N.new("CompositorNodeSeparateColor")
L.new(dist_d.outputs["Image"], unpack.inputs["Image"])
inv_cg_d = math("DIVIDE", unpack.outputs["Red"], math("MAXIMUM", unpack.outputs["Green"], 1e-4))

# Pi4 inverse depth (16-bit PNG, Non-Color) and foreground matte
pi_img = seq_image("invdepth", "Non-Color")
pi_sep = N.new("CompositorNodeSeparateColor")
L.new(pi_img.outputs["Image"], pi_sep.inputs["Image"])
inv_pi = math("MULTIPLY", pi_sep.outputs["Red"], INV_SCALE)
fg_img = seq_image("fgmask", "Non-Color")
fg_sep = N.new("CompositorNodeSeparateColor")
L.new(fg_img.outputs["Image"], fg_sep.inputs["Image"])
holdout = math("SUBTRACT", 1.0, fg_sep.outputs["Red"], clamp=True)

occ_d = float(sc.get("cm_occluder_depth", 0.0))
if occ_d > 0:
    # Occluder-plane mode: only real surfaces nearer than occ_d (e.g. a foreground railing) hide the CG;
    # everything beyond (background with noisy depth) never does. Characters come from the matte.
    d_pi = math("DIVIDE", 1.0, math("MAXIMUM", inv_pi, 1e-4))
    vis = math("DIVIDE", math("SUBTRACT", d_pi, occ_d * 0.92), occ_d * 0.16, clamp=True)
    print(f"occluder-plane depth test at {occ_d:.2f}")
else:
    # soft depth test: visible when Z_cg <= Z_pi4 (1 + TOL)  <=>  inv_cg (1 + TOL) >= inv_pi
    num = math("SUBTRACT", math("MULTIPLY", inv_cg_d, 1 + TOL), inv_pi)
    vis = math("ADD", math("DIVIDE", num, math("MAXIMUM", math("MULTIPLY", inv_pi, SOFT), 1e-6)), 0.5,
               clamp=True)
factor = math("MULTIPLY", vis, holdout, clamp=True)

apply = N.new("CompositorNodeSetAlpha")
menu(apply, "Type", "Apply Mask")
L.new(dist_rgb.outputs["Image"], apply.inputs["Image"])
L.new(factor, apply.inputs["Alpha"])
over = N.new("CompositorNodeAlphaOver")
L.new(mc.outputs["Image"], over.inputs["Background"])
L.new(apply.outputs["Image"], over.inputs["Foreground"])
gout = N.new("NodeGroupOutput")
L.new(over.outputs["Image"], gout.inputs[0])
# debug viewer of the visibility matte
view = N.new("CompositorNodeViewer")
L.new(factor, view.inputs[0])

sc.render.use_compositing = True
sc.render.image_settings.file_format = "PNG"
sc.render.filepath = os.path.join(out_dir, "occl_")
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "cycling_occlusion_comp.blend"))
preview = os.environ.get("CM_FRAMES")  # e.g. "40,180,320" -> render only those frames
# Warm-up: the movie-clip decoder can hand the compositor a black first frame in background renders.
# (A low-res warm-up is not enough: it must be a full-size render.)
sc.frame_set(int(preview.split(",")[0]) if preview else sc.frame_start)
_fp = sc.render.filepath
sc.render.filepath = os.path.join(out_dir, "_warmup.png")
_samples = sc.cycles.samples
sc.cycles.samples = 1
bpy.ops.render.render(write_still=True)
sc.cycles.samples = _samples
sc.render.filepath = _fp
if os.path.exists(os.path.join(out_dir, "_warmup.png")):
    os.remove(os.path.join(out_dir, "_warmup.png"))
if preview:
    for fr in [int(x) for x in preview.split(",")]:
        sc.frame_set(fr)
        sc.render.filepath = os.path.join(out_dir, f"occl_{fr:04d}.png")
        bpy.ops.render.render(write_still=True)
    print("RENDERED preview", preview)
else:
    print("SAVED comp blend; rendering", sc.frame_start, "-", sc.frame_end)
    bpy.ops.render.render(animation=True)
    print("RENDERED")
