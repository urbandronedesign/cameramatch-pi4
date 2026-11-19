"""Floating chrome spheres behind the characters, reflecting the plate, casting shadows.

blender -b <solved_with_matte.blend> --python tests/build_spheres_scene.py -- <out.blend> [ref_frame]
Needs in the job folder: result.npz, pi4_cache.npz, matte/fused, lowpoly.npz, plate_env.png.
- World: plate_env.png (equirect from the plate, mirrored behind the camera).
- Pi4_lowpoly_reflect: low-poly scene mesh, camera-invisible, video projected from the solved camera
  (vertex colours where the camera does not see) -> real nearby surroundings in the reflections.
- Pi4_shadow_catcher: same mesh as a Cycles shadow catcher (receives the spheres' shadows).
- Spheres: placed along image rays at a depth between the characters and the background (Pi4 depth),
  so they float behind the characters; gentle bob + spin.
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
out = os.path.abspath(argv[0])
sc = bpy.context.scene
job = sc.cameramatch.last_job
cam = sc.camera
clip = next(c for c in bpy.data.movieclips if not c.name.endswith(("_motion_mask", "_matte")))
W, H = clip.size
r = np.load(os.path.join(job, "result.npz"))
f_px = float(r["f_px"]) * W / int(r["plate_size"][0])
k1, k2 = float(r["k1"]), float(r["k2"])
idx = r["frame_indices"].astype(int)
T = len(idx)
ref = int(argv[1]) if len(argv) > 1 else sc.frame_start + T // 2
with np.load(os.path.join(job, "pi4_cache.npz")) as z:
    depth = z["depth"].astype(np.float32)
lp = np.load(os.path.join(job, "lowpoly.npz"))
sc["cm_keep_lights"] = True

# ------------------------------------------------------------- depth layering at the reference frame
import struct  # noqa: E402  (PNG read without cv2: use bpy image)

ri = ref - sc.frame_start
img = bpy.data.images.load(os.path.join(job, "matte", "fused", f"alpha_{idx[ri] + 1:04d}.png"))
am = np.array(img.pixels[:], np.float32).reshape(img.size[1], img.size[0], -1)[::-1, :, 0]
bpy.data.images.remove(img)
Dm = depth[ri]
hm, wm = Dm.shape
am_s = am[(np.arange(hm) * am.shape[0] // hm)][:, (np.arange(wm) * am.shape[1] // wm)]
d_char = float(np.median(Dm[am_s > 0.5])) if (am_s > 0.5).any() else float(np.percentile(Dm, 30))
upper = Dm[: hm // 2][am_s[: hm // 2] < 0.2]
d_bg = float(np.percentile(upper, 60)) if upper.size else d_char * 3
d_sph = float(np.clip(d_char + 0.2 * (d_bg - d_char), d_char + 0.8, d_char + 1.5))  # just behind them
print(f"ref frame {ref}: characters {d_char:.2f}, background {d_bg:.2f} -> spheres at {d_sph:.2f}")
sc["cm_occluder_depth"] = 0.85 * d_char  # foreground railing occludes; background depth noise never does


def undistort(xd, yd):
    x, y = xd, yd
    for _ in range(30):
        q = x * x + y * y
        d = 1 + k1 * q + k2 * q * q
        x, y = xd / d, yd / d
    return x, y


sc.frame_set(ref)
Mc = cam.matrix_world.copy()
F0, F1 = sc.frame_start, sc.frame_end
dur = F1 - F0
# Rising spheres: (image x, depth offset behind the characters, radius/depth, start, rise length (frames),
# tint, roughness). They start below the frame (behind the railing), rise past the couple, exit the top.
specs = [(0.24, 0.6, 0.034, 0.00, 0.70, (0.95, 0.95, 0.97), 0.05),
         (0.47, 1.4, 0.026, 0.12, 0.62, (1.00, 0.78, 0.45), 0.08),
         (0.66, 0.9, 0.040, 0.30, 0.66, (0.95, 0.95, 0.97), 0.05),
         (0.85, 1.8, 0.030, 0.05, 0.75, (1.00, 0.78, 0.45), 0.07),
         (0.36, 2.2, 0.022, 0.45, 0.55, (0.95, 0.95, 0.97), 0.05)]
spheres = []
for k, (ux, dz, rf, st, ln, tint, rough) in enumerate(specs):
    d_k = d_char + dz
    rad = rf * d_k
    # image-space path at the reference camera: from just below the bottom edge to just above the top edge
    margin = rad / d_k * f_px / H * 1.2      # sphere radius in frame-height units (+20%)
    pts = []
    for uy in (1.0 + margin, -margin):
        xn, yn = undistort((ux * W - W / 2) / f_px, (uy * H - H / 2) / f_px)
        pts.append(Mc @ Vector((xn * d_k, -yn * d_k, -d_k)))   # Blender camera space (y up, looks -Z)
    p_bot, p_top = pts
    side = (Mc.col[0].xyz).normalized()
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=rad, location=p_bot)
    s = bpy.context.object
    s.name = f"Chrome_sphere_{k}"
    bpy.ops.object.shade_smooth()
    m = bpy.data.materials.new(s.name)
    if m.node_tree is None:
        m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*tint, 1)
    b.inputs["Metallic"].default_value = 1.0
    b.inputs["Roughness"].default_value = rough
    s.data.materials.append(m)
    f_a, f_b = F0 + int(st * dur), F0 + int((st + ln) * dur)
    for fr in range(F0, F1 + 1, 2):
        u = min(max((fr - f_a) / max(f_b - f_a, 1), 0.0), 1.0)
        u = u * u * (3 - 2 * u)                               # ease in / out
        sway = 0.35 * rad * math.sin(2 * math.pi * (fr - F0) / 96 + k * 1.7)
        s.location = p_bot.lerp(p_top, u) + side * sway
        s.rotation_euler = (0.15 * math.sin(0.05 * fr + k), 0.1 * k, 0.03 * (fr - F0) + k)
        s.keyframe_insert("location", frame=fr)
        s.keyframe_insert("rotation_euler", frame=fr)
    spheres.append(s)
    print(f"sphere {k}: radius {rad:.2f}, depth {d_k:.2f}, rises frames {f_a}-{f_b}")

# ------------------------------------------------------------- low-poly mesh: reflections + shadow catcher
V, F, Col = lp["verts"], lp["faces"], lp["colors"]


def make_mesh(name):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.ravel())
    me.loops.add(F.size)
    me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, F.size, 3))
    me.update(calc_edges=True)
    lin = np.where(Col <= 0.04045, Col / 12.92, ((Col + 0.055) / 1.055) ** 2.4)
    ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    ca.data.foreach_set("color", np.c_[lin, np.ones(len(lin))].astype(np.float32).ravel())
    ob = bpy.data.objects.new(name, me)
    sc.collection.objects.link(ob)
    return ob


refl = make_mesh("Pi4_lowpoly_reflect")
mat = bpy.data.materials.new("Pi4_projector_reflect")
if mat.node_tree is None:
    mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
N, L = nt.nodes, nt.links


def mth(op, a, b=None):
    n = N.new("ShaderNodeMath")
    n.operation = op
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            L.new(v, n.inputs[i])
    return n.outputs[0]


tc = N.new("ShaderNodeTexCoord")
tc.object = cam
sp = N.new("ShaderNodeSeparateXYZ")
L.new(tc.outputs["Object"], sp.inputs[0])
nz = mth("MULTIPLY", sp.outputs["Z"], -1.0)
x_ = mth("DIVIDE", sp.outputs["X"], nz)
y_ = mth("DIVIDE", sp.outputs["Y"], nz)
q_ = mth("ADD", mth("MULTIPLY", x_, x_), mth("MULTIPLY", y_, y_))
dd = mth("ADD", mth("ADD", 1.0, mth("MULTIPLY", q_, k1)), mth("MULTIPLY", mth("MULTIPLY", q_, q_), k2))
uu = mth("ADD", mth("DIVIDE", mth("MULTIPLY", mth("MULTIPLY", x_, dd), f_px), W), 0.5)
vv = mth("ADD", mth("DIVIDE", mth("MULTIPLY", mth("MULTIPLY", y_, dd), f_px), H), 0.5)
cb = N.new("ShaderNodeCombineXYZ")
L.new(uu, cb.inputs["X"])
L.new(vv, cb.inputs["Y"])
vid = bpy.data.images.load(bpy.path.abspath(clip.filepath))
vid.source = "MOVIE"
tx = N.new("ShaderNodeTexImage")
tx.image = vid
tx.extension = "CLIP"
tx.image_user.frame_duration = clip.frame_duration
tx.image_user.frame_start = clip.frame_start
tx.image_user.use_auto_refresh = True
L.new(cb.outputs[0], tx.inputs["Vector"])
vis = mth("MULTIPLY", tx.outputs["Alpha"], mth("GREATER_THAN", nz, 1e-4))
vc = N.new("ShaderNodeAttribute")
vc.attribute_name = "Col"
mx = N.new("ShaderNodeMix")
mx.data_type = "RGBA"
L.new(vis, mx.inputs["Factor"])
L.new(vc.outputs["Color"], mx.inputs["A"])
L.new(tx.outputs["Color"], mx.inputs["B"])
em = N.new("ShaderNodeEmission")
L.new(mx.outputs["Result"], em.inputs["Color"])
o = N.new("ShaderNodeOutputMaterial")
L.new(em.outputs[0], o.inputs["Surface"])
refl.data.materials.append(mat)
refl.visible_camera = False      # plate already shows it
refl.visible_shadow = False
# Projected-video geometry smears at grazing angles in mirror reflections: keep it for soft bounce light
# only; the stretched-plate environment gives the clean reflections.
refl.visible_glossy = False
# A second surface coincident with the shadow catcher blocks light to it (whole scene reads as shadowed):
# keep the projected mesh in the file for look-dev, out of the render.
refl.hide_render = True

catcher = make_mesh("Pi4_shadow_catcher")
# Catchers act as holdouts for CG behind them: keep only ground-like (up-facing) faces so noisy foliage /
# railing / wall geometry can never bite into the spheres.
import bmesh  # noqa: E402

upv = Vector(tuple(float(x) for x in lp["up"]))
bm = bmesh.new()
bm.from_mesh(catcher.data)
bm.normal_update()
cam_h = Mc.translation.dot(upv)
# ground only: up-facing AND well below the camera (horizontal Poisson patches up in the foliage would
# otherwise act as holdouts and slice the spheres as they rise through them)
gone = [fc for fc in bm.faces
        if abs(fc.normal.dot(upv)) < 0.75 or fc.calc_center_median().dot(upv) > cam_h - 0.9]
bmesh.ops.delete(bm, geom=gone, context="FACES")
bm.to_mesh(catcher.data)
bm.free()
print(f"shadow catcher: kept {len(catcher.data.polygons)} ground faces (removed {len(gone)})")
catcher.is_shadow_catcher = True
catcher.visible_glossy = False
catcher.visible_diffuse = False
cm = bpy.data.materials.new("catcher")
catcher.data.materials.append(cm)

# ------------------------------------------------------------- lighting: plate environment + soft sun
for ob in [o for o in bpy.data.objects if o.type == "LIGHT"]:
    bpy.data.objects.remove(ob)
if sc.world is None:
    sc.world = bpy.data.worlds.new("World")
w = sc.world
if w.node_tree is None:
    w.use_nodes = True
wn = w.node_tree
wn.nodes.clear()
env = wn.nodes.new("ShaderNodeTexEnvironment")
env.image = bpy.data.images.load(os.path.join(job, "plate_env.png"))
bg = wn.nodes.new("ShaderNodeBackground")
bg.inputs["Strength"].default_value = 1.0
wn.links.new(env.outputs["Color"], bg.inputs["Color"])
wo = wn.nodes.new("ShaderNodeOutputWorld")
wn.links.new(bg.outputs["Background"], wo.inputs["Surface"])
sd = bpy.data.lights.new("Sun", "SUN")
sd.energy = 2.2
sd.angle = math.radians(12)
sun = bpy.data.objects.new("Sun", sd)
sc.collection.objects.link(sun)
# sun high, from behind-left of the camera so shadows fall away from camera onto the deck behind
fwd = -Mc.col[2].xyz
fwd.z = 0
fwd.normalize()
sun_dir = (fwd * 0.55 + Vector((0, 0, -1.0)) + fwd.cross(Vector((0, 0, 1))) * 0.35).normalized()
sun.rotation_euler = sun_dir.to_track_quat("-Z", "Y").to_euler()

for ob in bpy.data.objects:
    if ob.name.endswith(("_points", "_tracks")):
        ob.hide_render = True
bpy.ops.wm.save_as_mainfile(filepath=out)
print("SAVED", out)
