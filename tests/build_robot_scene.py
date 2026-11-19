"""Photoreal walking robot in the solved street shot.

blender -b <solved.blend> --python tests/build_robot_scene.py -- <out.blend> [y_start]
Needs in the job folder: result.npz, lowpoly.npz, plate_env.png; Pi4 point cloud object in the .blend.
- Floor: robust smooth fit to the Pi4 floor points (quadratic in the walking direction) -> Cycles shadow
  catcher (plate-projected albedo for bounce light onto the robot).
- Walls: low-poly Pi4 mesh minus the floor, emitting the projected plate (camera-invisible): the alley's
  real bounce light / reflections, and it blocks the sky like the real walls.
- World: plate environment. No fake lights: the robot is lit by the alley.
- Robot: tests/robot_rig.py, walking toward the camera along the street centreline, feet planted on the
  fitted floor; turns its head to the camera at the end.
Render: Cycles, motion blur, multilayer EXR with the Shadow Catcher pass (composited by tools/comp_robot.py).
"""
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robot_rig  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
out = os.path.abspath(argv[0])
Y_START = float(argv[1]) if len(argv) > 1 else 17.0
sc = bpy.context.scene
job = sc.cameramatch.last_job
cam = sc.camera
clip = next(c for c in bpy.data.movieclips if not c.name.endswith(("_motion_mask", "_matte")))
W, H = clip.size
sc.render.fps = int(round(clip.fps)) if clip.fps else 25
sc.render.fps_base = 1.0
sc.frame_start, sc.frame_end = 1, clip.frame_duration
F0, F1 = sc.frame_start, sc.frame_end
r = np.load(os.path.join(job, "result.npz"))
f_px = float(r["f_px"]) * W / int(r["plate_size"][0])
k1, k2 = float(r["k1"]), float(r["k2"])
lp = np.load(os.path.join(job, "lowpoly.npz"))
upv = Vector(tuple(float(x) for x in lp["up"]))
print("fps", sc.render.fps, "frames", F0, F1, "up", tuple(round(x, 3) for x in upv), "f_px", round(f_px, 1))

# ------------------------------------------------------------- floor + centreline from the Pi4 points
po = next(o for o in bpy.data.objects if o.name.endswith("_points"))
Mw = np.array(po.matrix_world)
V = np.empty(len(po.data.vertices) * 3, np.float32)
po.data.vertices.foreach_get("co", V)
P = V.reshape(-1, 3) @ Mw[:3, :3].T + Mw[:3, 3]
ys, mids = [], []
for y0 in np.arange(3.0, 26.0, 1.0):
    S = P[(P[:, 1] > y0) & (P[:, 1] < y0 + 1.0) & (P[:, 2] > -1.0) & (P[:, 2] < 1.0)]
    if len(S) > 100:
        ys.append(y0 + 0.5)
        mids.append(0.5 * (np.percentile(S[:, 0], 2) + np.percentile(S[:, 0], 98)))
cx = np.polyfit(ys, mids, 2)
lefts, rights = [], []
for y0 in ys:
    S = P[(P[:, 1] > y0 - 0.5) & (P[:, 1] < y0 + 0.5) & (P[:, 2] > -1.0) & (P[:, 2] < 1.0)]
    lefts.append(np.percentile(S[:, 0], 2))
    rights.append(np.percentile(S[:, 0], 98))
cl, cr = np.polyfit(ys, lefts, 2), np.polyfit(ys, rights, 2)
print("walls at y=6/12/20:", [(round(float(np.polyval(cl, y)), 2), round(float(np.polyval(cr, y)), 2)) for y in (6, 12, 20)])
xc = lambda y: float(np.polyval(cx, y))  # noqa: E731
sel = (P[:, 1] > 3) & (P[:, 1] < 26) & (np.abs(P[:, 0] - np.polyval(cx, P[:, 1])) < 0.7) & (P[:, 2] < -1.3)
Q = P[sel]
keep = np.ones(len(Q), bool)
for it in range(6):
    A = np.c_[np.ones(len(Q)), Q[:, 0], Q[:, 1], Q[:, 1] ** 2]
    coef, *_ = np.linalg.lstsq(A[keep], Q[keep, 2], rcond=None)
    res = Q[:, 2] - A @ coef
    sig = 1.4826 * np.median(np.abs(res[keep]))
    keep = np.abs(res) < max(2.5 * sig, 0.015)
print(f"floor fit: z = {coef[0]:.3f} + {coef[1]:.4f} x + {coef[2]:.4f} y + {coef[3]:.5f} y^2, "
      f"robust sigma {sig * 100:.1f} cm, inliers {keep.sum()}/{len(Q)}")
floor_z = lambda x, y: float(coef[0] + coef[1] * x + coef[2] * y + coef[3] * y * y)  # noqa: E731
sc["cm_floor_coef"] = [float(c) for c in coef]

# walking path: street centreline from Y_START toward the camera (arc-length parameterised)
yy = np.linspace(Y_START, 2.5, 2000)
xy = np.c_[[xc(y) for y in yy], yy]
seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
sarc = np.r_[0, np.cumsum(seg)]


def path(s):
    s = min(max(s, 0.0), sarc[-1] - 1e-3)
    i = int(np.searchsorted(sarc, s) - 1)
    i = max(0, min(i, len(seg) - 1))
    u = (s - sarc[i]) / seg[i]
    p = xy[i] * (1 - u) + xy[i + 1] * u
    t = (xy[i + 1] - xy[i]) / seg[i]
    return Vector((p[0], p[1])), Vector((t[0], t[1]))


# ------------------------------------------------------------- projected-plate material
vid = bpy.data.images.load(bpy.path.abspath(clip.filepath))
vid.source = "MOVIE"


def projector_material(name, emissive, albedo_gain=1.0):
    mat = bpy.data.materials.new(name)
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
    o = N.new("ShaderNodeOutputMaterial")
    if emissive:
        em = N.new("ShaderNodeEmission")
        L.new(mx.outputs["Result"], em.inputs["Color"])
        L.new(em.outputs[0], o.inputs["Surface"])
    else:
        g = N.new("ShaderNodeMix")
        g.data_type = "RGBA"
        g.blend_type = "MULTIPLY"
        g.inputs["Factor"].default_value = 1.0
        L.new(mx.outputs["Result"], g.inputs["A"])
        g.inputs["B"].default_value = (albedo_gain, albedo_gain, albedo_gain, 1)
        df = N.new("ShaderNodeBsdfDiffuse")
        L.new(g.outputs["Result"], df.inputs["Color"])
        L.new(df.outputs[0], o.inputs["Surface"])
    return mat


# ------------------------------------------------------------- walls (low-poly minus floor), emissive plate
Vl, Fl, Cl = lp["verts"], lp["faces"], lp["colors"]
me = bpy.data.meshes.new("Pi4_walls")
me.vertices.add(len(Vl))
me.vertices.foreach_set("co", Vl.ravel())
me.loops.add(Fl.size)
me.loops.foreach_set("vertex_index", Fl.ravel())
me.polygons.add(len(Fl))
me.polygons.foreach_set("loop_start", np.arange(0, Fl.size, 3))
me.update(calc_edges=True)
lin = np.where(Cl <= 0.04045, Cl / 12.92, ((Cl + 0.055) / 1.055) ** 2.4)
ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
ca.data.foreach_set("color", np.c_[lin, np.ones(len(lin))].astype(np.float32).ravel())
walls = bpy.data.objects.new("Pi4_walls", me)
sc.collection.objects.link(walls)
bm = bmesh.new()
bm.from_mesh(me)
bm.normal_update()
gone = []
for fc in bm.faces:
    c = fc.calc_center_median()
    if abs(fc.normal.dot(upv)) > 0.7 and c.z < floor_z(c.x, c.y) + 0.35:
        gone.append(fc)
bmesh.ops.delete(bm, geom=gone, context="FACES")
bm.to_mesh(me)
bm.free()
me.materials.append(projector_material("Pi4_walls_projected", emissive=True))
walls.visible_camera = False
walls.is_shadow_catcher = True   # real-world geometry: its occlusion of the sky is already in the plate
print(f"walls: {len(me.polygons)} faces (floor faces removed: {len(gone)})")

# ------------------------------------------------------------- proxy walls: the real ~12 m high calle walls
# The Pi4 mesh only covers what the camera saw (walls end at the frame top, nothing behind the camera), so
# without these the robot would be lit by open sky from the sides instead of the slit of sky overhead.
wall_lin = np.median(lin, axis=0)
pw_v, pw_f, pw_c = [], [], []
yv_w = np.linspace(-25.0, 45.0, 141)
zs = np.linspace(-0.6, 14.0, 30)
for sgn, cf in ((-1, cl), (1, cr)):
    base = len(pw_v)
    for y in yv_w:
        yy_c = min(max(y, ys[0]), ys[-1])            # extrapolate flat beyond the measured range
        x = float(np.polyval(cf, yy_c)) + sgn * 0.08
        for z in zs:
            pw_v.append((x, y, floor_z(x, y) + z))
            pw_c.append(wall_lin)
    nz_ = len(zs)
    for j in range(len(yv_w) - 1):
        for i in range(nz_ - 1):
            a0 = base + j * nz_ + i
            pw_f.append((a0, a0 + nz_, a0 + nz_ + 1, a0 + 1) if sgn < 0 else (a0, a0 + 1, a0 + nz_ + 1, a0 + nz_))
pm = bpy.data.meshes.new("Proxy_walls")
pm.from_pydata(pw_v, [], pw_f)
pm.update()
pca = pm.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
pca.data.foreach_set("color", np.c_[np.array(pw_c), np.ones(len(pw_c))].astype(np.float32).ravel())
proxy = bpy.data.objects.new("Proxy_walls", pm)
sc.collection.objects.link(proxy)
pm.materials.append(bpy.data.materials["Pi4_walls_projected"])
proxy.visible_camera = False
proxy.is_shadow_catcher = True
print("proxy walls: median wall colour (linear)", np.round(wall_lin, 3))

# ------------------------------------------------------------- floor shadow catcher (smooth fit)
nx, ny = 24, 160
fm = bpy.data.meshes.new("Pi4_floor_catcher")
verts, faces = [], []
yv = np.linspace(2.0, 30.0, ny)
for j, y in enumerate(yv):
    for i in range(nx):
        x = xc(y) + (i / (nx - 1) - 0.5) * 2.6
        verts.append((x, y, floor_z(x, y)))
for j in range(ny - 1):
    for i in range(nx - 1):
        a = j * nx + i
        faces.append((a, a + 1, a + nx + 1, a + nx))
fm.from_pydata(verts, [], faces)
fm.update()
# keep the normals up
if fm.polygons[0].normal.z < 0:
    fm.flip_normals()
floor = bpy.data.objects.new("Pi4_floor_catcher", fm)
sc.collection.objects.link(floor)
fca = fm.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
fca.data.foreach_set("color", np.tile([0.1, 0.1, 0.1, 1.0], len(verts)).astype(np.float32))
fm.materials.append(projector_material("Pi4_floor_projected", emissive=False, albedo_gain=1.6))
floor.is_shadow_catcher = True

# ------------------------------------------------------------- world: plate environment, no fake lights
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
wn.links.new(env.outputs["Color"], bg.inputs["Color"])
wo = wn.nodes.new("ShaderNodeOutputWorld")
wn.links.new(bg.outputs["Background"], wo.inputs["Surface"])


# ------------------------------------------------------------- robot materials
def principled(name):
    m = bpy.data.materials.new(name)
    if m.node_tree is None:
        m.use_nodes = True
    return m, m.node_tree, m.node_tree.nodes.get("Principled BSDF")


FLOOR0 = floor_z(xc(10.0), 10.0)


def paint_material():
    m, nt, b = principled("Robot_paint")
    N, L = nt.nodes, nt.links

    def node(t, **kw):
        n = N.new(t)
        for k, v in kw.items():
            setattr(n, k, v)
        return n

    def math_(op, a, bb=None, clamp=False):
        n = node("ShaderNodeMath", operation=op, use_clamp=clamp)
        for i, v in enumerate((a, bb)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                L.new(v, n.inputs[i])
        return n.outputs[0]

    def mrange(v, a, bb, c=0.0, d=1.0):
        n = node("ShaderNodeMapRange")
        L.new(v, n.inputs["Value"])
        n.inputs["From Min"].default_value, n.inputs["From Max"].default_value = a, bb
        n.inputs["To Min"].default_value, n.inputs["To Max"].default_value = c, d
        return n.outputs["Result"]

    tc = node("ShaderNodeTexCoord")
    geo = node("ShaderNodeNewGeometry")
    ao = node("ShaderNodeAmbientOcclusion", samples=8, only_local=True)
    ao.inputs["Distance"].default_value = 0.05
    crev = math_("SUBTRACT", 1.0, ao.outputs["AO"])
    n1 = node("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = 7.0
    n1.inputs["Detail"].default_value = 8.0
    L.new(tc.outputs["Object"], n1.inputs["Vector"])
    n2 = node("ShaderNodeTexNoise")
    n2.inputs["Scale"].default_value = 38.0
    n2.inputs["Detail"].default_value = 6.0
    L.new(tc.outputs["Object"], n2.inputs["Vector"])
    smudge = mrange(n1.outputs["Fac"], 0.5, 0.78)
    sepw = node("ShaderNodeSeparateXYZ")
    L.new(geo.outputs["Position"], sepw.inputs[0])
    low = mrange(sepw.outputs["Z"], FLOOR0 + 0.05, FLOOR0 + 0.5, 0.75, 0.0)
    grime = math_("MINIMUM", math_("ADD", math_("ADD", math_("MULTIPLY", crev, 1.3), math_("MULTIPLY", smudge, 0.3)),
                                   math_("MULTIPLY", low, mrange(n2.outputs["Fac"], 0.35, 0.65))), 1.0)
    # edge wear: bevel-normal vs true normal
    bev = node("ShaderNodeBevel", samples=8)
    bev.inputs["Radius"].default_value = 0.006
    dot = node("ShaderNodeVectorMath", operation="DOT_PRODUCT")
    L.new(bev.outputs["Normal"], dot.inputs[0])
    L.new(geo.outputs["True Normal"], dot.inputs[1])
    edge = mrange(dot.outputs["Value"], 0.995, 0.93)
    wear = math_("MULTIPLY", edge, mrange(n2.outputs["Fac"], 0.5, 0.62), clamp=True)
    base = node("ShaderNodeMix", data_type="RGBA")
    L.new(math_("MULTIPLY", grime, 0.85), base.inputs["Factor"])
    base.inputs["A"].default_value = (0.50, 0.49, 0.46, 1)
    base.inputs["B"].default_value = (0.075, 0.062, 0.05, 1)
    base2 = node("ShaderNodeMix", data_type="RGBA")
    L.new(wear, base2.inputs["Factor"])
    L.new(base.outputs["Result"], base2.inputs["A"])
    base2.inputs["B"].default_value = (0.58, 0.58, 0.6, 1)
    L.new(base2.outputs["Result"], b.inputs["Base Color"])
    L.new(wear, b.inputs["Metallic"])
    rough = math_("ADD", math_("ADD", 0.3, math_("MULTIPLY", n1.outputs["Fac"], 0.14)), math_("MULTIPLY", grime, 0.3))
    L.new(math_("MINIMUM", rough, 0.85), b.inputs["Roughness"])
    L.new(math_("MULTIPLY", math_("SUBTRACT", 1.0, grime, clamp=True), 0.35), b.inputs["Coat Weight"])
    b.inputs["Coat Roughness"].default_value = 0.18
    bump = node("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.025
    n3 = node("ShaderNodeTexNoise")
    n3.inputs["Scale"].default_value = 260.0
    L.new(tc.outputs["Object"], n3.inputs["Vector"])
    L.new(n3.outputs["Fac"], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def simple(name, base, metal, rough, coat=0.0, emit=None, rough_noise=0.0):
    m, nt, b = principled(name)
    b.inputs["Base Color"].default_value = (*base, 1)
    b.inputs["Metallic"].default_value = metal
    b.inputs["Roughness"].default_value = rough
    b.inputs["Coat Weight"].default_value = coat
    b.inputs["Coat Roughness"].default_value = 0.03
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit[0], 1)
        b.inputs["Emission Strength"].default_value = emit[1]
    if rough_noise:
        n = nt.nodes.new("ShaderNodeTexNoise")
        n.inputs["Scale"].default_value = 25.0
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = rough - rough_noise
        mr.inputs["To Max"].default_value = rough + rough_noise
        nt.links.new(n.outputs["Fac"], mr.inputs["Value"])
        nt.links.new(mr.outputs["Result"], b.inputs["Roughness"])
    return m


mats = {"paint": paint_material(),
        "graphite": simple("Robot_graphite", (0.028, 0.029, 0.033), 0.85, 0.42, rough_noise=0.08),
        "rubber": simple("Robot_rubber", (0.02, 0.02, 0.02), 0.0, 0.85),
        "chrome": simple("Robot_chrome", (0.85, 0.85, 0.86), 1.0, 0.08),
        "visor": simple("Robot_visor", (0.004, 0.004, 0.005), 0.0, 0.04, coat=1.0),
        "led": simple("Robot_led", (0.0, 0.0, 0.0), 0.0, 0.3, emit=((0.55, 0.85, 1.0), 0.12)),
        "decal": simple("Robot_decal", (0.035, 0.035, 0.038), 0.0, 0.5)}

coll = bpy.data.collections.new("Robot")
sc.collection.children.link(coll)
parts = robot_rig.build_robot(coll, mats)


def cam_pos(fr):
    sc.frame_set(fr)
    return cam.matrix_world.translation.copy()


bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
cam_track = {fr: cam_pos(fr) for fr in range(F0, F1 + 1)}
stats = robot_rig.animate_walk(parts, path, floor_z, F0, F1, sc.render.fps, look_target=lambda fr: cam_track[fr],
                               look_from=int(F0 + 0.62 * (F1 - F0)))
print("walk IK:", stats)
sc.frame_set(F1)
end = parts["pelvis"].matrix_world.translation
print(f"robot start y {Y_START:.1f}, end {tuple(round(x, 2) for x in end)}, "
      f"distance to camera at end {(end - cam_track[F1]).length:.2f} m")

# ------------------------------------------------------------- render settings
for ob in bpy.data.objects:
    if ob.name.endswith(("_points", "_tracks")):
        ob.hide_render = True
sc.render.engine = "CYCLES"
cp = bpy.context.preferences.addons["cycles"].preferences
cp.compute_device_type = "OPTIX"
cp.get_devices()
for d in cp.devices:
    d.use = d.type == "OPTIX"
sc.cycles.device = "GPU"
sc.cycles.samples = 256
sc.cycles.adaptive_threshold = 0.008
sc.cycles.use_denoising = True
sc.cycles.max_bounces = 10
sc.render.film_transparent = True
sc.render.use_motion_blur = True
sc.render.motion_blur_shutter = 0.5
sc.view_settings.view_transform = "Standard"
sc.view_settings.look = "None"
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
vl = sc.view_layers[0]
vl.cycles.use_pass_shadow_catcher = True
vl.use_pass_z = True
sc.render.use_compositing = False
im = sc.render.image_settings
if hasattr(im, "media_type"):
    im.media_type = "MULTI_LAYER_IMAGE"
    im.file_format = "OPEN_EXR_MULTILAYER"
else:
    im.file_format = "OPEN_EXR_MULTILAYER"
im.color_depth = "16"
im.exr_codec = "DWAA"
bpy.ops.wm.save_as_mainfile(filepath=out)
print("SAVED", out)
