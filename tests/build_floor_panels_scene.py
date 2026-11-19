"""Build the reprojected-floor + golden-panels scene from geometry.npz.

blender -b <solved_with_matte.blend> --python tests/build_floor_panels_scene.py -- <out.blend>
- Pi4_floor: heightfield mesh; projector shader = video sampled through the solved camera with the solved
  Polynomial k1/k2 (exactly what the camera saw), grey where the camera doesn't see. Camera-invisible in the
  shot render (the plate already shows it) but visible to reflections and to other cameras.
- Gold_panels: metallic quads on the reconstructed facades.
"""
import math
import os
import sys

import bpy
import numpy as np

out_blend = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
sc = bpy.context.scene
job = sc.cameramatch.last_job
cam = sc.camera
clip = next(c for c in bpy.data.movieclips if not c.name.endswith(("_motion_mask", "_matte")))
g = np.load(os.path.join(job, "geometry.npz"))
r = np.load(os.path.join(job, "result.npz"))
W, H = clip.size
f_px = float(r["f_px"]) * W / int(r["plate_size"][0])
k1, k2 = float(r["k1"]), float(r["k2"])
print("plate", W, H, "f_px", round(f_px, 2), "k", round(k1, 4), round(k2, 4))

# ---------------------------------------------------------------- floor mesh
fv, ff = g["floor_verts"], g["floor_faces"]
me = bpy.data.meshes.new("Pi4_floor")
me.vertices.add(len(fv))
me.vertices.foreach_set("co", fv.ravel())
me.loops.add(ff.size)
me.loops.foreach_set("vertex_index", ff.ravel())
me.polygons.add(len(ff))
me.polygons.foreach_set("loop_start", np.arange(0, ff.size, 4))
me.update(calc_edges=True)
floor = bpy.data.objects.new("Pi4_floor", me)
sc.collection.objects.link(floor)
for p in me.polygons:
    p.use_smooth = True

# projector material
mat = bpy.data.materials.new("Pi4_projector")
if mat.node_tree is None:
    mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
N, L = nt.nodes, nt.links


def m(op, a, b=None, clamp=False):
    n = N.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            L.new(v, n.inputs[i])
    return n.outputs[0]


tc = N.new("ShaderNodeTexCoord")
tc.object = cam  # coordinates in the shot camera's space (looks down -Z, Y up)
sep = N.new("ShaderNodeSeparateXYZ")
L.new(tc.outputs["Object"], sep.inputs[0])
negz = m("MULTIPLY", sep.outputs["Z"], -1.0)
xn = m("DIVIDE", sep.outputs["X"], negz)
yn = m("DIVIDE", sep.outputs["Y"], negz)
r2 = m("ADD", m("MULTIPLY", xn, xn), m("MULTIPLY", yn, yn))
dist = m("ADD", m("ADD", 1.0, m("MULTIPLY", r2, k1)), m("MULTIPLY", m("MULTIPLY", r2, r2), k2))
u = m("ADD", m("DIVIDE", m("MULTIPLY", m("MULTIPLY", xn, dist), f_px), W), 0.5)
v = m("ADD", m("DIVIDE", m("MULTIPLY", m("MULTIPLY", yn, dist), f_px), H), 0.5)
comb = N.new("ShaderNodeCombineXYZ")
L.new(u, comb.inputs["X"])
L.new(v, comb.inputs["Y"])
img = bpy.data.images.load(bpy.path.abspath(clip.filepath))
img.source = "MOVIE"
tex = N.new("ShaderNodeTexImage")
tex.image = img
tex.extension = "CLIP"
tex.interpolation = "Cubic"
tex.image_user.frame_duration = clip.frame_duration
tex.image_user.frame_start = clip.frame_start
tex.image_user.use_auto_refresh = True
L.new(comb.outputs[0], tex.inputs["Vector"])
infront = m("GREATER_THAN", negz, 1e-4)
vis = m("MULTIPLY", tex.outputs["Alpha"], infront)
mix = N.new("ShaderNodeMix")
mix.data_type = "RGBA"
mix.inputs["A"].default_value = (0.18, 0.18, 0.2, 1)
L.new(vis, mix.inputs["Factor"])
L.new(tex.outputs["Color"], mix.inputs["B"])
emit = N.new("ShaderNodeEmission")
L.new(mix.outputs["Result"], emit.inputs["Color"])
out = N.new("ShaderNodeOutputMaterial")
L.new(emit.outputs["Emission"], out.inputs["Surface"])
mat.use_fake_user = True  # live projector kept in the file (shot-camera-only reprojection)

# baked reprojection (every texel from the best frame) on grid UVs -> textured from any viewpoint
uv = me.uv_layers.new(name="grid")
fuv = g["floor_uv"]
uv.data.foreach_set("uv", fuv[ff.ravel()].ravel())
bake = bpy.data.materials.new("Pi4_floor_baked")
if bake.node_tree is None:
    bake.use_nodes = True
bn = bake.node_tree
bn.nodes.clear()
bt = bn.nodes.new("ShaderNodeTexImage")
bt.image = bpy.data.images.load(os.path.join(job, "floor_texture.png"))
bt.interpolation = "Cubic"
be = bn.nodes.new("ShaderNodeEmission")
bn.links.new(bt.outputs["Color"], be.inputs["Color"])
bo = bn.nodes.new("ShaderNodeOutputMaterial")
bn.links.new(be.outputs["Emission"], bo.inputs["Surface"])
me.materials.append(bake)
floor.visible_camera = False      # shot camera already has it in the plate
floor.visible_shadow = False

# ---------------------------------------------------------------- visible floor effect: glowing gold grid
grid = bpy.data.objects.new("Gold_grid", me.copy())
sc.collection.objects.link(grid)
upv = g["up"]
grid.location = tuple(0.04 * upv)  # lift a hair above the reconstructed ground
gm = bpy.data.materials.new("Gold_grid")
if gm.node_tree is None:
    gm.use_nodes = True
gn = gm.node_tree
gn.nodes.clear()
GN, GL = gn.nodes, gn.links


def gmath(op, a, b=None, clamp=False):
    n = GN.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            GL.new(v, n.inputs[i])
    return n.outputs[0]


geo = GN.new("ShaderNodeNewGeometry")
sxyz = GN.new("ShaderNodeSeparateXYZ")
GL.new(geo.outputs["Position"], sxyz.inputs[0])
SPACING, WIDTH = 1.0, 0.014


def line(coord):
    # distance (scene units) to the nearest grid line -> soft line mask
    d = gmath("MULTIPLY", gmath("ABSOLUTE", gmath("SUBTRACT", gmath("FRACT", gmath("ADD", gmath(
        "DIVIDE", coord, SPACING), 0.5)), 0.5)), SPACING)
    mr = GN.new("ShaderNodeMapRange")
    mr.interpolation_type = "SMOOTHSTEP"
    mr.inputs["From Min"].default_value = WIDTH
    mr.inputs["From Max"].default_value = 3.0 * WIDTH
    mr.inputs["To Min"].default_value = 1.0
    mr.inputs["To Max"].default_value = 0.0
    GL.new(d, mr.inputs["Value"])
    return mr.outputs["Result"]


lines_mask = gmath("MAXIMUM", line(sxyz.outputs["X"]), line(sxyz.outputs["Y"]))
camd = GN.new("ShaderNodeCameraData")
fade = gmath("SUBTRACT", 1.0, gmath("DIVIDE", camd.outputs["View Distance"], 14.0), clamp=True)
mask = gmath("MULTIPLY", gmath("ADD", lines_mask, 0.07), fade, clamp=True)  # lines + faint gold film
emis = GN.new("ShaderNodeEmission")
emis.inputs["Color"].default_value = (1.0, 0.52, 0.1, 1)
emis.inputs["Strength"].default_value = 1.6
transp = GN.new("ShaderNodeBsdfTransparent")
mixg = GN.new("ShaderNodeMixShader")
GL.new(mask, mixg.inputs["Fac"])
GL.new(transp.outputs[0], mixg.inputs[1])
GL.new(emis.outputs[0], mixg.inputs[2])
gout = GN.new("ShaderNodeOutputMaterial")
GL.new(mixg.outputs[0], gout.inputs["Surface"])
grid.data.materials.clear()
grid.data.materials.append(gm)
sm = grid.modifiers.new("smooth_wobble", "SMOOTH")  # irons out heightfield noise, keeps the step layout
sm.factor = 0.6
sm.iterations = 4
grid.visible_shadow = False

# ---------------------------------------------------------------- golden panels
pq = g["panels"]
pm = bpy.data.meshes.new("Gold_panels")
pv = pq.reshape(-1, 3)
pm.vertices.add(len(pv))
pm.vertices.foreach_set("co", pv.ravel())
pm.loops.add(len(pv))
pm.loops.foreach_set("vertex_index", np.arange(len(pv)))
pm.polygons.add(len(pq))
pm.polygons.foreach_set("loop_start", np.arange(0, len(pv), 4))
pm.update(calc_edges=True)
panels = bpy.data.objects.new("Gold_panels", pm)
sc.collection.objects.link(panels)
sol = panels.modifiers.new("thickness", "SOLIDIFY")
sol.thickness = 0.08
bev = panels.modifiers.new("bevel", "BEVEL")
bev.width = 0.04
bev.segments = 2
gold = bpy.data.materials.new("Gold")
if gold.node_tree is None:
    gold.use_nodes = True
b = gold.node_tree.nodes.get("Principled BSDF")
b.inputs["Base Color"].default_value = (1.0, 0.71, 0.29, 1)
b.inputs["Metallic"].default_value = 1.0
b.inputs["Roughness"].default_value = 0.07
gold.diffuse_color = (1.0, 0.766, 0.336, 1)
gold.metallic = 1.0
pm.materials.append(gold)
print("panels:", len(pq))

# ---------------------------------------------------------------- light + sky
for o in [o for o in bpy.data.objects if o.type == "LIGHT"]:
    bpy.data.objects.remove(o)
sun_d = bpy.data.lights.new("Sun", "SUN")
sun_d.energy = 4.0
sun_d.angle = math.radians(8)
sun = bpy.data.objects.new("Sun", sun_d)
sun.rotation_euler = (math.radians(55), 0, math.radians(35))
sc.collection.objects.link(sun)
if sc.world is None:
    sc.world = bpy.data.worlds.new("World")
w = sc.world
if w.node_tree is None:
    w.use_nodes = True
wn = w.node_tree
wn.nodes.clear()
bg = wn.nodes.new("ShaderNodeBackground")
bg.inputs["Strength"].default_value = 1.0
grad_tc = wn.nodes.new("ShaderNodeTexCoord")
sepw = wn.nodes.new("ShaderNodeSeparateXYZ")
wn.links.new(grad_tc.outputs["Generated"], sepw.inputs[0])
ramp = wn.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (0.35, 0.33, 0.3, 1)    # ground bounce
ramp.color_ramp.elements[1].color = (0.45, 0.6, 1.0, 1)     # zenith
ramp.color_ramp.elements[0].position = 0.48
hz = ramp.color_ramp.elements.new(0.52)
hz.color = (2.2, 2.1, 1.9, 1)                                 # bright overcast horizon band
ramp.color_ramp.elements[-1].position = 0.75
wn.links.new(sepw.outputs["Z"], ramp.inputs["Fac"])
wn.links.new(ramp.outputs["Color"], bg.inputs["Color"])
# camera rays see a soft neutral sky; reflections (gold) see the bright horizon band
cam_bg = wn.nodes.new("ShaderNodeBackground")
cam_bg.inputs["Color"].default_value = (0.62, 0.68, 0.78, 1)
lp = wn.nodes.new("ShaderNodeLightPath")
mixw = wn.nodes.new("ShaderNodeMixShader")
wn.links.new(lp.outputs["Is Camera Ray"], mixw.inputs["Fac"])
wn.links.new(bg.outputs["Background"], mixw.inputs[1])
wn.links.new(cam_bg.outputs["Background"], mixw.inputs[2])
wo = wn.nodes.new("ShaderNodeOutputWorld")
wn.links.new(mixw.outputs["Shader"], wo.inputs["Surface"])

for o in bpy.data.objects:
    if o.name.endswith(("_points", "_tracks")):
        o.hide_render = True
bpy.ops.wm.save_as_mainfile(filepath=out_blend)
print("SAVED", out_blend)
