"""Switch the floor/grid scene to the low-poly mesh (lowpoly.npz).

blender -b scene_final.blend --python tests/use_lowpoly.py -- <out.blend>
- Pi4_lowpoly: low-poly scene mesh, vertex colours (emission), flat shaded; camera-invisible in the shot
  (the plate already shows it), visible to reflections and to the witness camera.
- Gold_grid: rebuilt on the low-poly mesh; grid only on up-facing faces (normal . up), lifted a hair.
"""
import os
import sys

import bpy
import numpy as np

out = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
sc = bpy.context.scene
d = np.load(os.path.join(sc.cameramatch.last_job, "lowpoly.npz"))
V, Fc, Col, up = d["verts"], d["faces"], d["colors"], d["up"]

for name in ("Pi4_floor", "Gold_grid"):
    o = bpy.data.objects.get(name)
    if o is not None:
        o.hide_render = True
        o.hide_viewport = True

me = bpy.data.meshes.new("Pi4_lowpoly")
me.vertices.add(len(V))
me.vertices.foreach_set("co", V.ravel())
me.loops.add(Fc.size)
me.loops.foreach_set("vertex_index", Fc.ravel())
me.polygons.add(len(Fc))
me.polygons.foreach_set("loop_start", np.arange(0, Fc.size, 3))
me.update(calc_edges=True)
lin = np.where(Col <= 0.04045, Col / 12.92, ((Col + 0.055) / 1.055) ** 2.4)
ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
ca.data.foreach_set("color", np.c_[lin, np.ones(len(lin))].astype(np.float32).ravel())
lp = bpy.data.objects.new("Pi4_lowpoly", me)
sc.collection.objects.link(lp)
m = bpy.data.materials.new("Pi4_lowpoly_vcol")
if m.node_tree is None:
    m.use_nodes = True
nt = m.node_tree
nt.nodes.clear()
attr = nt.nodes.new("ShaderNodeAttribute")
attr.attribute_name = "Col"
em = nt.nodes.new("ShaderNodeEmission")
nt.links.new(attr.outputs["Color"], em.inputs["Color"])
o = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(em.outputs[0], o.inputs[0])
me.materials.append(m)
lp.visible_camera = False
lp.visible_shadow = False

# grid on the low-poly mesh, up-facing faces only
gmesh = me.copy()
grid = bpy.data.objects.new("Gold_grid_lowpoly", gmesh)
sc.collection.objects.link(grid)
grid.location = tuple(0.03 * up)
gmat = bpy.data.materials["Gold_grid"].copy()
gmat.name = "Gold_grid_lowpoly"
gn = gmat.node_tree
mix = next(n for n in gn.nodes if n.type == "MIX_SHADER")
fac_src = mix.inputs["Fac"].links[0].from_socket
geo = gn.nodes.new("ShaderNodeNewGeometry")
dot = gn.nodes.new("ShaderNodeVectorMath")
dot.operation = "DOT_PRODUCT"
dot.inputs[1].default_value = tuple(float(x) for x in up)
gn.links.new(geo.outputs["True Normal"], dot.inputs[0])
upmask = gn.nodes.new("ShaderNodeMapRange")
upmask.inputs["From Min"].default_value = 0.7
upmask.inputs["From Max"].default_value = 0.85
gn.links.new(dot.outputs["Value"], upmask.inputs["Value"])
mul = gn.nodes.new("ShaderNodeMath")
mul.operation = "MULTIPLY"
gn.links.new(fac_src, mul.inputs[0])
gn.links.new(upmask.outputs["Result"], mul.inputs[1])
gn.links.new(mul.outputs[0], mix.inputs["Fac"])
gmesh.materials.clear()
gmesh.materials.append(gmat)
grid.visible_shadow = False
bpy.ops.wm.save_as_mainfile(filepath=out)
print("SAVED", out, len(V), "verts", len(Fc), "tris")
