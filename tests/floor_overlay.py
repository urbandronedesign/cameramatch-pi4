"""Render the floor mesh with height-contour stripes from the shot camera (for overlay on the plate).

blender -b scene.blend --python tests/floor_overlay.py -- <out_dir> <frames...>
"""
import os
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
out = os.path.abspath(argv[0])
frames = [int(x) for x in argv[1:]]
os.makedirs(out, exist_ok=True)
sc = bpy.context.scene
floor = bpy.data.objects["Pi4_floor"]
floor.visible_camera = True
for o in bpy.data.objects:
    if o.name == "Gold_panels":
        o.hide_render = True
mat = bpy.data.materials.new("contours")
if mat.node_tree is None:
    mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
geo = nt.nodes.new("ShaderNodeNewGeometry")
sep = nt.nodes.new("ShaderNodeSeparateXYZ")
nt.links.new(geo.outputs["Position"], sep.inputs[0])
# stripes every 0.25 units in world Z and X/Y grid every 1 unit -> wireframe-like lines
wave = nt.nodes.new("ShaderNodeTexWave")
wave.wave_type = "BANDS"
wave.bands_direction = "Z"
wave.inputs["Scale"].default_value = 4.0
wave.inputs["Distortion"].default_value = 0
nt.links.new(geo.outputs["Position"], wave.inputs["Vector"])
ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (1, 0.1, 0.6, 1)
ramp.color_ramp.elements[1].color = (0.1, 1, 0.3, 1)
nt.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
em = nt.nodes.new("ShaderNodeEmission")
nt.links.new(ramp.outputs["Color"], em.inputs["Color"])
o = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(em.outputs["Emission"], o.inputs["Surface"])
floor.data.materials.clear()
floor.data.materials.append(mat)
sc.render.engine = "BLENDER_EEVEE"
sc.render.film_transparent = True
sc.render.use_compositing = False
sc.view_settings.view_transform = "Standard"
for f in frames:
    sc.frame_set(f)
    sc.render.filepath = os.path.join(out, f"ov_{f:04d}.png")
    bpy.ops.render.render(write_still=True)
print("DONE")
