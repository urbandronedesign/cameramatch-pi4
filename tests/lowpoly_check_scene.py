"""Swap the heightfield floor for the low-poly mesh (projector material) so the floor checks can run on it.

blender -b scene_final.blend --python tests/lowpoly_check_scene.py -- <out.blend>
"""
import os
import sys

import bpy
import numpy as np

out = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
sc = bpy.context.scene
d = np.load(os.path.join(sc.cameramatch.last_job, "lowpoly.npz"))
V, Fc = d["verts"], d["faces"]
old = bpy.data.objects["Pi4_floor"]
old.name = "Pi4_floor_heightfield"
old.hide_render = True
me = bpy.data.meshes.new("Pi4_lowpoly")
me.vertices.add(len(V))
me.vertices.foreach_set("co", V.ravel())
me.loops.add(Fc.size)
me.loops.foreach_set("vertex_index", Fc.ravel())
me.polygons.add(len(Fc))
me.polygons.foreach_set("loop_start", np.arange(0, Fc.size, 3))
me.update(calc_edges=True)
ob = bpy.data.objects.new("Pi4_floor", me)
sc.collection.objects.link(ob)
me.materials.append(bpy.data.materials["Pi4_projector"])
bpy.ops.wm.save_as_mainfile(filepath=out)
print("SAVED", out, len(V), "verts", len(Fc), "tris")
