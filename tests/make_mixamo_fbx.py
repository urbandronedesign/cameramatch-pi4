"""Mixamo-style FBX test files from a Mixamo character in glTF (three.js Xbot.glb: mixamorig rig + 'walk').

blender -b --factory-startup --python tests/make_mixamo_fbx.py -- <Xbot.glb> <out_dir>
Writes, like a Mixamo download (one character, one action, armature scale 0.01):
  xbot_walk_inplace.fbx     the walk 'In Place'
  xbot_walk_rootmotion.fbx  the same walk with the hips travelling forward (In Place unticked)
"""
import importlib
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = os.path.abspath(argv[0]), os.path.abspath(argv[1])
cm = importlib.import_module("bl_ext.user_default.cameramatch_pi4")
C = cm.characters
os.makedirs(out, exist_ok=True)


def load():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    walk = bpy.data.actions["walk"]
    for a in list(bpy.data.actions):
        if a is not walk:
            bpy.data.actions.remove(a)
    arm.animation_data.action = walk
    if getattr(walk, "slots", None):
        arm.animation_data.action_slot = walk.slots[0]
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 0, int(walk.frame_range[1])
    return sc, arm, walk


def export(path):
    bpy.ops.export_scene.fbx(filepath=path, use_selection=False, object_types={"ARMATURE", "MESH"},
                             add_leaf_bones=False, bake_anim=True, bake_anim_use_all_actions=False,
                             bake_anim_use_nla_strips=False, bake_anim_force_startend_keying=True)
    print("[mkfbx] wrote", path, flush=True)


# 1. in place
sc, arm, walk = load()
export(os.path.join(out, "xbot_walk_inplace.fbx"))

# 2. root motion: the hips travel forward by the cycle's own walk distance
sc, arm, walk = load()
info = C.analyse_walk(sc, arm, arm, walk)
dist = info["speed_mpf"] * info["cycle"]                  # metres per cycle
fwd_w = Vector((info["forward"][0], info["forward"][1], 0.0)) * dist
hips = info["hips"]
B = arm.matrix_world @ arm.data.bones[hips].matrix_local   # bone rest space -> world (includes the 0.01 scale)
d_loc = B.to_3x3().inverted() @ fwd_w                       # the same displacement in the hips' location space
path = f'pose.bones["{hips}"].location'
fcs = {fc.array_index: fc for fc in C._fcurves(walk) if fc.data_path == path}
for i, fc in fcs.items():
    kp = fc.keyframe_points
    t0, t1 = kp[0].co.x, kp[-1].co.x
    for k in kp:
        u = (k.co.x - t0) / (t1 - t0)
        k.co.y += d_loc[i] * u
        k.handle_left.y += d_loc[i] * u
        k.handle_right.y += d_loc[i] * u
    fc.update()
check = C.analyse_walk(sc, arm, arm, walk)
print(f"[mkfbx] root motion added: hips travel {check['hip_travel_m']:.2f} m per cycle "
      f"(walk distance {dist:.2f} m)", flush=True)
export(os.path.join(out, "xbot_walk_rootmotion.fbx"))
