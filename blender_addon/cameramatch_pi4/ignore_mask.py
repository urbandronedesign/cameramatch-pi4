# SPDX-License-Identifier: GPL-3.0-or-later
"""Rasterise a Blender Mask (drawn in the Movie Clip Editor) into a per-frame image sequence for the solver.

White = never track there. Evaluated by Blender's own compositor Mask node, so animated splines, feather and
splines parented to tracks behave exactly as in the editor. Written to <job>/ignore_mask/m_####.png
(#### = scene frame) + ignore_mask.json with the clip frame offset; the server merges it with Pi4's motion
mask (and the semantic mask) before tracking.
"""
import json
import os

import bpy

LONG_SIDE = 640


def rasterize(mask, clip, scene, job_dir, log=print):
    out = os.path.join(job_dir, "ignore_mask")
    os.makedirs(out, exist_ok=True)
    W, H = clip.size
    s = LONG_SIDE / max(W, H)
    tmp = bpy.data.scenes.new("CM_mask_raster")
    try:
        tmp.render.engine = "BLENDER_WORKBENCH"
        tmp.render.resolution_x, tmp.render.resolution_y = max(16, round(W * s)), max(16, round(H * s))
        tmp.render.resolution_percentage = 100
        first = clip.frame_start
        tmp.frame_start, tmp.frame_end = first, first + clip.frame_duration - 1
        tmp.render.fps, tmp.render.fps_base = scene.render.fps, scene.render.fps_base
        cam = bpy.data.objects.new("CM_mask_cam", bpy.data.cameras.new("CM_mask_cam"))
        tmp.collection.objects.link(cam)
        tmp.camera = cam
        ng = bpy.data.node_groups.new("CM_mask_raster", "CompositorNodeTree")
        ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        mn = ng.nodes.new("CompositorNodeMask")
        mn.mask = mask
        try:
            mn.inputs["Feather"].default_value = True
        except (KeyError, TypeError):
            pass
        go = ng.nodes.new("NodeGroupOutput")
        ng.links.new(mn.outputs["Mask"], go.inputs[0])
        tmp.compositing_node_group = ng
        tmp.render.use_compositing = True
        tmp.render.use_sequencer = False
        im = tmp.render.image_settings
        if hasattr(im, "media_type"):
            im.media_type = "IMAGE"
        im.file_format = "PNG"
        im.color_mode = "BW"
        im.color_depth = "8"
        tmp.render.filepath = os.path.join(out, "m_")
        with bpy.context.temp_override(scene=tmp):
            bpy.ops.render.render(animation=True, scene=tmp.name)
        with open(os.path.join(job_dir, "ignore_mask.json"), "w") as fh:
            json.dump({"pattern": "ignore_mask/m_{:04d}.png", "first_scene_frame": first,
                       "mask": mask.name, "size": [tmp.render.resolution_x, tmp.render.resolution_y]}, fh)
        log(f"ignore mask '{mask.name}': {clip.frame_duration} frames rasterised to {out}")
    finally:
        ng_ = bpy.data.node_groups.get("CM_mask_raster")
        cam_ = bpy.data.objects.get("CM_mask_cam")
        bpy.data.scenes.remove(tmp)
        if cam_:
            data = cam_.data
            bpy.data.objects.remove(cam_)
            bpy.data.cameras.remove(data)
        if ng_:
            bpy.data.node_groups.remove(ng_)
    return out
