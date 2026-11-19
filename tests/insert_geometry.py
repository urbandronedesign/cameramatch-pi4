"""Drop CG cubes onto the solved ground and composite them over the plate (with lens distortion).

blender -b <solved.blend> --python tests/insert_geometry.py -- <out_dir>
For several reference frames: take refined track points in the lower part of the view,
RANSAC a ground plane (normal ~ camera up), intersect the ray through (0.5W, 0.6H) with it and
place a cube there (size ~6% of its distance). CG is rendered undistorted, distorted with the
solved Polynomial k1/k2 (Movie Distortion), and alpha-overed on the movie clip.
"""
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

out_dir = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
os.makedirs(out_dir, exist_ok=True)
scene = bpy.context.scene
cam = scene.camera
clip = next(c for c in bpy.data.movieclips if not c.name.endswith("_motion_mask"))
W, H = clip.size
f = cam.data.lens / cam.data.sensor_width * W
tr = next(o for o in bpy.data.objects if o.name.endswith("_tracks"))
P = np.zeros(len(tr.data.vertices) * 3)
tr.data.vertices.foreach_get("co", P)
P = P.reshape(-1, 3)
rng = np.random.default_rng(0)
print("tracks", len(P), "plate", W, H, "f_px", round(f, 1))

colors = [(1, 0.1, 0.05, 1), (0.1, 0.9, 0.2, 1), (0.1, 0.4, 1, 1), (1, 0.85, 0.1, 1), (0.9, 0.2, 0.9, 1),
          (0.1, 0.9, 0.9, 1)]
placed = []
for n, fr in enumerate(range(scene.frame_start + 20, scene.frame_end - 10, 50)):
    scene.frame_set(fr)
    Mw = np.array(cam.matrix_world)
    Rcw, C = Mw[:3, :3], Mw[:3, 3]
    Xc = (P - C) @ Rcw  # camera coords (Blender: x right, y up, -z forward)
    z = -Xc[:, 2]
    u = f * Xc[:, 0] / np.maximum(z, 1e-6) + W / 2
    v = -f * Xc[:, 1] / np.maximum(z, 1e-6) + H / 2
    sel = (z > 0) & (u > 0.1 * W) & (u < 0.9 * W) & (v > 0.45 * H) & (v < 0.95 * H)
    pts = P[sel]
    up = Rcw[:, 1]
    if len(pts) < 15:
        print("frame", fr, "skip: only", len(pts), "ground candidates")
        continue
    thr = 0.03 * float(np.median(z[sel]))
    best = None
    for _ in range(400):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        if np.linalg.norm(nrm) < 1e-9:
            continue
        nrm /= np.linalg.norm(nrm)
        if nrm @ up < 0:
            nrm = -nrm
        if nrm @ up < 0.6:
            continue
        inl = np.abs((pts - a) @ nrm) < thr
        if best is None or inl.sum() > best[0].sum():
            best = (inl, nrm, a)
    if best is None or best[0].sum() < 10:
        print("frame", fr, "skip: no ground plane")
        continue
    q = pts[best[0]]
    cen = q.mean(0)
    nrm = np.linalg.svd(q - cen)[2][2]
    nrm = nrm if nrm @ up > 0 else -nrm
    # Anchor on the tracked ground inlier closest to image position (0.5W, 0.6H), snapped to the plane.
    qu, qv = u[sel][best[0]], v[sel][best[0]]
    k = int(np.argmin(np.hypot(qu - 0.5 * W, qv - 0.6 * H)))
    pos = q[k] - ((q[k] - cen) @ nrm) * nrm
    t = float(np.linalg.norm(pos - C))
    size = 0.06 * t
    xax = Rcw[:, 0] - (Rcw[:, 0] @ nrm) * nrm
    xax /= np.linalg.norm(xax)
    yax = np.cross(nrm, xax)
    bpy.ops.mesh.primitive_cube_add(size=size)
    cube = bpy.context.object
    cube.name = f"CG_check_{fr:04d}"
    cube.matrix_world = Matrix([[*xax, 0], [*yax, 0], [*nrm, 0], [0, 0, 0, 1]]).transposed()
    cube.matrix_world.translation = Vector(pos + nrm * size / 2)
    mat = bpy.data.materials.new(cube.name)
    if mat.node_tree is None:
        mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = colors[n % len(colors)]
    mat.diffuse_color = colors[n % len(colors)]
    cube.data.materials.append(mat)
    placed.append((fr, round(float(t), 2), int(best[0].sum()), int(len(pts))))
    print(f"frame {fr}: cube at dist {t:.2f}, plane inliers {best[0].sum()}/{len(pts)}, "
          f"normal.up {nrm @ up:.3f}")

bpy.ops.object.light_add(type="SUN", rotation=(0.8, 0.2, 0.5))
bpy.context.object.data.energy = 3.0

# Compositor: undistorted CG -> Movie Distortion (distort) -> over the plate.
scene.render.film_transparent = True
ng = bpy.data.node_groups.new("Pi4 check comp", "CompositorNodeTree")
scene.compositing_node_group = ng
ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
rl = ng.nodes.new("CompositorNodeRLayers")
mc = ng.nodes.new("CompositorNodeMovieClip")
mc.clip = clip
md = ng.nodes.new("CompositorNodeMovieDistortion")
md.clip = clip
try:
    md.distortion_type = "DISTORT"
except AttributeError:  # Blender 5.x: menu socket
    md.inputs["Type"].default_value = "Distort"
ao = ng.nodes.new("CompositorNodeAlphaOver")
out = ng.nodes.new("NodeGroupOutput")
ins = {s.name: s for s in ao.inputs if s.type == "RGBA"}
print("alpha over inputs", list(ins))
bg = ins.get("Background") or [s for s in ao.inputs if s.type == "RGBA"][0]
fg = ins.get("Foreground") or [s for s in ao.inputs if s.type == "RGBA"][1]
ng.links.new(rl.outputs["Image"], md.inputs["Image"])
ng.links.new(mc.outputs["Image"], bg)
ng.links.new(md.outputs["Image"], fg)
ng.links.new(ao.outputs["Image"], out.inputs[0])
scene.render.use_compositing = True
scene.view_settings.view_transform = "Standard"  # keep the plate's colours untouched
for o in bpy.data.objects:  # helper geometry stays in the viewport, not in the check render
    if o.name.endswith(("_points", "_tracks")):
        o.hide_render = True
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = os.path.join(out_dir, "comp_")
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "cycling_with_geometry.blend"))
print("PLACED", placed)
bpy.ops.render.render(animation=True)
print("RENDERED", scene.frame_start, scene.frame_end)
