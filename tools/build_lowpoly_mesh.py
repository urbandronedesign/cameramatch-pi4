"""Low-poly scene mesh from the fused point cloud (Pi4 depth through the refined camera).

python /tools/build_lowpoly_mesh.py /work/<job> [--tris 20000] [--voxel 0.06] [--depth 10]
1. Fuse static points of every 2nd frame (rider removed by the fused matte, depth edges dropped),
   with normals (from the depth map) and colours, in Blender world coordinates.
2. Voxel downsample + statistical outlier removal (averages per-frame depth noise).
3. Screened Poisson reconstruction, low-density trim, corridor crop around the camera path.
4. Quadric decimation to --tris triangles + Taubin smoothing -> low-poly mesh.
Writes /work/<job>/lowpoly.npz (verts, faces, colors, up) and lowpoly.ply.
"""
import argparse
import os

import cv2
import numpy as np
import open3d as o3d

CV_TO_BL_WORLD = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--tris", type=int, default=20000)
    ap.add_argument("--voxel", type=float, default=0.06)
    ap.add_argument("--depth", type=int, default=10, help="Poisson octree depth")
    ap.add_argument("--max_depth", type=float, default=25.0, help="ignore points farther than this (units)")
    ap.add_argument("--corridor", type=float, default=18.0, help="keep mesh within this distance of the path")
    ap.add_argument("--frame_step", type=int, default=2)
    ap.add_argument("--px_step", type=int, default=2)
    a = ap.parse_args()

    r = np.load(os.path.join(a.job, "result.npz"))
    with np.load(os.path.join(a.job, "pi4_cache.npz")) as z:
        depth_all, src = z["depth"].astype(np.float32), str(z["src"])
    c2w = r["c2w_refined"]
    T = len(c2w)
    W, H = (int(v) for v in r["plate_size"])
    f, k1, k2 = float(r["f_px"]), float(r["k1"]), float(r["k2"])
    cx, cy = (float(v) for v in r["principal_px"])
    idx = r["frame_indices"].astype(int)
    anchor = CV_TO_BL_WORLD @ np.linalg.inv(c2w[0])
    centers = np.array([(anchor @ c2w[i])[:3, 3] for i in range(T)])
    up = np.mean([anchor[:3, :3] @ (c2w[i, :3, :3] @ np.array([0, -1.0, 0])) for i in range(T)], 0)
    up /= np.linalg.norm(up)

    s = a.px_step
    vv, uu = np.mgrid[0:H:s, 0:W:s].astype(np.float64)
    xd, yd = (uu - cx) / f, (vv - cy) / f
    xn, yn = xd.copy(), yd.copy()
    for _ in range(30):
        q = xn * xn + yn * yn
        dq = 1 + k1 * q + k2 * q * q
        xn, yn = xd / dq, yd / dq

    cap = cv2.VideoCapture(os.path.join(a.job, src))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    mdir = os.path.join(a.job, "matte", "fused")
    P, N, Cc = [], [], []
    for i in range(0, T, a.frame_step):
        Z = cv2.resize(depth_all[i], (W, H), interpolation=cv2.INTER_LINEAR)[::s, ::s].astype(np.float64)
        mp = os.path.join(mdir, f"alpha_{idx[i] + 1:04d}.png")
        al = cv2.imread(mp, 0) if os.path.isfile(mp) else None   # no matte (empty street): keep everything
        fg = cv2.dilate(al, np.ones((5, 5), np.uint8))[::s, ::s] > 40 if al is not None else np.zeros_like(Z, bool)
        Xc = np.stack([xn * Z, yn * Z, Z], -1)
        du, dv = np.gradient(Xc, axis=1), np.gradient(Xc, axis=0)
        n = np.cross(du, dv)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
        n[(n * Xc).sum(-1) > 0] *= -1                      # face the camera
        gz = np.hypot(*np.gradient(Z)) / np.maximum(Z, 1e-6)
        ok = (~fg) & (Z > 0.2) & (Z < a.max_depth) & (gz < 0.06)
        M = anchor @ c2w[i]
        P.append(Xc[ok] @ M[:3, :3].T + M[:3, 3])
        N.append(n[ok] @ M[:3, :3].T)
        Cc.append(frames[idx[i]][::s, ::s][ok].astype(np.float64) / 255)
    P, N, Cc = np.concatenate(P), np.concatenate(N), np.concatenate(Cc)
    print(f"fused points: {len(P):,}")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd.normals = o3d.utility.Vector3dVector(N)
    pcd.colors = o3d.utility.Vector3dVector(Cc)
    pcd = pcd.voxel_down_sample(a.voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=1.8)
    print(f"after voxel {a.voxel} + outlier removal: {len(pcd.points):,}")

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=a.depth, scale=1.1,
                                                                           linear_fit=True)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.12))
    # Support trim: Poisson closes the surface over sky/open space ("balloons"). Keep only vertices
    # within a few voxels of a measured point.
    tree = o3d.geometry.KDTreeFlann(pcd)
    V = np.asarray(mesh.vertices)
    far = np.zeros(len(V), bool)
    lim2 = (2.5 * a.voxel) ** 2
    for vi in range(len(V)):
        _, _, d2 = tree.search_knn_vector_3d(V[vi], 1)
        far[vi] = d2[0] > lim2
    mesh.remove_vertices_by_mask(far)
    print(f"support trim removed {far.mean() * 100:.1f}% of Poisson vertices")
    V = np.asarray(mesh.vertices)
    # corridor crop around the camera path
    dmin = np.full(len(V), np.inf)
    for k in range(0, T, 3):
        dmin = np.minimum(dmin, np.linalg.norm(V - centers[k], axis=1))
    mesh.remove_vertices_by_mask(dmin > a.corridor)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    print(f"poisson mesh: {len(mesh.triangles):,} tris")

    low = mesh.simplify_quadric_decimation(target_number_of_triangles=a.tris)
    low = low.filter_smooth_taubin(number_of_iterations=8)
    low.remove_degenerate_triangles()
    low.remove_duplicated_vertices()
    low.remove_non_manifold_edges()
    low.compute_vertex_normals()
    print(f"low-poly mesh: {len(low.vertices):,} verts, {len(low.triangles):,} tris")
    o3d.io.write_triangle_mesh(os.path.join(a.job, "lowpoly.ply"), low)
    np.savez(os.path.join(a.job, "lowpoly.npz"), verts=np.asarray(low.vertices).astype(np.float32),
             faces=np.asarray(low.triangles).astype(np.int32),
             colors=np.asarray(low.vertex_colors).astype(np.float32), up=up)


if __name__ == "__main__":
    main()
