# CameraMatch Pi4 — Blender 5.3 camera solve with π⁴

Camera matching for Blender in two stages:

1. **π⁴ (feed-forward, 512 px):** a local [π⁴](https://github.com/yyfz/Pi4) model in Docker predicts
   per-frame camera poses, depth, a point cloud and moving-object masks in one pass.
2. **Plate-resolution refinement:** KLT feature tracks at full resolution (moving objects masked by π⁴),
   3D points initialised from π⁴ (triangulation, falling back to π⁴ depth), then robust bundle adjustment
   of every pose, every point, one focal length and k1/k2 radial distortion (Blender *Polynomial* model).

> **License:** this add-on is GPL-3.0. OneFormer (semantic masks) is MIT-licensed, pulled into the Docker image. The π⁴ code and weights are **CC BY-NC 4.0,
> non-commercial only**. They are pulled into the Docker image, never bundled with the add-on.

## Accuracy (synthetic ground truth, 1920×1080, 120 frames, 24 mm, k1 −0.08 / k2 0.02)
| | π⁴ only | π⁴ + refinement |
|---|---|---|
| Trajectory error (Sim3-aligned) | 0.17 % of path | **0.06 %** |
| Rotation error, median | 0.49° | **0.16°** |
| Focal (GT 1280 px) | 1267 px | **1275.8 px** |
| k1 / k2 | — | **−0.079 / 0.018** |
| Reprojection median / p90 | — | **0.38 / 1.18 px** |

Blender's own Movie Distortion node, fed the GT k1/k2, undistorts the plate to within 0.1 px of the clean
render, so the exported distortion is in Blender's own convention. Run time on an RTX 5090: about 110 s for
120 frames at 1080p (36 s π⁴, about 60 s refinement).

## Layout
```
docker/Dockerfile        CUDA 13 + torch 2.12.1+cu130 (Blackwell / RTX 50xx) + Pi4 @ pinned commit + scipy
docker-compose.yml       service "pi4" on 127.0.0.1:8765, mounts ./work, weights cache volume
server/server.py         HTTP: GET /health, POST /solve (Pi4 [+ refine]), POST /refine (re-run on cache)
server/refine.py         KLT tracking + IRLS bundle adjustment (analytic sparse Jacobian via torch)
blender_addon/cameramatch_pi4/   Blender extension (Movie Clip Editor > Sidebar > Pi4)
tests/                   smoke, end-to-end, synthetic GT (render / distort / eval), Jacobian + distortion checks
dist/                    built extension zip
work/                    shared jobs folder (host <-> container)
```

## Setup
1. Docker Desktop (WSL2 backend) with GPU support. Image, build cache and weights live in Docker's disk
   image (here `D:\DockerData`).
2. `docker compose up -d --build` (first start downloads the weights from Hugging Face `yyfz233/Pi4`; no token needed).
3. `docker compose logs -f pi4` until `model ready`.
4. Blender 5.3 > Edit > Preferences > Get Extensions > ⌄ > Install from Disk > `dist/cameramatch_pi4-0.5.0.zip`.
   In the add-on preferences, set **Work folder** to `<your clone>/work`, the folder docker-compose mounts as
   `/work`. Alternatively, set the `CAMERAMATCH_WORK` environment variable before starting Blender.

## Use
Movie Clip Editor > open a clip (movie file or PNG/JPG sequence) > set the clip's **sensor width**
(Track > Camera; focal in mm depends on it) > Sidebar **Pi4** > **Check Server** > **Solve Camera (Pi4)**.

You get a collection `Pi4_<clip>`:
- animated camera (first frame at origin looking +Y, Z up; clip as background; scene range set),
- `_points`: dense π⁴ point cloud (per-point pixel-footprint radius, *Point Size* on the modifier),
- `_tracks` (refined only): bundle-adjusted sparse 3D track points,
- `<clip>_motion_mask` clip when *Detect Moving Objects* is on,
- the clip's tracking camera gets focal + (refined) **Polynomial k1/k2**. Use a *Movie Distortion* node
  to distort CG renders to match the plate.

Settings: *Refine at Plate Resolution* (default on), *Active Tracks* (600 is the tested sweet spot),
*Zoom Lens (Animate Focal)* (per-frame focal for zoom shots),
*Frame Stride* / *Max Frames* / *Model Resolution* (GPU memory budget), *Animate Focal Length* (π⁴ only).

## RTX 5090 budget (512 px, motion detection on)
| frames | VRAM | Pi4 time |
|---|---|---|
| 120 | 6 GB | 36 s |
| 300 | 12.2 GB | 81 s |
| 750 | ~34 GB (upstream figure): turn motion off or raise stride | — |

## Known limits
- **Metric scale is unreliable:** π⁴'s estimate was 2.5× off on the synthetic scene. Use **Set Scale from
  Distance** (two points of known distance) to fix it.
- Zooms: one focal per frame with a smoothness prior (*Zoom Lens*). k1/k2 stay constant through the zoom,
  the principal point is fixed at the centre, and there is no tangential distortion. On a 1.7× synthetic zoom
  the focal error is 1 % median, but the first frames are less accurate (about 8 % at the start of the zoom).
- More tracks are now cheap (3000 tracks / 225k observations: 53 s of bundle adjustment), but on the shots tested
  they don't lower the error: 600 remains the default.
- Pure nodal pans: depth is unobservable; the solve is held up by the π⁴ priors.
- Video decoded with OpenCV (H.264/H.265/MJPEG fine; ProRes may not be). TIFF/EXR sequences unsupported.
- Upstream π⁴ is very new (released 2026-11-17); pinned to commit `32dfbea`.

## Solver and robustness tools (v0.6.0)
**Bundle adjustment:** `server/ba_lm.py`, a GPU Levenberg-Marquardt solver with Schur complement. The 3D
points are eliminated per point (3×3 blocks), the reduced camera system is solved by dense Cholesky, and
IRLS gives the robust Cauchy/Huber passes. Same model and priors as before; the old scipy TRF/LSMR solver stays
available (`"solver": "scipy"`) and `server/bench_ba.py` compares both on any job.

| Same tracks, RTX 5090 | scipy TRF + LSMR | **LM + Schur** |
|---|---|---|
| Synthetic GT (120 f, 23.5k obs): BA time | 40.6 s | **7.4 s** |
| … trajectory / rotation / focal (GT 1280) | 0.071 % / 0.22° / 1280.8 px | 0.073 % / 0.23° / 1281.8 px |
| Venice (284 f, 41k obs): BA time, median error | 71.7 s, 0.3475 px | **13.3 s, 0.3473 px** |
| Venice, 3000 tracks (225k obs, 8.7M Schur pairs) | too slow | **53 s** |

scipy stopped at its evaluation cap; LM reaches the true minimum of the same objective.

**Zoom shots:** *Zoom Lens (Animate Focal)* solves one focal per frame, with a prior to Pi4's per-frame
focal and a 1 %/frame smoothness prior. The camera lens is keyframed. On a synthetic 24 → 41 mm zoom:

| | Pi4 alone | constant focal | **per-frame focal** |
|---|---|---|---|
| Focal error, median | 4.8 % | 21.6 % | **1.0 %** |
| Trajectory error | — | 2.64 % | **0.78 %** |
| Rotation error | — | 3.53° | **0.45°** |

Test it with `tests/gt_render.py -- <dir> 120 1.7` (zoom factor) + `gt_distort.py`.

**Set Scale from Distance:** select 2 vertices of the point cloud (Edit Mode) or 2 objects, then enter the
real distance. The scale is saved in `<job>/scene_scale.json`, so re-imports and Build Set keep it.

**Export Tracks to Clip:** the refined KLT tracks become Blender tracking markers (`Pi4_####`, full length,
every frame), for inspection or Blender's own solver. On Venice: 1169 tracks, 162k markers, 0.32 px median
from the solved projection.

Checks: `blender -b <solved.blend> --python tests/check_robust_features.py -- [zoom_job first_png gt.json]`.

## Ignore trees / water / sky (v0.8.0, on by default)
Leaves in the wind, waves and clouds are not static, yet they can supply most of the trackable corners. They
then pull the camera solve, and because they are the majority, no outlier rule can reject them. *Ignore
Trees / Water / Sky* runs **OneFormer** (ADE20K, Swin-Tiny, `shi-labs/oneformer_ade20k_swin_tiny`, MIT
license) on every 5th frame. It masks tree, plant, palm, flower, water, sea, river, waterfall, fountain, sky and
flag, dilated by 2 px, and adds the mask to Pi4's moving-object mask, so no track starts there. The cost is about
7 s per solve with the model cached (25 s on the first load).

| Shot (holdout = 15 % of tracks kept out) | masked | median | holdout median / p90 |
|---|---|---|---|
| Park path, phone (distortion off) | 64.5 % (tree 61 %) | 0.858 → **0.401 px** | 1.558 → **0.450** / 6.37 → **1.30 px** |
| ToS balcony at night | 11 % (sky, sea) | 0.361 → **0.321 px** | 0.521 → **0.478** / 3.80 → **1.85 px** |
| Venice alley | 1.4 % (sky) | 0.349 → 0.339 px | 0.415 → 0.392 px |
| Japan crowd / students / synthetic GT | 0–3 % | unchanged | unchanged (within noise) |

On the park path, the ground-grid drift halves (0.161 → 0.073 px/frame, median 13.8 → 5.4 px), and the ground
plane holds 50 % of the points instead of 15 %: with the leaves gone, the solve sees the ground.
When more than half of the frame is masked, the log suggests turning *Solve Lens Distortion* off. Distortion
needs tracks across the whole frame: with the canopy removed and distortion on, k1/k2 ran away to 0.87 / −1.2.

*Rejected:* detecting non-rigid tracks from their residuals, either the temporal structure (smoothed/raw ratio
≈ 0.8 on every shot, rigid or not) or a spatially pooled score. On clean shots the natural residual spread
(3.2× from p10 to p90 on the synthetic GT) is larger than the foliage/ground difference (2×).

## Rolling shutter (v0.7.0)
*Rolling Shutter* (Pi4 tab, refine options) solves the sensor readout time as one global value: a signed
fraction of a frame, where + means the top row is read first. Each observation uses the pose at its row's
capture time, following the camera's angular and linear velocity from the neighbouring frames. Portrait video
is read out across columns (an upright phone), which `rs_axis="auto"` selects. Render & Comp then renders the
CG with the matching Cycles rolling shutter: shutter window = readout + exposure, top row first.

| | global-shutter model | **rolling-shutter model** |
|---|---|---|
| Synthetic handheld GT, readout 0.8: trajectory / rotation error | 0.671 % / 0.651° | **0.117 % / 0.275°** |
| … focal (GT 1280), readout found | 1294 px, — | **1283 px, +0.898** |
| Japan crowd (handheld camera): median, holdout | 0.393 / 0.597 px | **0.362 / 0.555 px** (readout +0.59) |
| Park path (phone, portrait) | 0.860 px | 0.859 px (readout ≈ 0) |

- Test the ground truth with `tests/gt_render_rs.py` (12 sub-frames per frame) followed by `tests/gt_rolling_compose.py`.
- `tests/check_rs_render.py` checks the Cycles side. On the crowd shot, rendered spheres land 1.66 px from
  the tracked features, against 1.72 px for the solver's own projection (2.71 px with Cycles rolling shutter off).
- **The phone walk is not limited by rolling shutter.** Its residuals are 1.2–1.6 px in the tree canopy, which holds
  71 % of the tracks, against 0.65 px on the ground: the leaves move in the wind. The phone's electronic
  stabilisation probably removed most of the rolling shutter. Foliage-heavy shots need region masking or
  weighting, which is not implemented yet.
- Only top-to-bottom row readout can be reproduced in Cycles. Other readouts are solved, but the CG is rendered
  with a global shutter, and the log says so.

## Hard-footage test (2026-11-19)
Five shots, each targeting one weakness, solved with default settings (LM solver, 600 tracks):

| Shot | Source | Weakness | Median / p90 | Holdout median (ratio) | Grid stick (median, drift/frame) |
|---|---|---|---|---|---|
| Venice alley (reference) | Mixkit 4600 | — | 0.35 / 0.92 px | 0.41 px (1.19×) | 1.1 px, 0.011 px |
| Synthetic GT (reference) | gt_render | — | 0.36 / 1.16 px | 0.58 px (1.64×) | — |
| ToS room | ToS 13543–13700 | apparent zoom, people moving | 0.19 / 0.65 px | 0.29 px (1.51×) | 10.7 px, 0.44 px (desk plane) |
| ToS balcony | ToS 1778–1904 | low light, dark foreground | 0.36 / 1.10 px | 0.52 px (1.44×) | 3.3 px, 0.15 px |
| Japan street | Mixkit 4437 | handheld in a dense crowd | 0.39 / 1.13 px | 0.60 px (1.52×) | ground hidden by people |
| Students | Mixkit 4519 | people in front, slow push | 0.18 / 0.55 px | 0.22 px (1.24×) | 4.1 px, 0.054 px |
| Park path (portrait) | Mixkit 40656 | handheld phone walk | **0.86 / 2.44 px** | **1.56 px (1.81×)** | 12.7 px, 0.16 px |

*Holdout:* 15 % of the tracks are kept out of the solve, then triangulated from the solved cameras.
*Grid stick:* a ground-plane grid's intersections tracked with KLT on the ground texture (moving objects
masked) versus their projection (`tools/overlay_check.py`, which also writes an overlay video). This metric
includes plane-fit error: a flat plane on a cluttered desk or a curved path is not the real surface.

What it showed:
- **Every shot solves.** Crowds, darkness and occluders are handled by Pi4's motion masks plus the robust
  bundle adjustment.
- **Handheld phone footage is the weak spot**: two to four times the error of the other shots, most likely from
  rolling shutter and motion blur, which the camera model doesn't include.
- **Zoom vs dolly is ambiguous on low-parallax shots.** On the ToS room, Pi4 sees no zoom, and both
  interpretations reach exactly the same error. Even with the focal prior released (`zoom_anchor="flat"`), the
  tracks choose a dolly with a 5 % zoom. The real optical zoom is proven only on synthetic footage.
- **Focal and distortion trade off on slow push-ins.** On the students shot, with and without k1/k2 give the
  same error but focals 14 % apart (2726 / 3115 px). Use *Lock Focal* / no distortion on such shots.
- **Two diagnostics were tried and rejected, because they are not reliable:**
  - The statistical (Laplace) focal uncertainty is far too optimistic (±0.08 % on the students shot).
  - Choosing the lens model by holdout error picks *no distortion* on the synthetic GT, where it is wrong
    (rotation error 2.6° instead of 0.23°). The holdout measures self-consistency, not correctness.

Tools: `tools/motion_profile.py` (zoom / jitter / parallax profile of a clip, used to find the shots),
`tools/overlay_check.py` (grid overlay + stick metric), `server/bench_ba.py --holdout 0.15`.

## Foreground matte refinement (MatAnyone 2)
`tools/matte_pipeline.py` (on-demand `matte` image: SAM 2.1 Apache-2.0 + [MatAnyone 2](https://huggingface.co/PeiqingYang/MatAnyone2),
NTU S-Lab License 1.0, non-commercial) turns π⁴'s coarse motion mask into a soft, edge-accurate alpha:
1. SAM 2.1 refines π⁴'s mask (box + interior/exterior point prompts) every 30 frames; floating blobs dropped
   (camera-attached rig always touches the frame border).
2. MatAnyone 2 is re-seeded from those masks every 30 frames (crossfaded), so parts entering later (arms) are picked up.
3. Fusion: MatAnyone 2 alpha, with SAM 2 structure (eroded 1 px) filling MatAnyone dropouts on thin / low-contrast
   parts. SAM-only regions are kept only if attached to the matte or near the camera per π⁴ depth.

**In Blender:** Pi4 tab > *Foreground Matte* > **Refine Matte (MatAnyone 2)** (or tick *Refine Matte After Solve*).
It runs on the last solve via the `matte-server` service (port 8766, started by `docker compose up -d`) and loads
`<clip>_matte` (soft alpha clip, scene-aligned) and `<clip>_pi4_invdepth` (Non-Color image sequence, value × 4 = 1/depth)
for holdouts and depth-based occlusion in the compositor. Takes about 50 s for 300 frames.

Command-line comparison of all methods:
```
docker compose --profile tools run --rm matte python /tools/matte_pipeline.py /work/<job> \
    --methods pi4,sam2,matanyone2,matanyone2_seeded,fused        # -> /work/<job>/matte/<method>/alpha_####.png
```
Cycling shot (288×512, 300 frames): boundary on real image edges 46 % (π⁴) → 91–93 % (SAM 2 / MatAnyone 2 / fused);
flicker 27.2 → 18.8; floating false positives 3.5 % → 0.1 % of the matte. About 50 s in total on an RTX 5090.

## Set rebuild: floor + facades + reprojected video (experimental)
```
docker compose --profile tools run --rm matte python /tools/build_scene_geometry.py /work/<job> --source pi4
docker compose --profile tools run --rm matte python /tools/bake_floor_texture.py /work/<job>
blender -b <solved_with_matte.blend> --python tests/build_floor_panels_scene.py -- <scene.blend>
blender -b <scene.blend> --python tests/composite_occlusion.py -- <job>/occlusion <shot_out>   # shot composite
blender -b <scene.blend> --python tests/render_witness.py -- <witness_out> 512 32              # 3rd-person view
```
- Floor: Pi4 depth back-projected through the refined camera (rider removed by the matte), up-facing points
  binned into a gravity-aligned heightfield (steps kept by median filtering).
- Video on the floor: `Pi4_floor_baked` (each texel from the nearest unoccluded frame that saw it, 99 % coverage)
  and `Pi4_projector` (live shot-camera projection with the solved k1/k2, exact in the shot view).
- Facades: segmented vertical-plane RANSAC; every panel is validated against Pi4 depth in the frames that see
  it (panels on reflections / sky are dropped).
- `--source ba` builds from bundle-adjusted track points instead (sparser, coarser stairs).

**Low-poly scene mesh** (smoother surfaces, cleaner grid, walls included):
```
docker compose --profile tools run --rm matte python /tools/build_lowpoly_mesh.py /work/<job>   # ~30 s
blender -b <scene_final.blend> --python tests/use_lowpoly.py -- <scene_lowpoly.blend>
```
Fused Pi4 point cloud (3.1 M pts, rider removed) → voxel + outlier filter → screened Poisson → support trim
(drops Poisson "balloons" over the sky) → quadric decimation to 20 k tris → Taubin smoothing. The gold grid
is drawn only on up-facing faces.

Limitation found on the reframed-360 sample clip: Pi4's per-frame depth is not consistent with a rigid camera
solve (a depth warp between neighbouring frames never beats no-warp, at any translation scale, while the same test
on the synthetic plate locks at 2 grey levels). No surface built from that depth can register tightly on that clip.
Normal camera footage, or a multi-view stereo pass on the solved cameras, is needed for that.

## CG Integration buttons (v0.5.0)
Pi4 tab (Movie Clip Editor) and 3D Viewport > Sidebar > Pi4 > **CG Integration**, on the last solve:
1. **Build Set + Lighting:** the matte server builds `lowpoly.npz` and `plate_env.png` (reused unless *Rebuild
   Mesh*). Blender then creates the `CM_Set` collection:
   - a robust quadratic floor fit, used as a shadow catcher;
   - the plate-projected set mesh, emissive and invisible to the camera;
   - with *Street Walls* on, proxy walls along the two walls detected on either side of the camera's path;
   - the plate environment as the world.

   Every set object is a shadow catcher. The fitted floor is stored on the scene (`cm_floor_coef`).
2. **Place on Floor:** drops the selected objects so their lowest point rests on the fitted floor.
3. **Add Walking Robot:** a procedural robot walking from the **3D cursor** toward the camera over the scene
   range. Feet are planted on the fitted floor, the walk stops 3 m short of the camera, and the robot can look
   at the camera at the end.
4. **Render & Comp:** a Cycles render of the scene range to `<job>/render/r_####.exr` (multilayer, Shadow
   Catcher pass), with a progress bar. The matte server then composites it (`tools/comp_plate.py`) into
   `<job>/comp/`:
   - `comp_####.png`
   - `<clip>_comp.mp4`
   - `<clip>_plate_vs_comp.mp4`

   The comp loads back as the clip `<clip>_comp`.

UI test: `blender solved.blend --python tests/gui_pipeline.py -- out.blend <cursor_x> <cursor_y> [from to samples]`.
On the Venice shot the full chain (set, robot, 31 frames at 64 samples, comp, encode) takes about 3 minutes.

## CG character in a street: walking robot (example)

Shot: Mixkit #4600 "Narrow and old alley in Venice" (Mixkit free license), slow forward dolly, 284 frames @ 25 fps.
Solve: *Solve Lens Distortion* off, *Lock Focal to Pi4* on -> 0.35 px median, f = 28.8 mm.

1. `tools/build_lowpoly_mesh.py` + `tools/build_plate_envmap.py` on the job (no matte needed for an empty street).
2. `blender -b solved.blend --python tests/build_robot_scene.py -- robot.blend [y_start]`
   - floor: robust quadratic fit to the Pi4 floor points (0.5 cm) -> shadow catcher;
   - walls: Pi4 low-poly mesh + 14 m proxy walls along the fitted street edges, emitting the projected plate,
     **flagged as shadow catchers** (they are real-world geometry: otherwise Cycles counts their occlusion
     of the sky as a CG shadow and darkens the whole floor);
   - lit only by the plate environment through that set: sky slit above + wall bounce, no added lights;
   - `tests/robot_rig.py`: rigid-panel humanoid, planted-foot walk (2-bone IK, heel strike / toe off,
     pelvis bob/sway, arm swing, calf pistons), head turns to the camera at the end.
3. `blender -b robot.blend --python tests/render_frames.py -- <exr_dir> all 192` (multilayer EXR:
   Combined + Shadow Catcher).
4. `python /tools/comp_plate.py plate.mp4 <exr_dir> <out_dir>` (matte container): plate x shadow pass, CG
   black level / softness / grain measured from the plate, light wrap.

## Tests
```
blender -b --factory-startup --python tests/blender_smoke.py                    # import/axes, no server
blender -b --factory-startup --python tests/blender_e2e.py -- <video|first_png>  # full chain
# Synthetic ground truth:
blender -b --factory-startup --python tests/gt_render.py -- <abs>/work/gt_scene 120
docker compose exec pi4 python /path/gt_distort.py /work/gt_scene -0.08 0.02   # copy script in first
# solve with job input "../gt_scene/plate", then: python gt_eval.py /work/gt_scene /work/<job>
blender -b --factory-startup --python tests/check_blender_distortion.py -- <abs>/work/gt_scene 60
# CG insertion check on a solved shot (real UI operator, then cubes + occlusion-aware composite):
blender --python tests/gui_solve.py -- <video> <solved.blend> [sensor_mm]
blender -b <solved.blend> --python tests/insert_geometry.py -- <out_dir>
docker compose exec pi4 python /path/occlusion_maps.py /work/<job> <W> <H> 1      # Pi4 inv-depth + fg mattes
blender -b <out_dir>/cycling_with_geometry.blend --python tests/composite_occlusion.py -- <job>/occlusion <out_dir>
```
The occlusion composite lives in the saved .blend's compositor: lens-distorted CG, then a soft depth test of
premultiplied inverse depth against π⁴'s inverse depth (12 % tolerance), times (1 − π⁴ foreground matte),
laid over the clip. Cycles shadow catchers under the CG.
