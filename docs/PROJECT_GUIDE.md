# CameraMatch Pi4: project guide

This is the working reference for the project: what was built, how to run each step, what the shots
showed, and what went wrong along the way. README.md documents the add-on; this file covers everything
around it.

Last updated 2026-11-19.

---

## 1. What this is

CameraMatch Pi4 is a **Blender 5.3 extension** that solves the camera of a video with π⁴
([yyfz/Pi4](https://github.com/yyfz/Pi4)), a feed-forward 4D geometry model. A plate-resolution
bundle adjustment then refines that solve. Around it there is a set of tools for putting CG into the
solved shot:
- foreground mattes,
- occlusion,
- set geometry rebuilt from the point cloud,
- environment lighting taken from the plate,
- a physically lit composite.

**License constraint:** use is **non-commercial only**. The π⁴ code and weights are CC BY-NC 4.0,
MatAnyone 2 is under the NTU S-Lab License 1.0 (non-commercial), and SAM 2.1 is Apache-2.0. The
add-on code itself is GPL-3.0. Model weights live in Docker volumes and are never bundled with the
add-on.

**Machine:** Windows 11, RTX 5090 (32 GB, Blackwell sm_120), Docker Desktop on WSL2.
**Docker data lives on `D:\DockerData`. Never put Docker data on C:.**

---

## 2. Architecture

```
Blender 5.3 (add-on "cameramatch_pi4", Movie Clip Editor > Sidebar > Pi4)
   |  HTTP on localhost, job folders shared through ./work
   v
pi4 container        127.0.0.1:8765   POST /solve, POST /refine, GET /progress, GET /health
matte-server         127.0.0.1:8766   POST /matte, POST /set, POST /comp, GET /progress, GET /health
matte (tools)        on demand        docker compose --profile tools run --rm matte python /tools/...
```

| Path | Role |
|---|---|
| `docker/Dockerfile` | `cameramatch-pi4`: CUDA 13, torch 2.12.1+cu130, Pi4 pinned to commit `32dfbea`, scipy |
| `docker/Dockerfile.matte` | `cameramatch-matte`: SAM 2.1, MatAnyone 2, OpenCV, scipy, Open3D, OpenEXR |
| `server/server.py` | Pi4 solve and refine jobs; writes `result.npz` and `pi4_cache.npz` |
| `server/refine.py` | KLT tracks, point initialisation, IRLS driver; legacy scipy TRF+LSMR solver (class `BA`) |
| `server/ba_lm.py` | Default solver: GPU Levenberg-Marquardt with Schur complement, optional per-frame focal (zoom) |
| `server/bench_ba.py` | Compares the solvers on any job, with ground-truth metrics |
| `server/progress.py` | Per-job stages, weights and log, served at `/progress` (copied as `tools/progress.py`) |
| `blender_addon/cameramatch_pi4/` | The extension (v0.5.0): `__init__.py` (UI, operators, progress), `set_builder.py` (floor fit, set, render setup), `robot.py` (rig, walk, materials). Installed as a copy, see §9 |
| `tools/` | Host-side pipeline scripts (matte, mesh, environment map, composite) |
| `tests/` | Blender scene builders, render drivers, validation scripts |
| `work/` | Job folders (`<clip>_<date>_<time>/`) and per-shot outputs |
| `samples/` | Source clips. The full Tears of Steel film is in `samples/tos/` |

**Robustness tools (v0.6.0):**
- **Set Scale from Distance:** writes `<job>/scene_scale.json`. The import records the scale it actually
  applied on the scene (`cm_applied_scale`), and Set Scale computes from that, not from the file.
- **Export Tracks to Clip:** writes `Pi4_####` tracks from `track_obs_full`.

**Contents of a job folder** (`work/<clip>_<timestamp>/`):
- `input/`: the source clip.
- `result.npz`: poses, focal, k1/k2, tracks and points; the refined solve when refinement ran.
- `pi4_cache.npz`: depth × metric scale (float16), dynamic mask, source frame indices.
- `matte/fused/alpha_####.png` and `occlusion/{invdepth,fgmask}_####.png`: written by Refine Matte.
- `lowpoly.npz` / `.ply`, `plate_env.png`, `plate_clean.png`: written by the tools.

---

## 3. Everyday commands

```bash
# services
docker compose up -d                      # pi4 + matte-server
docker compose logs -f pi4                # wait for "model ready"
# Blender
"C:/Program Files/Blender Foundation/Blender 5.3/blender.exe"
# a tool in the matte image (Git Bash: MSYS_NO_PATHCONV=1 stops path mangling)
MSYS_NO_PATHCONV=1 docker compose run --rm -v "$(pwd -W)/work:/work" -v "$(pwd -W)/samples:/samples" \
    --entrypoint python matte /tools/<script>.py ...
```

Neither ffmpeg nor OpenCV is installed on the Windows host. Run them inside the matte container.

---

## 4. The add-on

**Buttons:**
- **Check Server**
- **Solve Camera (Pi4)**: a modal operator with a progress bar and an ETA
- **Re-import**
- **Refine Matte (MatAnyone 2)**
- **Show Log**: opens the `CameraMatch_Log` text block in its own window
- **Clear Log**

The progress bar also appears in the status bar.

**CG Integration section** (v0.5.0), also in 3D Viewport > Sidebar > Pi4:
- **Build Set + Lighting:** creates the `CM_Set` collection (floor catcher, projected walls, street proxy walls)
  plus the plate environment world.
- **Place on Floor**
- **Add Walking Robot:** starts from the 3D cursor.
- **Render & Comp:** writes the EXRs to `<job>/render`, then the comp and MP4s to `<job>/comp`.

See README, "CG Integration buttons". The robot pipeline in §5.4 is what these buttons automate. Its
street walls are detected along the camera's travel direction, with no fixed +Y assumption.

**Key settings:**

| Setting | Default | Notes |
|---|---|---|
| Refine at Plate Resolution | on | KLT plus bundle adjustment |
| Solve Lens Distortion | on | **Turn off for slow or long-lens shots**: k1/k2 is poorly constrained and drags the focal |
| Lock Focal to Pi4 | off | **Turn on for slow cinema shots** (sets focal_sigma to 0.002) |
| Zoom Lens (Animate Focal) | off | Per-frame focal with a smoothness prior; the lens is keyframed |
| Active Tracks | 600 | Tested sweet spot. More tracks make the solver slower without improving accuracy |
| Max Frames / Stride / Model Resolution | 300 / 1 / 512 | GPU budget: 300 frames ≈ 12 GB, about 80 s |
| Refine Matte After Solve | off | Runs the matte server automatically after a solve |

**What gets imported** into a `Pi4_<clip>` collection:
- the animated camera, with the first frame at the origin looking down +Y, Z up;
- `_points`, the dense point cloud;
- `_tracks`, the bundle-adjusted track points;
- a motion-mask clip;
- the clip's tracking camera, set to the solved focal and Polynomial k1/k2.

After Refine Matte you also get a `<clip>_matte` clip and a `<clip>_pi4_invdepth` sequence
(Non-Color; value × 4 = 1/depth).

The scene frame rate is set from the clip (fixed 2026-11-19; earlier versions left 25 fps clips at 24).

Headless drivers:
- `tests/gui_solve.py`: runs the real modal operator. Set the environment variables `CM_MAX_FRAMES`,
  `CM_SOLVE_DISTORTION=0/1` and `CM_LOCK_FOCAL=0/1`.
- `tests/gui_matte.py`
- `tests/reimport_solve.py`

---

## 5. Pipelines

### 5.1 Solve
π⁴ runs at 512 px, about 0.27 s per frame. KLT then tracks at full resolution with moving objects
masked, points are initialised by midpoint triangulation, and IRLS bundle adjustment (Huber, then
Cauchy) solves every pose, every point, the focal and k1/k2. Each track is capped at 40
observations.

Benchmarks (LM vs scipy, zoom) are in README, "Solver and robustness tools". LM: 5–6× faster for the
same minimum; 225k observations in under a minute.

### 5.2 Matte (`tools/matte_pipeline.py`, via matte-server)
SAM 2.1 seeds every 30 frames, MatAnyone 2 is re-seeded in chunks, and the two are fused. The mode
is chosen automatically:
- **rig mode**: blobs must touch the frame border. Used for a camera-attached rider.
- **scene mode**: blobs must be supported by Pi4's dynamic mask. Used for people in the scene.

### 5.3 Set geometry
- `tools/build_lowpoly_mesh.py`: fuses Pi4 depth through the refined cameras (matte excluded when one
  exists), then runs voxel filtering → Poisson → support trim → corridor crop → quadric decimation →
  Taubin smoothing.
- `tools/build_plate_envmap.py`: builds `plate_env.png`. When the projection coverage is under 10 %
  (narrow lenses) it switches to **stretched-plate** mode: a sky band from the brightest 8 % of the
  upper half, with the plate mirrored behind the camera.
- `tools/build_scene_geometry.py` and `bake_floor_texture.py`: the older heightfield floor, façade
  RANSAC and baked floor texture.

### 5.4 CG insertion recipes (one scene builder per look)

| Recipe | Builder | Composite |
|---|---|---|
| Check cubes with occlusion | `tests/insert_geometry.py` | `tests/composite_occlusion.py` (Pi4 inverse-depth test, or occluder-plane mode) |
| Floor, video reprojection, gold panels | `tests/build_floor_panels_scene.py`, `tests/use_lowpoly.py` | `tests/composite_occlusion.py` |
| Rising chrome/gold spheres behind people | `tests/build_spheres_scene.py` | `tests/composite_occlusion.py` (occluder-plane mode, matte holdout) |
| **Walking robot in a street** | add-on buttons (or the legacy `tests/build_robot_scene.py`) | Render & Comp button (or `tests/render_frames.py` → `tools/comp_plate.py`) |

**Robot pipeline in detail** (the most physically complete recipe):
1. **Floor:** a robust quadratic fit to the Pi4 floor points, used as the shadow catcher. Its material
   is the projected plate times 1.6, so it bounces light onto the robot.
2. **Walls:** the Pi4 low-poly mesh (floor faces removed) plus **14 m proxy walls** along the fitted
   street edges. Both emit the projected plate, are invisible to the camera, and are **flagged as
   shadow catchers**.
3. **Lighting:** the plate environment through that set, with no added lights.
4. **Rig:**
   - rigid panels keyframed every frame, with linear interpolation;
   - planted feet through 2-bone IK;
   - heel strike and toe off;
   - pelvis bob, sway and yaw;
   - arm swing;
   - calf pistons;
   - the head turns to look at the camera at about 62 % of the shot.

   `animate_walk` returns IK statistics; keep `max_ratio` below 0.97.
5. **Render:** Cycles on OptiX, 192 samples, motion blur with shutter 0.5, Standard view transform,
   multilayer EXR with Combined and Shadow Catcher passes.
6. **Composite** (`tools/comp_robot.py`): plate × shadow pass, then the CG with its black level
   matched to the plate (measured), a 0.6 px blur, light wrap and grain measured from flat areas of
   the plate.

---

## 6. Shots and results

| Shot | Source | Solve | Deliverables |
|---|---|---|---|
| Synthetic GT | `tests/gt_render.py` | trajectory 0.06 %, rotation 0.16°, f 1275.8/1280 px, k1 −0.079/−0.08 | validation only |
| Cycling (reframed 360) | `samples/cycling.mp4` | solves, but Pi4 depth is **inconsistent with a rigid solve** | cubes, floor and gold panels, low-poly mesh (`work/floor_panels`, `work/insert_test`) |
| Tears of Steel bridge | `samples/tos_bridge.mp4` (film frames 600–965) | 0.33 px median, f = 3623 px; distortion off, focal locked | `work/tos_test/bridge_spheres_final.mp4`, `bridge_spheres_plate_vs_comp.mp4`, `spheres.blend` |
| Venice alley | `samples/venice_alley.mp4` (Mixkit #4600) | 0.35 px median, f = 28.8 mm; distortion off, focal locked | `work/venice/venice_robot_final.mp4`, `venice_robot_plate_vs_comp.mp4`, `robot.blend` |

Credits:
- Tears of Steel: © Blender Foundation, mango.blender.org, CC BY 3.0.
- Venice alley: Mixkit free license.

**Choosing a shot:** `tools/find_shots.py` finds cuts and measures motion. Good candidates are slow,
steady, have real parallax and contain no existing VFX; Tears of Steel street shots already contain
the film's own CG. A forward dolly down a street is ideal for a walking character.

---

## 7. Hard-won lessons (read before changing things)

### Compositing and lighting
- **Every real-world proxy in a Cycles scene must be a shadow catcher.** Otherwise its occlusion of
  the sky counts as a CG shadow. In the robot scene, walls without the flag darkened the whole floor to
  about 20 %. Check: the shadow-pass median must be about 1.0 away from the CG.
- **A scanned set ends where the camera stopped seeing it.** Above the frame top and behind the
  camera, open sky leaks in and flattens the lighting. Close the set with proxy geometry.
- **Two coincident surfaces** (a reflection mesh and a catcher) block light to each other. Keep one, or
  split them by face.
- **Catchers act as holdouts for CG behind them.** Restrict them to ground faces (`|n·up| ≥ 0.75` and
  well below the camera) when CG floats through foliage.
- **A relative depth test** gives partial opacity on noisy backgrounds. Use occluder-plane mode
  (`cm_occluder_depth`) plus the matte for the characters.
- **The first movie-clip or movie-texture render in a background session can come back black.** Do a
  full-size warm-up render first; both render drivers already do.
- **Keep it physical:** the user's preference. No fake shadows or lights; if a shadow lands where the
  camera can't see it, say so rather than cheat.

### Solve
- On slow or long-lens shots, solving distortion lets k2 run away (1.21 on the bridge) and the focal
  drift. Turn off distortion and lock the focal.
- **Pi4's metric scale is unreliable** (2.5× off on the GT scene, though plausible on real streets).
  Sanity-check against something of known size, such as doors or paving.
- scipy's robust losses stall on this bundle adjustment; IRLS around a linear loss works. Thinning
  tracks took one solve from 20 min to 4 min.

### Tooling
- `np.load(npz)` is lazy: indexing it in a loop re-reads the file (110 s over the bind mount). Load
  each array once.
- PowerShell 5.1 `Set-Content -Encoding utf8` writes a BOM, which breaks `blender_manifest.toml`.
  Manifest permission strings must be 64 characters or fewer.
- In Git Bash, set `MSYS_NO_PATHCONV=1` for docker commands. Write Python edits with
  `PYTHONUTF8=1` or an explicit `encoding="utf-8"`; the host default is cp1252.
- Blender 5.x API changes:
  - tracking markers: `track.markers.insert_frame(frame, co=...)` (`co` must be a keyword argument);
  - multilayer EXR needs `image_settings.media_type = "MULTI_LAYER_IMAGE"`;
  - `action.fcurves` is gone (layered actions), so set `preferences.edit.keyframe_new_interpolation_type`
    before inserting keys;
  - the compositor uses `scene.compositing_node_group`;
  - the Movie Distortion "Type" is a menu socket.
- The RTX 5090 needs torch cu130 builds; upstream's cu126 builds won't run on it.

### Review discipline
Always look at the frames before calling a result done:
- 1:1 crops;
- a gait or contact strip of consecutive frames;
- a plate-vs-comp comparison;
- numeric checks of the passes (shadow-pass percentiles, black frames).

Most of the real bugs were caught this way.

---

## 8. Known limits and next steps

- Zooms: per-frame focal, but k1/k2 are constant. The principal point is fixed. The first frames of a zoom
  are less accurate.
- Many tracks don't improve accuracy on the shots tested so far (they add weaker features), even though they're
  cheap now.
- Pi4 depth can be non-rigid on unusual footage (the reframed 360 clip). Multi-view stereo on the
  solved cameras would fix the set geometry there.
- The robot is built from primitives. The next step in realism is a modelled, high-detail asset that
  keeps the same rig, lighting and composite.
- Candidates: writing tracks into Blender's clip tracker, a metric-scale fix from a known measurement,
  a curved walk path (the robot button walks a straight line), and an "Insert Asset" button for any rigged
  character.

---

## 9. Maintenance notes

- The installed add-on is a **copy** at
  `%APPDATA%\Blender Foundation\Blender\5.3\extensions\user_default\cameramatch_pi4\`. After editing
  `blender_addon/`, copy `__init__.py` there (or reinstall the zip from `dist/`).
- After changing `tools/`, nothing needs rebuilding: the folder is bind-mounted. The matte-server only
  picks up changes after a restart (`docker compose restart matte-server`). After rebuilding the
  matte image, recreate the container with `docker compose up -d matte-server`; a restart keeps the
  old image.
- Build the extension zip with
  `blender --command extension build --source-dir blender_addon/cameramatch_pi4 --output-dir dist`.
- Run the GUI tests with output to a file (`> log 2>&1`), not through a `| grep | tail` pipe: a GUI
  Blender stalled mid-render with its output piped.
- After changing `server/`, rebuild the pi4 image: `docker compose up -d --build pi4`.
- Large outputs accumulate in `work/`. Clean them up manually when needed; nothing deletes them
  automatically.
