# CameraMatch Pi4: project guide

The working reference for this project: what was built, how to run it, what the shots showed, what went
wrong and what was learned. README.md documents the add-on for users; this file covers everything around it.
It is written for both of us (the user and Claude): read it before changing anything, and update it when
something changes.

Last updated 2026-11-19, at **v0.13.1**. Public repository:
[github.com/urbandronedesign/cameramatch-pi4](https://github.com/urbandronedesign/cameramatch-pi4)
(GPL-3.0 code, releases with the add-on zip on GitHub).

---

## 1. What this is

CameraMatch Pi4 is a **Blender 5.3 extension** that matches the camera of a video and integrates CG into it:
1. **Solve:** π⁴ ([yyfz/Pi4](https://github.com/yyfz/Pi4)), a feed-forward 4D geometry model, gives poses,
   depth, a point cloud and moving-object masks in one pass; a plate-resolution bundle adjustment (KLT tracks,
   GPU Levenberg-Marquardt) refines poses, focal (or per-frame focal for zooms), k1/k2 and rolling shutter.
2. **Integrate:** a set rebuilt from the point cloud (floor shadow catcher, walls), the plate as lighting,
   characters (imported glTF / FBX / Mixamo, or a procedural robot) walking on paths with foot locking and
   collision avoidance, Cycles render and a physically matched composite.
3. **Hand off:** export of camera, set, tracks, CG, plate, masks, depth, lens ST maps and lighting for Maya,
   Houdini, Nuke, Unreal and others.

The UI is a guided six-step panel; an installer and an engine button hide Docker.

**License constraint: non-commercial use only.** π⁴ code and weights are CC BY-NC 4.0, MatAnyone 2 is under
the NTU S-Lab License 1.0 (non-commercial), SAM 2.1 is Apache-2.0, OneFormer is MIT. The add-on code is
GPL-3.0. Weights live in Docker volumes and are never bundled.

**Machine:** Windows 11, RTX 5090 (32 GB, Blackwell sm_120), Docker Desktop on WSL2.
**Docker data lives on `D:\DockerData` (Docker Desktop's `CustomWslDistroDir`). Never put Docker data on C:.**

**Git:** commits use the repo-local identity `urbandronedesign <7166404+urbandronedesign@users.noreply.github.com>`
(the global identity is someone else's, leave it). Commit, push and publish releases **only when the user asks**
("go" after a proposal counts); the permission layer blocks unrequested publishing.

---

## 2. Architecture

```
Blender 5.3 (extension "cameramatch_pi4": Movie Clip Editor or 3D Viewport > Sidebar (N) > Pi4)
   |  HTTP on localhost; job folders shared through ./work (mounted as /work)
   v
pi4 container        127.0.0.1:8765   POST /solve, /refine   GET /progress, /health, /cancel
matte-server         127.0.0.1:8766   POST /matte, /set, /comp, /check, /maps   GET /progress, /health, /cancel
matte (tools)        on demand        docker compose run --rm matte python /tools/...
```
The add-on starts / stops both services itself (`engine.py`, `docker compose up -d pi4 matte-server`,
launching Docker Desktop if it is closed) and polls their health in a background thread.

| Path | Role |
|---|---|
| `setup.ps1` | One-time installer: checks WSL2, GPU + driver, Docker and its data drive, Blender; builds (or `-Pull`s) the images, starts the engine, installs and configures the add-on |
| `docker/Dockerfile` | `cameramatch-pi4`: CUDA 13, torch 2.12.1+cu130, Pi4 pinned to `32dfbea`, scipy, transformers 5.23 (OneFormer). `server/` is COPIED in: rebuild after changes |
| `docker/Dockerfile.matte` | `cameramatch-matte`: SAM 2.1, MatAnyone 2, OpenCV, Open3D, OpenEXR. `tools/` is bind-mounted: restart after changes |
| `server/server.py` | Pi4 solve and refine jobs, `detect_zoom`, `/cancel`; writes `result.npz`, `pi4_cache.npz`, `refine_stats.json` |
| `server/refine.py` | KLT tracks, point initialisation, IRLS driver, `diag` (3×3 rejection grid, bad frames), saves semantic masks |
| `server/ba_lm.py` | Default solver: GPU LM with Schur complement; per-frame focal (zoom); rolling-shutter readout |
| `server/semantic.py` | OneFormer ADE20K masks (trees, water, sky...) every 5th frame |
| `server/progress.py` = `tools/progress.py` | Per-job stages / log, `/progress`; `update()` raises `Cancelled` after `/cancel` |
| `tools/matte_server.py` | `/matte` (SAM + MatAnyone), `/set` (mesh + env), `/comp` (comp_plate, with the job's lens), `/check` (overlay), `/maps` (export sequences) |
| `tools/comp_plate.py` | The composite: plate × shadow pass + CG, **CG and shadow distorted with the solved k1/k2**, black level, softness, light wrap, grain |
| `tools/export_maps.py` | Plate / masks / depth sequences at plate resolution for the export; depth scaled to the solve |
| `tools/overlay_check.py` | Ground-grid overlay video + KLT stick metric (Check Overlay) |
| `tools/build_lowpoly_mesh.py`, `build_plate_envmap.py` | Set mesh (Poisson, decimated) and `plate_env.png` |
| `tools/make_walkthrough.py`, `publish_images.ps1` | README walkthrough video/GIF; GHCR image publishing (not done yet) |
| `blender_addon/cameramatch_pi4/__init__.py` | UI (guided panel + Advanced sub-panels), operators, progress, import |
| `.../engine.py` | Health poller, Start / Stop Engine |
| `.../guide.py` | Footage presets, solve grade + suggestions + one-click fixes, plain-language errors |
| `.../set_builder.py` | Floor fit, set (catchers, walls, street proxy walls), world, render setup |
| `.../characters.py` | Walk-cycle analysis, NLA looping, arrival, foot slide, root-motion removal, object-transform stripping |
| `.../paths.py`, `collide.py` | Curves → ground paths, `move_one`; walls KD-tree, movers registry, lateral avoidance, walkable street |
| `.../robot.py` | Procedural walking robot |
| `.../ignore_mask.py` | Blender Mask → per-frame PNGs for the solve |
| `.../export.py` | Alembic / FBX / USD / Nuke / ST maps / camera.json / README in the export folder |
| `tests/` | Scene builders, GUI drivers, checks (§7) |
| `work/` | Job folders and test outputs (git-ignored) |
| `samples/` | Clips and assets (git-ignored; Mixamo test files in `samples/mixamo/`, never committed) |

**A job folder** (`work/<clip>_<date>_<time>/`):

| File | Written by |
|---|---|
| `input/` | the add-on (copy of the clip or `seq/` images) |
| `result.npz`, `result.json` | Solve (`result.json` keeps the first solve's summary; Re-solve updates `refine_stats.json`) |
| `pi4_cache.npz` | Solve: depth × metric scale (float16), dynamic mask, source path |
| `refine_stats.json` | every Solve / Re-solve: errors, `diag`, `zoom_auto` (what the grade reads) |
| `masks/mask_#####.png`, `semantic/sem_#####.png` | Pi4 motion masks (model res); semantic masks (source frame + 1) |
| `ignore_mask/`, `ignore_mask.json` | rasterised Ignore Mask |
| `matte/fused/alpha_####.png`, `occlusion/` | Refine Matte |
| `lowpoly.npz/.ply`, `plate_env.png`, `plate_clean.png` | Build Set (matte server) |
| `check/overlay.mp4`, `check/metrics.json` | Check Overlay |
| `render/`+`comp/`, `preview/`+`comp_preview/` | Final Render, Preview (50 %) |
| `export/`, `export_maps/` | Export (default folder); map staging |
| `scene_scale.json` | Set Real Scale |

---

## 3. Everyday commands

```bash
# install / check (PowerShell)
powershell -ExecutionPolicy Bypass -File setup.ps1 [-CheckOnly] [-Pull]
# services (normally the add-on's Start / Stop Engine does this)
docker compose up -d pi4 matte-server
curl localhost:8765/health ; curl localhost:8766/health
docker compose logs --tail 30 matte-server         # first look when the engine light is not green
# Blender
"C:/Program Files/Blender Foundation/Blender 5.3/blender.exe"
# a tool in the matte image (Git Bash: MSYS_NO_PATHCONV=1 stops path mangling)
MSYS_NO_PATHCONV=1 docker compose run --rm -v "$(pwd -W)/work:/work" --entrypoint python matte /tools/<script>.py ...
# sync the add-on into Blender after editing it (ALL modules)
cp blender_addon/cameramatch_pi4/*.py "$APPDATA/Blender Foundation/Blender/5.3/extensions/user_default/cameramatch_pi4/"
# build the extension zip
blender --background --factory-startup --command extension build --source-dir blender_addon/cameramatch_pi4 --output-dir dist
```
Neither ffmpeg nor OpenCV is on the Windows host (nor numpy in its Python): run them in the matte container.

---

## 4. The add-on

**Guided panel** (identical in the Movie Clip Editor and the 3D Viewport, tab Pi4):
- **Engine light** + Start / Stop (green = both servers ready; "Engine off"; "partly running" = one server
  down, usually a crash loop: check its log).
- **Six steps**, ticked from the scene state; the first unfinished one is open, *Show All* opens the rest:
  1. **Load Footage** (or *Try the Sample Shot*: Mixkit #4600 + CC0 RobotExpressive downloaded into work/samples);
  2. **Footage preset** (Auto, Phone, Drone, Tripod / slow pan, Zoom lens, Cinema camera, Night) + **Solve Camera**,
     Re-solve;
  3. **Check the Solve:** grade Good / Fair / Poor, suggestions with one-click fixes (they set the option and
     re-solve, or create an Ignore Mask in Mask mode), **Check Overlay**, Show Footage, Set Real Scale;
  4. **Build Set + Lighting**;
  5. **Add Character** (glTF / GLB / FBX, scaled to Character Height, on the floor at the cursor), New Path,
     **Walk Along Path**, **Walk to Cursor**, Add Walking Robot, Place on Floor, Avoid Collisions;
  6. **Preview** (50 %, 24 samples) / **Final Render**: render + comp, loaded as a clip.
- **Export to Other Software** box (once solved): **Export Camera + Scene** (§5.6).
- Progress box: stage, %, time left, **Cancel**; results and errors in plain words, *Start Engine* when needed.
- Collapsed sub-panels: *Advanced: Solve*, *Advanced: Set & Characters*, *Advanced: Render*, *More Tools*
  (Refine Matte, Export Tracks, Re-import, Check Server, scene scale).

`_main_clip()` prefers the Footage pointer (`scene.cameramatch.clip`), so everything works from the 3D Viewport,
and derived clips (`_check`, `_comp`, `_preview`, `_matte`, `_motion_mask`) are never taken for the footage.

**Key settings**

| Setting | Default | Notes |
|---|---|---|
| Footage preset | Auto | Sets the rows below together (`guide.PRESETS`) |
| Zoom | Auto | Zoom solve when Pi4's smoothed per-frame focal varies > 8 % (`server.detect_zoom`) |
| Rolling Shutter | Auto | On for portrait clips; Render & Comp matches it in Cycles (rows only) |
| Solve Lens Distortion | on | **Off for slow / long-lens / cinema shots** (k1/k2 poorly constrained, drags the focal) |
| Lock Focal to Pi4 | off | **On for slow, low-parallax shots** (focal_sigma 0.002) |
| Ignore Trees / Water / Sky | on | OneFormer masks; never hurt in tests, halved the error on foliage |
| Ignore Mask | — | A Mask drawn in the clip editor; nothing is tracked inside it |
| Active Tracks | 600 | Sweet spot; more tracks are slower and not better |
| Max Frames / Stride / Model Resolution | 300 / 1 / 512 | GPU budget: 300 frames ≈ 12 GB |
| Character Height | 1.75 m | Add Character scales to it (0 keeps the file's size) |
| Avoid Collisions | on | Robots / Move Along Path avoid walls and earlier movers |

**What a solve imports** (`Pi4_<clip>` collection): animated camera (first frame at the origin looking +Y,
Z up), `_points` (dense cloud), `_tracks` (bundle-adjusted points), a motion-mask clip, the clip's tracking camera
with focal and Polynomial k1/k2. Scene range, resolution and fps follow the clip. Objects the camera starts
inside (the startup Cube) are hidden from render.

---

## 5. Pipelines

### 5.1 Solve
π⁴ at 512 px (≈ 0.15–0.27 s / frame) → KLT at full resolution with moving objects, semantic and Ignore masks
excluded → midpoint triangulation (Pi4 depth fallback) → IRLS LM bundle adjustment (Cauchy, then Huber) of every
pose, point, focal (or per-frame focal), k1/k2 and readout. Tracks capped at 40 observations. LM is 5–6× faster
than the old scipy path; 225k observations in under a minute.

### 5.2 Matte (`tools/matte_pipeline.py`, matte server)
SAM 2.1 seeds every 30 frames, MatAnyone 2 re-seeded in chunks, fused. Rig mode (blobs touch the border) or
scene mode (blobs supported by Pi4's dynamic mask), chosen automatically.

### 5.3 Set
`build_lowpoly_mesh.py` fuses Pi4 depth through the refined cameras → voxel filter → Poisson → trim → crop →
decimate → Taubin. `build_plate_envmap.py` builds `plate_env.png` (stretched-plate mode under 10 % coverage).
`set_builder.build_set`: robust quadratic **floor** (shadow catcher, projected plate × 1.6 for bounce), **walls**
(low-poly mesh minus floor) and **14 m street proxy walls**, all emitting the projected plate, hidden from the
camera, **shadow catchers**; world = plate environment, no lights.

### 5.4 Characters and paths
- **Imported characters** (`characters.py`): the walk cycle is analysed (planted-foot speed **profile** over its
  fractional length, forward axis, median sole height, hip travel). Root-motion cycles get an in-place copy;
  non-walks are refused. The NLA loops the cycle at a rate that matches the travel speed; at the end of the path
  the walk stops and *Idle* plays. Add Character clears the importer's NLA tracks and object-transform keys.
- **Paths** (`paths.py`): curve → arc-length ground path, heading over ±0.4 m, `move_one` keeps the object's real
  tilt (any rotation mode).
- **Avoidance** (`collide.py`): walls = KD-tree of set edges in the 0.15–2 m band; movers registry
  (`scene["cm_movers"]`); lateral line search per conflict combined with max. The **walkable street** is the
  fitted floor *in sight of the camera path*: points drawn behind a wall are moved sideways onto the street;
  animated hierarchies (characters) are movers, never walls.
- **Robot** (`robot.py`): rigid panels, 2-bone IK feet, heel / toe, pelvis bob, arm swing, head look-at.

### 5.5 Render and composite
Cycles OptiX, 192 samples (Preview 24 at 50 %), persistent data, motion blur 0.5 (+ rolling shutter window),
Standard view, multilayer EXR (Combined + Shadow Catcher). `comp_plate`: **CG and shadow pass distorted with the
solved lens**, plate × shadow, CG black level matched, 0.6 px softness, light wrap, grain measured on the plate;
MP4 + plate-vs-comp MP4.

### 5.6 Export (`export.py` + `tools/export_maps.py`)
One folder: `<clip>.abc / .fbx / .usdc` (camera, set, tracks, 150 locators, optional CG; USD with materials +
`textures/`, FBX with embedded textures, Alembic geometry only), `<clip>.nk` (Camera3, plate, STMaps, sequence
Reads, ReadGeo), `stmap_undistort.exr` / `stmap_redistort.exr`, `camera.json` / `.csv`, `plate/`, `masks/`
(moving, trees_water_sky, matte, ignore), `depth/` (Z EXR, scaled to the solve), `lighting/plate_env.png`,
`README.txt`. Plate / masks / depth come from the matte server (`/maps`), then move into the folder.

---

## 6. Shots and results

| Shot | Source | Solve | Deliverables |
|---|---|---|---|
| Synthetic GT | `tests/gt_render.py` | trajectory 0.06–0.07 %, rotation 0.16–0.23°, f 1275.8/1280 px | validation |
| Cycling (reframed 360) | `samples/cycling.mp4` | solves; Pi4 depth not rigid | `work/floor_panels`, `work/insert_test` |
| Tears of Steel bridge | `samples/tos_bridge.mp4` (frames 600–965) | 0.33 px, distortion off, focal locked | `work/tos_test/bridge_spheres_final.mp4` |
| Venice alley, robot | `samples/venice_alley.mp4` (Mixkit #4600) | 0.35 px, distortion off, focal locked | `work/venice/venice_robot_final.mp4` |
| Venice, final shot | same, job `…_114507` | idem | `work/final/venice_final_shot.mp4` (CC0 RobotExpressive + robot crossing) |
| **Venice, Mixamo shot** | same | idem | `work/mixamo_shot/venice_mixamo_shot.mp4` (Mixamo Soldier + X Bot, 284 fr) |
| Venice, guided test | same, 120 fr, job `…_143133` | 0.28 px, **k1 0.153** (distortion on) | `work/guided/` (screenshots, walkthrough), export + rebuild tests |
| Hard footage (5 shots) | `samples/hard/` | see README "Hard-footage test" | `work/hard_*` |

Credits: Tears of Steel © Blender Foundation (CC BY 3.0); Venice alley Mixkit free license; RobotExpressive CC0
(Tomás Laulhé / Quaternius); Mixamo test characters from three.js' examples (local tests only, not redistributed).

**Choosing a shot:** `tools/find_shots.py`. Slow, steady, real parallax, no existing VFX; a forward dolly down
a street is ideal for walking characters.

---

## 7. Test suite

Run GUI tests with output to a file (`> log 2>&1`), never through a pipe. **A test passes only if it checks the
pixels / numbers that matter**: the first guided run "passed" with an all-black comp.

| Test | Checks | Last result |
|---|---|---|
| `gui_guided.py` | whole guided flow in the real UI from the startup scene, screenshots, comp pixels | 20/20 |
| `gui_engine_down.py` | Docker closed → Solve says so → Start Engine (launches Docker) → Solve | ready in 13 s |
| `gui_export.py` | Export button (modal maps) in the real UI, every file and Read node | all pass, 51 s / 120 fr |
| `check_export.py` | abc / fbx / usd round trip by reprojection, zoom kept, json + ST map vs tracks, Nuke rebuild | ≤ 0.001 px; 0.318 px |
| `rebuild_from_export.py` + `compare_export_rebuild.py` | shot rebuilt in a factory Blender from the export only vs the plugin render | IoU 0.987–0.999, 0.09 px |
| `check_mixamo.py` (+ `make_mixamo_fbx.py`) | Add Character + Walk to Cursor: scale, soles, slide, facing; non-walk refused | 5 rigs pass |
| `check_collide_walk.py`, `check_paths.py`, `check_offfloor.py` | avoidance, paths, walls, paths drawn into walls | pass |
| `check_robust_features.py`, `check_rs_render.py`, `check_ignore_mask.py` | solver features, rolling shutter, masks | pass |
| `gt_*`, `check_blender_distortion.py`, `bench_ba.py` | ground truth and solver benchmarks | see README |

---

## 8. Hard-won lessons (read before changing things)

### Compositing and lighting
- **Every real-world proxy in a Cycles comp must be a shadow catcher**, or its occlusion of the sky counts as a
  CG shadow (the robot floor went to 20 %). Check: shadow-pass median ≈ 1.0 away from the CG.
- **Close the set:** a scanned set ends where the camera stopped seeing; open sky flattens the lighting.
- Coincident surfaces block each other's light; catchers act as holdouts (restrict them to ground faces when CG
  passes through foliage); relative depth tests give partial opacity on noise (use occluder-plane + matte).
- **The first movie-texture render in a background session can come back black:** warm-up render first.
- **Apply the lens to the CG.** For two releases Render & Comp laid the pinhole render on the distorted plate
  (1–3 px on centred CG, ~5 % of the distance to the centre at the edges); only rebuilding the shot from its export
  showed it. `comp_plate` now distorts CG + shadow with the job's k1/k2.
- **Objects that enclose the camera** (the startup Cube at the origin, where the first camera is) fill every
  frame: `hide_camera_blockers` runs after import and before render.
- **Keep it physical** (the user's rule): no fake shadows or lights.

### Solve
- Slow / long-lens shots: distortion runs away (k2 1.21 on the bridge) and the focal drifts → distortion off,
  focal locked. Zoom vs dolly and focal vs distortion are genuinely ambiguous there; the user picks (presets).
- Pi4's metric scale is unreliable (2.5× off on GT): check against known sizes (Set Real Scale).
- Zoom auto-detection from Pi4's focal: fixed lenses ≤ 3.8 %, a 1.6× zoom 46 %; the real `tos_zoom` clip shows
  0.3 % (Pi4 misses it): a missed zoom still needs the Zoom lens preset.
- Rolling shutter: portrait phone video reads across columns; Cycles only does top-to-bottom rows; the readout
  comes out ~12 % high on GT (central-difference velocities); holdout needs row-time poses.

### Validation
- Reprojection error alone hides problems: use holdout tracks, the overlay stick metric and GT together.
  Holdout measures self-consistency, not correctness (never choose the lens model with it). Laplace uncertainties
  are useless here (± 0.08 % on a focal that can move 14 %).
- When a shot underperforms, look at **where** the residuals are: the phone-walk error was the tree canopy
  (71 % of tracks), not rolling shutter. Non-rigid regions need semantic masks; then turn distortion off.
- The grade (Good: median < 0.5 px and p90 < 1.5 px; Fair < 1.0 / 3.0) matches every shot reviewed.
- **Compare exports by reprojection in pixels, at the observed (distorted) pixels**, not by matrices; a pinhole
  projection samples the wrong pixel with k1 0.15.
- **Pi4 depth vs the solve:** 4 % per point on Venice, 27 % on the ToS bridge. One global factor (median over the
  inlier observations); per-frame factors chase each frame's track subset (500-observation frames gave 1.02,
  100-observation frames 1.07).
- Pi4's moving mask is a soft probability: 5 % of each Venice frame (static wall in shadow) is above 0.5. Fine for
  tracking, not for roto (use the matte).

### Characters, paths and avoidance
- Drive the root with the cycle's planted-foot speed **profile** (63 → 5 cm/s slide), sampled at the true
  fractional cycle length; heading over ±0.4 m of arc; median sole height; stop the cycle at the path's end.
- Read rotations with `matrix_basis.to_euler()`, never `rotation_euler`: glTF rigs are in quaternion mode and a
  90° axis tilt was silently lost (Soldier walked sideways, 37 cm in the floor).
- Importers leave one active NLA track per animation and may key the armature object itself: clear both.
  Root-motion walks: remove the hips' drift over the cycle. Non-walks: refuse with a message.
- Avoidance: lateral line search + max envelope (iterative push-and-smooth fought itself); clear walls by the
  half-width; dense wall samples. Walkable = floor **in sight of the camera** (the floor fit runs ~0.4 m behind
  walls); push sideways, never along the path (a path into a bend folded back on itself); an imported character's
  child meshes were counted as walls until animated hierarchies were treated as movers.

### Export
- Blender's FBX importer shifts animation by 1 frame (`anim_offset=1`): looks like a 0.75 px camera error.
- Blender 5.3's USD armature writer hits "Code marked as unreachable": characters go to USD as baked meshes.
- Alembic carries no materials: USD (preview surfaces + textures/) or FBX (embedded) for characters. FBX's
  texture COPY also copies the footage (the set's projection material uses it).
- A scene whose track mesh no longer matches `result.npz` (re-solved, not re-imported) is flagged on export.
- OpenEXR's Python API groups channels (`RGBA`); the files themselves have R, G, B, A.

### Ease of use / UI
- Never block Blender's UI on HTTP: background health thread, 1 s timer redraws only on change.
- Cancel goes through the progress tracker (`Cancelled` at the next update); tool subprocesses are killed by a
  watcher; Pi4's forward pass cannot be interrupted.
- Preview speed is per-frame overhead: 1.5 s/frame at 50 % / 24 samples, 1.1 s with persistent data.
- Fit labels to the sidebar width (`_chars()`): long status text was cut off.

### Tooling
- **After editing server or tool code, syntax-check it and look at the health endpoint.** A stray bracket in
  `matte_server.py` put it in a crash loop that only showed as "Engine partly running" in a screenshot; the tests
  had called `comp_plate` directly.
- `np.load(npz)` is lazy: load each array once (110 s over the bind mount otherwise).
- PowerShell 5.1 `Set-Content -Encoding utf8` writes a BOM (breaks `blender_manifest.toml`; fine for .ps1).
  Manifest tagline ≤ 64 characters, permission strings ≤ 64.
- Git Bash: `MSYS_NO_PATHCONV=1` for docker; complex heredocs break on quotes → write patch scripts to the
  scratchpad. Host Python is cp1252: `encoding="utf-8"`. Relative paths in Blender resolve against its own
  working directory: use absolute paths in test scripts.
- Blender 5.x: `markers.insert_frame(frame, co=...)`; multilayer EXR needs `media_type = "MULTI_LAYER_IMAGE"`;
  `action.fcurves` is gone (layers → strips → channelbags; set the keyframe interpolation preference before
  keying); `scene.compositing_node_group`; Movie Distortion type is a menu socket.
- The RTX 5090 needs torch cu130 builds.

### Review discipline
Look at the frames before calling anything done: 1:1 crops, consecutive-frame strips, plate-vs-comp, numeric
pass checks, **and screenshots of the real UI** for UI work. Most real bugs (black comp, robot in the wall,
sideways Soldier, missing lens distortion, the crash loop) were found this way, not by return codes.

---

## 9. Known limits and next steps

- **Prebuilt images are not published:** `setup.ps1 -Pull` falls back to a 20–40 min build. Publishing needs
  `gh auth refresh -s write:packages`, ~45 GB upload and the user's decision (Pi4 / MatAnyone code is inside,
  non-commercial redistribution).
- Zooms: per-frame focal, constant k1/k2, centred principal point; the export's ST maps use the median focal.
- The Nuke script was verified against Nuke's conventions, not inside Nuke (not installed).
- Pi4 depth can be non-rigid or off on unusual footage (27 % on the bridge); multi-view stereo on the solved
  cameras would fix set geometry and depth there.
- Avoidance is lateral only (no waiting); a passage too narrow for two movers is reported, not solved.
- Real Mixamo downloads (Adobe login) not tested directly; three.js' Mixamo exports and Mixamo-style FBX pass.
- The Docker-Desktop-closed path is tested; a machine without Docker / WSL relies on `setup.ps1` messages.

---

## 10. Maintenance and release checklist

1. Edit; for `server/`: `docker compose up -d --build pi4`; for `tools/`: `docker compose restart matte-server`;
   then **check both `/health` endpoints**. After rebuilding the matte image use `up -d matte-server` (restart
   keeps the old image).
2. Sync **all** add-on modules into `%APPDATA%\...\extensions\user_default\cameramatch_pi4\`.
3. Run the tests that cover the change (§7) and review frames / screenshots.
4. Update README (user-facing section), this guide, `CLAUDE.md` if a quick fact changed.
5. When the user asks: bump `blender_manifest.toml`, build the zip, commit (repo-local identity, Claude
   co-author line), push, `gh release create v<version> dist/cameramatch_pi4-<version>.zip` with short notes
   (say when the engine must be rebuilt / restarted). Release videos are assets, never committed (`/dist/*.mp4`
   is ignored).
6. `work/` grows large (renders, exports, depth): clean it by hand when needed.

---

## 11. Version history

| Version | Content |
|---|---|
| 0.4–0.5 | progress + log; CG Integration buttons (Build Set, robot, Render & Comp) |
| 0.6 | GPU LM + Schur solver, zoom, Set Scale, Export Tracks |
| 0.7 | rolling shutter (solve + Cycles) |
| 0.8 | trees / water / sky semantic masks |
| 0.9 | Ignore Mask |
| 0.10 | Re-solve, robot along a curve, Move Along Path |
| 0.11 | collision avoidance, foot-locked imported characters |
| 0.12 | installer, engine button, guided panel, presets, grade + fixes, cancel, preview, Add Character, sample shot |
| 0.12.1 | Mixamo: root motion, non-walks, tilted rigs, importer leftovers; Docker-closed start tested |
| 0.13 | export: Alembic / FBX / USD / Nuke / ST maps / json; plate, masks, depth |
| 0.13.1 | lens-correct Render & Comp, materials + lighting in the export, rebuild-from-export test |
