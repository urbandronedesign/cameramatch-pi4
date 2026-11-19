# CameraMatch Pi4: notes for Claude

Read `docs/PROJECT_GUIDE.md` before working here: architecture, commands, shots, test suite, lessons, release
checklist. Keep it up to date when something changes.

Non-negotiables:
- **Non-commercial use only** (π⁴ CC BY-NC, MatAnyone 2 NTU S-Lab).
- **Docker data stays on `D:\DockerData`. Never use C:.**
- **Keep CG integration physical** (no fake shadows or lights).
- **Review the actual frames before reporting a result:** 1:1 crops, consecutive-frame strips,
  plate-vs-comp comparisons, numeric pass checks, and screenshots of the real UI for UI work.
  A test only counts if it checks the pixels / numbers that matter.
- **Commit, push and publish releases only when the user asks** (repo-local identity urbandronedesign noreply).

Quick facts:
- Blender: `C:/Program Files/Blender Foundation/Blender 5.3/blender.exe`.
- The installed add-on is a copy: after editing, sync **all** `blender_addon/cameramatch_pi4/*.py` into
  `%APPDATA%/Blender Foundation/Blender/5.3/extensions/user_default/cameramatch_pi4/`.
- `server/` changes: `docker compose up -d --build pi4`. `tools/` changes: `docker compose restart matte-server`.
  Then syntax-check and **curl both `/health` endpoints** (a crash loop only shows as "Engine partly running").
- Tools run in the matte image:
  `MSYS_NO_PATHCONV=1 docker compose run --rm -v "$(pwd -W)/work:/work" --entrypoint python matte /tools/...`.
  The host has no ffmpeg, cv2 or numpy.
- Host Python defaults to cp1252: use `encoding="utf-8"`; long patches go in scratchpad Python scripts.
- In Cycles comps, every real-world proxy (walls, set mesh) must be a shadow catcher; the CG must go through
  the solved lens (comp_plate does it).
- Test scenes: `work/guided/guided.blend` (Venice 120 fr, k1 0.153, consistent with its job),
  `work/mixamo_shot/mixamo_shot.blend` (full shot, k1 0; its track mesh predates the job's npz).
