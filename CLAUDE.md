# CameraMatch Pi4: notes for Claude

Read `docs/PROJECT_GUIDE.md` before working here. It covers the architecture, the commands, the
shot history and the lessons learned. Keep it up to date when something changes.

Non-negotiables:
- **Non-commercial use only** (π⁴ CC BY-NC, MatAnyone 2 NTU S-Lab).
- **Docker data stays on `D:\DockerData`. Never use C:.**
- **Keep CG integration physical** (no fake shadows or lights).
- **Review the actual frames before reporting a result:** 1:1 crops, consecutive-frame strips,
  plate-vs-comp comparisons, and numeric pass checks.

Quick facts:
- Blender: `C:/Program Files/Blender Foundation/Blender 5.3/blender.exe`.
- The installed add-on is a copy: sync `blender_addon/cameramatch_pi4/__init__.py` into
  `%APPDATA%/Blender Foundation/Blender/5.3/extensions/user_default/cameramatch_pi4/` after editing it.
- Tools run in the matte image:
  `MSYS_NO_PATHCONV=1 docker compose run --rm -v "$(pwd -W)/work:/work" --entrypoint python matte /tools/...`.
  The host has no ffmpeg or cv2.
- Host Python defaults to cp1252: use `encoding="utf-8"` or `PYTHONUTF8=1` when editing files.
- In Cycles comps, every real-world proxy (walls, set mesh) must be a shadow catcher.
