# CameraMatch Pi4 - one-time setup for Windows (PowerShell 5.1+).
#
#   powershell -ExecutionPolicy Bypass -File setup.ps1            # check, build the engine, install the add-on
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Pull      # download prebuilt engine images (faster)
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -CheckOnly # only report what is missing
#
# Checks WSL2, the NVIDIA GPU and driver, Docker Desktop and where it keeps its data (free space), builds or
# downloads the two engine images, starts the engine (the first start downloads the models), installs the
# Blender add-on and points it at this folder. Ends with READY or the exact list of what is missing.
# It never moves Docker's data: set the location in Docker Desktop > Settings > Resources > Advanced >
# Disk image location BEFORE building (the images, build cache and model weights take about 60 GB).
#
# Non-commercial use only: the Pi4 model (CC BY-NC 4.0) and MatAnyone 2 (NTU S-Lab License 1.0).

param(
    [switch]$Pull,
    [switch]$CheckOnly,
    [switch]$NoAddon,
    [string]$Registry = "ghcr.io/urbandronedesign",
    [string]$Blender = ""
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Work = Join-Path $Root "work"
$Missing = New-Object System.Collections.Generic.List[string]
$Warn = New-Object System.Collections.Generic.List[string]
$NeedGB = 60

function Say($icon, $text) { Write-Host ("  {0} {1}" -f $icon, $text) }
function Ok($t) { Say "[ok]" $t }
function Bad($t, $fix) { Say "[!!]" $t; $Missing.Add("$t -> $fix") | Out-Null }
function Note($t) { Say "[..]" $t; $Warn.Add($t) | Out-Null }

Write-Host ""
Write-Host "CameraMatch Pi4 setup ($Root)"
Write-Host "Non-commercial use only (Pi4: CC BY-NC 4.0, MatAnyone 2: NTU S-Lab License 1.0)."
Write-Host ""

# ---------------------------------------------------------------- 1. WSL2
Write-Host "1. Windows Subsystem for Linux"
$wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
if (-not $wsl) {
    Bad "WSL is not installed" "run 'wsl --install' in an administrator PowerShell, then reboot"
} else {
    $null = & wsl.exe --status 2>$null
    if ($LASTEXITCODE -eq 0) { Ok "WSL2 present" } else { Bad "WSL is not set up" "run 'wsl --install' (administrator), then reboot" }
}

# ---------------------------------------------------------------- 2. GPU
Write-Host "2. NVIDIA GPU"
$smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $smi) {
    Bad "no NVIDIA driver found (nvidia-smi)" "install the current NVIDIA Game Ready / Studio driver"
} else {
    $q = & nvidia-smi.exe --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ($q) {
        $p = $q.Split(",") | ForEach-Object { $_.Trim() }
        $vram = [double]$p[1] / 1024
        Ok ("{0}, {1:N0} GB VRAM, driver {2}" -f $p[0], $vram, $p[2])
        if ($vram -lt 12) { Note ("{0:N0} GB VRAM: lower Max Frames / Model Resolution in Advanced: Solve (tested on 32 GB)" -f $vram) }
        if ([double]($p[2].Split(".")[0]) -lt 570) { Bad "driver $($p[2]) is too old for CUDA 13" "update the NVIDIA driver (570 or newer)" }
    } else {
        Bad "nvidia-smi did not answer" "reinstall the NVIDIA driver"
    }
}

# ---------------------------------------------------------------- 3. Docker
Write-Host "3. Docker Desktop"
$docker = (Get-Command docker.exe -ErrorAction SilentlyContinue).Source
if (-not $docker -and (Test-Path "C:\Program Files\Docker\Docker\resources\bin\docker.exe")) {
    $docker = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
}
$dockerUp = $false
if (-not $docker) {
    Bad "Docker Desktop is not installed" "install it from https://www.docker.com/products/docker-desktop/ (WSL2 backend), set its Disk image location, then run setup.ps1 again"
} else {
    $null = & $docker info --format "{{.ServerVersion}}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        $dd = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        if ((Test-Path $dd) -and -not $CheckOnly) {
            Say "[..]" "starting Docker Desktop..."
            Start-Process $dd
            for ($i = 0; $i -lt 60; $i++) {
                Start-Sleep -Seconds 3
                $null = & $docker info --format "{{.ServerVersion}}" 2>$null
                if ($LASTEXITCODE -eq 0) { break }
            }
        }
    }
    $null = & $docker info --format "{{.ServerVersion}}" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $dockerUp = $true
        Ok "Docker is running"
    } else {
        Bad "Docker is installed but not running" "start Docker Desktop, then run setup.ps1 again"
    }
    # where Docker keeps its data (images, build cache, model weights)
    $store = Join-Path $env:APPDATA "Docker\settings-store.json"
    if (-not (Test-Path $store)) { $store = Join-Path $env:APPDATA "Docker\settings.json" }
    $dataDir = Join-Path $env:LOCALAPPDATA "Docker\wsl"
    if (Test-Path $store) {
        try {
            $cfg = Get-Content $store -Raw | ConvertFrom-Json
            if ($cfg.CustomWslDistroDir) { $dataDir = $cfg.CustomWslDistroDir }
            elseif ($cfg.dataFolder) { $dataDir = $cfg.dataFolder }
        } catch { }
    }
    $drive = (Split-Path -Qualifier $dataDir).TrimEnd(":")
    $free = (Get-PSDrive -Name $drive -ErrorAction SilentlyContinue).Free / 1GB
    $have = 0
    if ($dockerUp) {
        foreach ($im in @("cameramatch-pi4:latest", "cameramatch-matte:latest")) {
            $null = & $docker image inspect $im 2>$null
            if ($LASTEXITCODE -eq 0) { $have++ }
        }
    }
    Say "[..]" ("Docker data: {0} (drive {1}:, {2:N0} GB free)" -f $dataDir, $drive, $free)
    if ($have -lt 2 -and $free -lt $NeedGB) {
        Bad ("only {0:N0} GB free on {1}: for Docker data (about {2} GB needed)" -f $free, $drive, $NeedGB) "free space, or move Docker's Disk image location to a larger drive (Docker Desktop > Settings > Resources > Advanced)"
    } elseif ($drive -eq "C" -and $have -lt 2) {
        Note "Docker keeps its data on C:. If C: is small, move the Disk image location to another drive first (Docker Desktop > Settings > Resources > Advanced)"
    }
}

# ---------------------------------------------------------------- 4. Blender
Write-Host "4. Blender"
if (-not $Blender) {
    $cands = Get-ChildItem "C:\Program Files\Blender Foundation\Blender *\blender.exe" -ErrorAction SilentlyContinue |
        Sort-Object { [version](($_.Directory.Name -replace "Blender ", "") + ".0") } -Descending
    if ($cands) { $Blender = $cands[0].FullName }
}
if ($Blender -and (Test-Path $Blender)) {
    $bv = ((Split-Path -Leaf (Split-Path -Parent $Blender)) -replace "Blender ", "")
    if ([version]($bv + ".0") -lt [version]"5.1.0") { Bad "Blender $bv is too old" "install Blender 5.1 or newer (5.3 tested)" }
    else { Ok "Blender $bv ($Blender)" }
} else {
    Bad "Blender not found" "install Blender 5.3 from blender.org, or pass -Blender <path to blender.exe>"
}

if ($CheckOnly) {
    Write-Host ""
    if ($Missing.Count -eq 0) { Write-Host "Checks passed. Run setup.ps1 without -CheckOnly to install." }
    else { Write-Host "Missing:"; $Missing | ForEach-Object { Write-Host "  - $_" } }
    exit ([int]($Missing.Count -gt 0))
}
if ($Missing.Count -gt 0) {
    Write-Host ""
    Write-Host "Fix these first, then run setup.ps1 again:"
    $Missing | ForEach-Object { Write-Host "  - $_" }
    exit 1
}

# ---------------------------------------------------------------- 5. engine images
Write-Host "5. Engine images"
New-Item -ItemType Directory -Force -Path $Work | Out-Null
Push-Location $Root
$ver = (Select-String -Path "blender_addon\cameramatch_pi4\blender_manifest.toml" -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
foreach ($pair in @(@("pi4", "cameramatch-pi4"), @("matte-server", "cameramatch-matte"))) {
    $svc, $img = $pair
    $null = & $docker image inspect "${img}:latest" 2>$null
    if ($LASTEXITCODE -eq 0) { Ok "$img present"; continue }
    if ($Pull) {
        Say "[..]" "downloading $Registry/${img}:$ver (several GB)..."
        & $docker pull "$Registry/${img}:$ver"
        if ($LASTEXITCODE -eq 0) { & $docker tag "$Registry/${img}:$ver" "${img}:latest"; Ok "$img downloaded"; continue }
        Note "download failed: building $img locally instead"
    }
    Say "[..]" "building $img (20-40 min the first time)..."
    & $docker compose build $svc
    if ($LASTEXITCODE -ne 0) { Bad "building $img failed" "see the output above"; Pop-Location; exit 1 }
    Ok "$img built"
}

# ---------------------------------------------------------------- 6. start + model download
Write-Host "6. Engine start (the first start downloads the Pi4 model, about 4 GB)"
& $docker compose up -d pi4 matte-server
$ready = $false
for ($i = 0; $i -lt 600; $i++) {
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:8765/health" -TimeoutSec 2
        $m = Invoke-RestMethod -Uri "http://127.0.0.1:8766/health" -TimeoutSec 2
        if ($h.status -eq "ready" -and $m.status -eq "ready") { $ready = $true; break }
        if ($h.status -eq "error") { break }
        if ($i % 10 -eq 0) { Say "[..]" "solver: $($h.status), matte: $($m.status)" }
    } catch { if ($i % 10 -eq 0) { Say "[..]" "waiting for the engine..." } }
    Start-Sleep -Seconds 3
}
if ($ready) { Ok ("engine ready on {0}" -f $h.gpu) }
else { Bad "the engine did not become ready" "docker compose logs pi4 (GPU passthrough? Docker Desktop > Settings > Resources > WSL integration)" }

# ---------------------------------------------------------------- 7. add-on
if (-not $NoAddon) {
    Write-Host "7. Blender add-on"
    $zip = Join-Path $Root "dist\cameramatch_pi4-$ver.zip"
    if (-not (Test-Path $zip)) {
        & $Blender --background --factory-startup --command extension build --source-dir (Join-Path $Root "blender_addon\cameramatch_pi4") --output-dir (Join-Path $Root "dist") | Out-Null
    }
    & $Blender --background --command extension install-file -r user_default -e $zip | Out-Null
    $py = Join-Path $env:TEMP "cm_setup_prefs.py"
    @"
import bpy
p = bpy.context.preferences.addons['bl_ext.user_default.cameramatch_pi4'].preferences
p.work_dir = r'$Work'
p.engine_dir = r'$Root'
bpy.ops.wm.save_userpref()
print('CM_PREFS_OK')
"@ | Set-Content -Encoding utf8 $py
    $out = & $Blender --background --python $py 2>&1
    Remove-Item $py -ErrorAction SilentlyContinue
    if ($out -match "CM_PREFS_OK") { Ok "add-on $ver installed, Work folder = $Work" }
    else { Bad "the add-on could not be configured" "Blender > Preferences > Add-ons > CameraMatch Pi4 > Work folder = $Work" }
}
Pop-Location

Write-Host ""
if ($Missing.Count -eq 0) {
    Write-Host "READY. Open Blender > Movie Clip Editor (or 3D Viewport) > Sidebar (N) > Pi4 tab."
    Write-Host "No footage yet? Click 'Try the Sample Shot' in step 1."
} else {
    Write-Host "Missing:"
    $Missing | ForEach-Object { Write-Host "  - $_" }
}
$Warn | ForEach-Object { Write-Host "  note: $_" }
exit ([int]($Missing.Count -gt 0))
