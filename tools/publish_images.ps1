# Maintainers: push the two engine images to the GitHub Container Registry so users can `setup.ps1 -Pull`.
#
#   gh auth refresh -s write:packages          # once: lets gh push packages
#   powershell -ExecutionPolicy Bypass -File tools\publish_images.ps1
#
# Pushes ghcr.io/<owner>/cameramatch-pi4:<version> and cameramatch-matte:<version> (about 20 + 25 GB).
# Model weights are NOT in the images (they download into Docker volumes on first start).
# The images contain Pi4 (CC BY-NC 4.0) and MatAnyone 2 (NTU S-Lab License 1.0) code: redistribution is for
# non-commercial use only, with their licence notices. After the first push, make each package public on
# GitHub (Packages > package settings > Change visibility).

param([string]$Owner = "urbandronedesign")

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ver = (Select-String -Path (Join-Path $Root "blender_addon\cameramatch_pi4\blender_manifest.toml") -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
$user = (& gh api user --jq .login)
& gh auth token | docker login ghcr.io -u $user --password-stdin
if ($LASTEXITCODE -ne 0) { Write-Host "docker login to ghcr.io failed (gh auth refresh -s write:packages)"; exit 1 }
foreach ($img in @("cameramatch-pi4", "cameramatch-matte")) {
    $dst = "ghcr.io/$Owner/${img}:$ver"
    docker tag "${img}:latest" $dst
    docker push $dst
    if ($LASTEXITCODE -ne 0) { Write-Host "push failed: $dst"; exit 1 }
    Write-Host "pushed $dst"
}
