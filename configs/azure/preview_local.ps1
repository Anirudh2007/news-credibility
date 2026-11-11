# Runs the Azure stack on this PC: the same compose files, the same Caddy reverse proxy and the same settings as
# the VM (template explanations, restart policies, the UI reached only through the proxy), so the browser shows
# what the hosted site will look like. Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File configs\azure\preview_local.ps1          # http://localhost
#   powershell -ExecutionPolicy Bypass -File configs\azure\preview_local.ps1 -Https   # https://localhost
#   powershell -ExecutionPolicy Bypass -File configs\azure\preview_local.ps1 -Stop
# One difference from the VM: for "localhost" Caddy cannot get a Let's Encrypt certificate (no public name), so
# -Https uses Caddy's own local certificate and the browser warns once. On the VM the padlock is real.
param([switch]$Https, [switch]$Stop)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")   # news-credibility/
$files = @("-f", "docker-compose.yml", "-f", "configs/azure/docker-compose.azure.yml")
$env:SITE_ADDRESS = if ($Https) { "localhost" } else { "http://localhost" }   # the VM uses its DNS name here
$url = if ($Https) { "https://localhost" } else { "http://localhost" }

if ($Stop) { docker compose @files down; return }   # volumes (history, caches) are kept

docker info *> $null
if ($LASTEXITCODE -ne 0) { throw "Docker is not running: start Docker Desktop and try again." }

Write-Host "Building the image and starting API, UI and Caddy (compose waits until the models are loaded)..."
docker compose @files up -d --build
if ($LASTEXITCODE -ne 0) { throw "docker compose failed (see the messages above)." }

Write-Host "Waiting for $url ..."
$deadline = (Get-Date).AddMinutes(5)
do {
    Start-Sleep -Seconds 3
    $code = curl.exe -s -k -o NUL -w "%{http_code}" $url
} until ($code -eq "200" -or (Get-Date) -gt $deadline)
if ($code -ne "200") { throw "No answer from $url (last status: $code). Check: docker compose $($files -join ' ') logs caddy" }

Write-Host ""
Write-Host "Preview running at $url (same stack as the Azure VM). Stop it with -Stop." -ForegroundColor Green
Start-Process $url
