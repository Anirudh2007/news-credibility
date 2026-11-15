# Uploads the app to the Azure VM and starts it behind HTTPS. Run from news-credibility/ after create_vm.ps1:
#   powershell -ExecutionPolicy Bypass -File configs\azure\deploy.ps1
# Sends: the code (no venvs, caches or databases), models/ (without the research-only Indian index and unused
# experiment folders), data/reference/, and the .env with the API keys. Re-run it after code changes.
# After a change to frontend/ only: add -FrontendOnly (sends the code, rebuilds and restarts only the UI; the API
# keeps running and the models are not uploaded again):
#   powershell -ExecutionPolicy Bypass -File configs\azure\deploy.ps1 -FrontendOnly
param(
    [string]$Group = "newscred-rg",
    [string]$Vm = "newscred-vm",
    [string]$Key = "$env:USERPROFILE\.ssh\newscred_azure",
    [switch]$FrontendOnly
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")   # news-credibility/
if (-not (Get-Command az -ErrorAction SilentlyContinue)) { $env:Path += ";C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin" }
$fqdn = az vm show -d -g $Group -n $Vm --query fqdns -o tsv
if (-not $fqdn) { throw "VM $Vm not found or not running (az vm start -g $Group -n $Vm)." }
$target = "azureuser@$fqdn"
$ssh = "ssh -i `"$Key`" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"

# Native commands (ssh, tar, scp) do not stop a PowerShell script when they fail, so each step is checked:
# without this the script printed "Started" even when the VM could not be reached.
function Assert-Step([string]$What) {
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed (exit code $LASTEXITCODE). Nothing after this step was done. If SSH timed out, your IP " +
              "address has probably changed since the VM was created (SSH is allowed from one address only). Allow the " +
              "new one, then run this script again: az network nsg rule update -g $Group --nsg-name ${Vm}NSG " +
              "-n default-allow-ssh --source-address-prefixes <your-public-ip>/32"
    }
}

Write-Host "1/4 Waiting for the VM's first-boot setup (Docker install)..."
cmd /c "$ssh $target `"cloud-init status --wait > /dev/null; mkdir -p news-credibility; docker --version`""
Assert-Step "Connecting to the VM over SSH"

Write-Host "2/4 Code..."
# The archive is written to a file first and tar's exit code is checked. In a "tar | ssh" pipe only ssh's exit
# code is visible, so a tar that failed part-way went unnoticed and the VM got an incomplete copy.
$codeTar = Join-Path $env:TEMP "newscred-code-$PID.tar"
try {
    cmd /c "tar -cf `"$codeTar`" --exclude=./data --exclude=./models --exclude=./.git --exclude=./.venv --exclude=./venv --exclude=node_modules --exclude=./frontend/dist --exclude=__pycache__ --exclude=.pytest_cache --exclude=.env --exclude=*.db ."
    if ($LASTEXITCODE -ne 0) { throw "Packing the code failed (tar exit code $LASTEXITCODE). Nothing was uploaded." }
    # The old code folders are deleted before the new ones are extracted. Extracting over them kept every file
    # that has since been removed from the project (the old Streamlit UI and its test). models/, data/ and .env stay.
    cmd /c "$ssh $target `"cd news-credibility && rm -rf src api tests frontend configs && tar -xf -`" < `"$codeTar`""
    Assert-Step "Uploading the code"
} finally {
    Remove-Item $codeTar -ErrorAction SilentlyContinue
}

if ($FrontendOnly) {
    Write-Host "Rebuilding and restarting the UI only..."
    cmd /c "$ssh $target `"cd news-credibility && SITE_ADDRESS=$fqdn docker compose -f docker-compose.yml -f configs/azure/docker-compose.azure.yml up -d --build --no-deps --remove-orphans frontend`""
    Assert-Step "Rebuilding the web app on the VM"
    Write-Host ""
    Write-Host "UI updated: https://$fqdn (refresh the page; it can take about a minute)" -ForegroundColor Green
    return
}

Write-Host "3/4 Models and reference data (about 2 GB; depends on your upload speed)..."
foreach ($path in "models", "data/reference") {
    if (-not (Test-Path $path)) { throw "$path not found in $(Get-Location). Nothing after the code upload was done." }
}
# About 2 GB, so this one is streamed and not written to a file first. tar's exit code is lost in the pipe, but
# tar prints nothing when it succeeds, so its messages are saved and any message stops the script.
$tarLog = Join-Path $env:TEMP "newscred-tar-$PID.log"
try {
    cmd /c "tar -cf - --exclude=models/retrieval_index_india --exclude=models/meta_v4 --exclude=models/media_reuse_clf.joblib models data/reference 2> `"$tarLog`" | $ssh $target `"tar -xf - -C news-credibility`""
    Assert-Step "Uploading the models and reference data"
    $tarErrors = Get-Content $tarLog -Raw -ErrorAction SilentlyContinue
    if ($tarErrors) { throw "Packing the models and reference data failed, so the copy on the VM is incomplete. tar said: $tarErrors" }
} finally {
    Remove-Item $tarLog -ErrorAction SilentlyContinue
}
$envFile = @(".env", "..\.env") | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($envFile) { cmd /c "scp -i `"$Key`" -o StrictHostKeyChecking=accept-new `"$envFile`" ${target}:news-credibility/.env"; Assert-Step "Uploading .env" }
else { Write-Host "   no .env found: live fact-checks stay off" -ForegroundColor Yellow }

Write-Host "4/4 Building the image and starting (10-15 minutes the first time)..."
cmd /c "$ssh $target `"cd news-credibility && SITE_ADDRESS=$fqdn docker compose -f docker-compose.yml -f configs/azure/docker-compose.azure.yml up -d --build --remove-orphans`""
Assert-Step "Building and starting the containers on the VM"

Write-Host ""
Write-Host "Started. The first start loads the models for a few minutes, then: https://$fqdn" -ForegroundColor Green
