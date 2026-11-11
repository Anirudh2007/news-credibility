# Uploads the app to the Azure VM and starts it behind HTTPS. Run from news-credibility/ after create_vm.ps1:
#   powershell -ExecutionPolicy Bypass -File configs\azure\deploy.ps1
# Sends: the code (no venvs, caches or databases), models/ (without the research-only Indian index and unused
# experiment folders), data/reference/, and the .env with the API keys. Re-run it after code changes.
param(
    [string]$Group = "newscred-rg",
    [string]$Vm = "newscred-vm",
    [string]$Key = "$env:USERPROFILE\.ssh\newscred_azure"
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")   # news-credibility/
if (-not (Get-Command az -ErrorAction SilentlyContinue)) { $env:Path += ";C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin" }
$fqdn = az vm show -d -g $Group -n $Vm --query fqdns -o tsv
if (-not $fqdn) { throw "VM $Vm not found or not running (az vm start -g $Group -n $Vm)." }
$target = "azureuser@$fqdn"
$ssh = "ssh -i `"$Key`" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"

Write-Host "1/4 Waiting for the VM's first-boot setup (Docker install)..."
cmd /c "$ssh $target `"cloud-init status --wait > /dev/null; mkdir -p news-credibility; docker --version`""

Write-Host "2/4 Code..."
cmd /c "tar -cf - --exclude=./data --exclude=./models --exclude=./.venv --exclude=./venv --exclude=__pycache__ --exclude=.pytest_cache --exclude=.env --exclude=*.db . | $ssh $target `"tar -xf - -C news-credibility`""

Write-Host "3/4 Models and reference data (about 2 GB; depends on your upload speed)..."
cmd /c "tar -cf - --exclude=models/retrieval_index_india --exclude=models/meta_v4 --exclude=models/media_reuse_clf.joblib models data/reference | $ssh $target `"tar -xf - -C news-credibility`""
$envFile = @(".env", "..\.env") | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($envFile) { cmd /c "scp -i `"$Key`" -o StrictHostKeyChecking=accept-new `"$envFile`" ${target}:news-credibility/.env" }
else { Write-Host "   no .env found: live fact-checks stay off" -ForegroundColor Yellow }

Write-Host "4/4 Building the image and starting (10-15 minutes the first time)..."
cmd /c "$ssh $target `"cd news-credibility && SITE_ADDRESS=$fqdn docker compose -f docker-compose.yml -f configs/azure/docker-compose.azure.yml up -d --build`""

Write-Host ""
Write-Host "Started. The first start loads the models for a few minutes, then: https://$fqdn" -ForegroundColor Green
