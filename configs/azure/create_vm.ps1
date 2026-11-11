# Creates the Azure VM that hosts the app (uses Azure credit: Standard_B2as_v2, about USD 0.05 per hour).
# Run from news-credibility/ after `az login`:
#   powershell -ExecutionPolicy Bypass -File configs\azure\create_vm.ps1
# Delete everything again:  az group delete -n newscred-rg --yes
param(
    [string]$Location = "indiasouthcentral",   # Azure for Students allows only some regions
    [string]$Size = "Standard_B2as_v2",        # 2 vCPU, 8 GiB RAM
    [string]$Group = "newscred-rg",
    [string]$Vm = "newscred-vm",
    [string]$Key = "$env:USERPROFILE\.ssh\newscred_azure.pub"
)
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")   # news-credibility/
if (-not (Get-Command az -ErrorAction SilentlyContinue)) { $env:Path += ";C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin" }
if (-not (Test-Path $Key)) { throw "SSH public key not found: $Key" }

$label = "newscred-" + -join ((97..122) | Get-Random -Count 5 | ForEach-Object { [char]$_ })
$myIp = (Invoke-RestMethod -Uri "https://api.ipify.org" -UseBasicParsing).Trim()

az group create -n $Group -l $Location --tags project=news-credibility -o none
Write-Host "Creating $Vm ($Size, $Location) with DNS name $label ... (2-4 minutes)"
az vm create -g $Group -n $Vm --image Ubuntu2404 --size $Size `
    --admin-username azureuser --ssh-key-values $Key `
    --public-ip-sku Standard --public-ip-address-dns-name $label `
    --os-disk-size-gb 64 --storage-sku Premium_LRS `
    --custom-data "configs/azure/cloud-init.yaml" -o none

# SSH only from this computer's current address; the web ports open to everyone.
az network nsg rule update -g $Group --nsg-name "${Vm}NSG" -n default-allow-ssh --source-address-prefixes "$myIp/32" -o none
az vm open-port -g $Group -n $Vm --port 80,443 --priority 1010 -o none

$fqdn = az vm show -d -g $Group -n $Vm --query fqdns -o tsv
Write-Host ""
Write-Host "VM ready: $fqdn  (SSH allowed from $myIp only)" -ForegroundColor Green
Write-Host "Stop paying for compute when not needed:  az vm deallocate -g $Group -n $Vm"
Write-Host "Start again:                               az vm start -g $Group -n $Vm"
