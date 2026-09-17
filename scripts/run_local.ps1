# Локальный прогон на Windows: модели-основы Chronos и (по желанию) загрузка новостей GDELT.
# Нужны Python 3.11 или 3.12 и доступ к sberindex.ru, huggingface.co и data.gdeltproject.org.
#
# Файл сохранён в UTF-8 с BOM: без BOM Windows PowerShell 5.1 читает его в cp1251 и ломает кириллицу.
#
# Запуск из корня репозитория:
#   powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1            # только Chronos
#   powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1 -News      # Chronos и GDELT
#   powershell -ExecutionPolicy Bypass -File scripts\run_local.ps1 -NewsOnly  # только GDELT
#
# Итог: results_local.zip в корне репозитория (прогнозы, статусы, версии пакетов, разметка новостей).
# Исходные данные СберИндекса в архив не попадают.
param([switch]$News, [switch]$NewsOnly)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function Run($exe, [string[]]$a) { & $exe @a; if ($LASTEXITCODE -ne 0) { throw "ошибка: $exe $($a -join ' ')" } }

Step "Python"
$py = $null
foreach ($v in @("3.12", "3.11")) {
    try { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $py = @("py", "-$v"); break } } catch {}
}
if (-not $py) { throw "нужен Python 3.11 или 3.12: https://www.python.org/downloads/ (галочка Add to PATH и py launcher)" }
if (-not (Test-Path .venv\Scripts\python.exe)) { Run $py[0] @($py[1], "-m", "venv", ".venv") }
$vpy = (Resolve-Path .venv\Scripts\python.exe).Path
Run $vpy @("--version")

Step "Зависимости"
Run $vpy @("-m", "pip", "install", "--upgrade", "pip")
Run $vpy @("-m", "pip", "install", "-r", "requirements.txt")
if (-not $NewsOnly) { Run $vpy @("-m", "pip", "install", "-r", "requirements-foundation.txt") }
Run $vpy @("-m", "pip", "install", "-e", ".", "--no-deps")

Step "Данные СберИндекса (сверка sha256)"
$raw = "data\raw\sberindex_mo_spending.parquet"
$local = "..\sberindex_mo_spending.parquet"
if (-not (Test-Path $raw) -and (Test-Path $local)) {
    New-Item -ItemType Directory -Force data\raw | Out-Null
    Copy-Item $local $raw
    Write-Host "взята выгрузка из папки SberIndex"
}
if (Test-Path $raw) {
    Run $vpy @("-m", "sshocks.cli", "data")
} else {
    # если скачивание падает с CERTIFICATE_VERIFY_FAILED: скачайте parquet в браузере
    # (https://sberindex.ru/api/dataset/v1/download/potrebitelskie-beznalicnye-rashody-na-urovne-munizipalnyh-obrazovanij/parquet)
    # и положите в data\raw\sberindex_mo_spending.parquet
    Run $vpy @("-m", "sshocks.cli", "data", "--download")
}
$h = (Get-FileHash $raw -Algorithm SHA256).Hash.ToLower()
if ($h -ne "dbe07d6a3a974ab4fd13258a36382a3b056aa35d12ed6c2b132ae5775115e0f1") { throw "sha256 выгрузки не совпадает с data\hashes.json: $h" }

$pack = @()
if (-not $NewsOnly) {
    Step "Chronos: 2016 МО, 4 точки, 3 горизонта (на CPU от 10 до 40 минут)"
    Run $vpy @("-m", "sshocks.cli", "forecast", "--config", "configs/foundation.yaml")
    $st = Import-Csv outputs_foundation\forecast_status.csv
    $bad = $st | Where-Object { $_.status -ne "ok" }
    if ($bad) { $bad | Format-Table -AutoSize; throw "часть моделей пропущена, см. note выше (обычно нет доступа к huggingface.co)" }
    & $vpy -m pip freeze | Out-File -Encoding utf8 outputs_foundation\pip_freeze.txt
    $pack += "outputs_foundation"
}
if ($News -or $NewsOnly) {
    Step "GDELT 2023-2024 (около 730 дневных архивов, 1-3 часа; повторный запуск докачивает)"
    Run $vpy @("-m", "sshocks.cli", "news", "--download")
    $pack += @("data\news", "outputs\news_registry_recall.csv")
}

Step "Архив для отправки"
$tmp = "results_local"
if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
New-Item -ItemType Directory $tmp | Out-Null
foreach ($p in $pack) { Copy-Item -Recurse $p $tmp }
Copy-Item outputs\data_passport.json $tmp
if (Test-Path results_local.zip) { Remove-Item results_local.zip }
Compress-Archive -Path "$tmp\*" -DestinationPath results_local.zip
Remove-Item -Recurse -Force $tmp
Write-Host "`nГотово: $(Resolve-Path results_local.zip)" -ForegroundColor Green
