$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$LogPath = Join-Path $ProjectRoot "miktex_admin_check_update.log"
$env:PATH = "C:\Program Files\MiKTeX\miktex\bin\x64;" + $env:PATH

"MiKTeX admin update check started: $(Get-Date -Format o)" | Set-Content -Path $LogPath -Encoding UTF8
">>> miktex --admin packages update-package-database" | Add-Content -Path $LogPath -Encoding UTF8
cmd.exe /c "miktex --admin packages update-package-database" 2>&1 | Add-Content -Path $LogPath -Encoding UTF8
"exit=$LASTEXITCODE" | Add-Content -Path $LogPath -Encoding UTF8
">>> miktex --admin packages check-update" | Add-Content -Path $LogPath -Encoding UTF8
cmd.exe /c "miktex --admin packages check-update" 2>&1 | Add-Content -Path $LogPath -Encoding UTF8
"exit=$LASTEXITCODE" | Add-Content -Path $LogPath -Encoding UTF8
"MiKTeX admin update check finished: $(Get-Date -Format o)" | Add-Content -Path $LogPath -Encoding UTF8
