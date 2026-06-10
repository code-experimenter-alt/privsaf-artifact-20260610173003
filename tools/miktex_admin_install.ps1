$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$LogPath = Join-Path $ProjectRoot "miktex_admin_install.log"
$env:PATH = "C:\Program Files\MiKTeX\miktex\bin\x64;" + $env:PATH

"MiKTeX admin package install started: $(Get-Date -Format o)" | Set-Content -Path $LogPath -Encoding UTF8
"PATH=$env:PATH" | Add-Content -Path $LogPath -Encoding UTF8

function Invoke-Logged {
    param([string] $CommandLine)
    ">>> $CommandLine" | Add-Content -Path $LogPath -Encoding UTF8
    $output = cmd.exe /c $CommandLine 2>&1
    $output | Add-Content -Path $LogPath -Encoding UTF8
    "exit=$LASTEXITCODE" | Add-Content -Path $LogPath -Encoding UTF8
}

Invoke-Logged "mpm --admin --update-db"

$packages = @(
    "ieeetran",
    "comment",
    "cite",
    "caption",
    "algorithms",
    "float",
    "algorithmicx",
    "amscls",
    "multirow",
    "booktabs",
    "psnfss"
)

foreach ($pkg in $packages) {
    Invoke-Logged "mpm --admin --install=$pkg"
}

Invoke-Logged "initexmf --admin --update-fndb"
Invoke-Logged "mpm --admin --verify"
"MiKTeX admin package install finished: $(Get-Date -Format o)" | Add-Content -Path $LogPath -Encoding UTF8
