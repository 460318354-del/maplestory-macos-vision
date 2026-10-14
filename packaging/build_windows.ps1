$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "未找到 Windows Python Launcher。请先安装 Python 3.11 或更高版本，并勾选 Add Python to PATH。"
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3 -m venv .venv
}

$Python = (Resolve-Path ".venv\Scripts\python.exe").Path
& $Python -m pip install --upgrade pip
& $Python -m pip install .
& $Python -m pip install --upgrade pyinstaller

& $Python -m PyInstaller packaging\MapleVision.spec --clean --noconfirm

$Executable = Join-Path $ProjectRoot "dist\MapleVision\MapleVision.exe"
if (-not (Test-Path $Executable)) {
    throw "打包完成但未找到 $Executable"
}

$Archive = Join-Path $ProjectRoot "dist\MapleVision-windows.zip"
if (Test-Path $Archive) {
    Remove-Item $Archive -Force
}
Compress-Archive -Path (Join-Path $ProjectRoot "dist\MapleVision") -DestinationPath $Archive -Force

Write-Host ""
Write-Host "Windows 打包完成：$Executable"
Write-Host "Windows 压缩包：$Archive"
Write-Host "启动命令：dist\MapleVision\MapleVision.exe --gui"
