# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

project_root = Path(SPECPATH).resolve().parent.parent
source_root = project_root / "src"
config_file = project_root / "config" / "default.yaml"
template_root = project_root / "assets" / "templates"

hiddenimports = [
    "win32api",
    "win32con",
    "win32gui",
    "win32process",
    "mss",
]

datas = [
    (str(config_file), "config"),
    (str(template_root), "assets/templates"),
]

a = Analysis(
    [str(project_root / "packaging" / "windows_entry.py")],
    pathex=[str(source_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "AppKit",
        "Quartz",
        "objc",
        "pyobjc",
        "pyobjc_core",
        "pyobjc_framework_Quartz",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MapleVision",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="MapleVision",
)
