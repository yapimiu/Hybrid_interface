# PyInstaller configuration for Hybrid Interface.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

project_root = Path(SPECPATH)
a = Analysis(
    [str(project_root / "app" / "main.py")],
    pathex=[str(project_root)],
    binaries=collect_dynamic_libs("pylsl"),
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt6", "PySide6", "PySide2", "scipy"],
    noarchive=False,
)
pyz = PYZ(a.pure)

if sys.platform == "darwin":
    exe = EXE(
        pyz, a.scripts, [], exclude_binaries=True,
        name="HybridInterface", console=False,
    )
    folder = COLLECT(
        exe, a.binaries, a.datas, strip=False, upx=False,
        name="HybridInterface",
    )
    app = BUNDLE(
        folder, name="Hybrid Interface.app",
        bundle_identifier="com.yapimiu.hybridinterface",
        info_plist={"NSHighResolutionCapable": True},
    )
else:
    exe = EXE(
        pyz, a.scripts, a.binaries, a.datas, [],
        name="HybridInterface", console=False,
    )
