# PyInstaller configuration for Hybrid Interface.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, copy_metadata
from PyInstaller.utils.hooks.qt import pyqt5_library_info
import PyQt5

# Some Qt5 wheels report a lossy prefix for non-ASCII venv paths.
# Repair only the invalid auto-detected paths, using the installed wheel location.
qt_prefix = Path(PyQt5.__file__).parent / "Qt5"
if qt_prefix.is_dir() and not Path(pyqt5_library_info.location["PluginsPath"]).is_dir():
    old_prefix = pyqt5_library_info.location["PrefixPath"]
    for key, value in pyqt5_library_info.location.items():
        if value.startswith(old_prefix):
            pyqt5_library_info.location[key] = str(qt_prefix) + value[len(old_prefix):]
    pyqt5_library_info.qt_inside_package = True
    pyqt5_library_info.qt_lib_dir = qt_prefix / ("bin" if sys.platform == "win32" else "lib")

project_root = Path(SPECPATH)
a = Analysis(
    [str(project_root / "app" / "main.py")],
    pathex=[str(project_root)],
    binaries=collect_dynamic_libs("pylsl"),
    datas=[(str(project_root / "p300_analysis" / "LICENSE"), "p300_analysis"),
           (str(project_root / "p300_analysis" / "UPSTREAM.json"), "p300_analysis"),
           (str(project_root / "vendor" / "hybrid_bci"), "vendor/hybrid_bci"),
           (str(project_root / "app" / "bci" / "launch_upstream.py"), "app/bci")] + [
        item for name in ("numpy", "scipy", "pylsl")
        for item in copy_metadata(name)
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt6", "PySide6", "PySide2", "mne", "sklearn"],
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
