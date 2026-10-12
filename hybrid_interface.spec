# PyInstaller configuration for Hybrid Interface.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files, collect_submodules, copy_metadata
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
    datas=collect_data_files("mne") + [
        item for name in ("numpy", "scipy", "mne", "scikit-learn", "pylsl")
        for item in copy_metadata(name)
    ],
    # MNE resolves public exports lazily from .pyi files; static analysis alone misses them.
    hiddenimports=collect_submodules(
        "mne", filter=lambda name: ".tests" not in name and ".conftest" not in name) + collect_submodules(
        "sklearn", filter=lambda name: ".tests" not in name),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt6", "PySide6", "PySide2"],
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
