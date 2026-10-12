"""Установить закреплённую liblsl только в активное venv (без системного installer)."""

import argparse
import hashlib
from importlib.util import find_spec
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen
import zipfile


RELEASE = "https://github.com/sccn/liblsl/releases/download/v1.17.7/"
ASSETS = {
    "Darwin": ("liblsl-1.17-Darwin-universal.pkg",
               "b3f1a07c0f73fa60ca0beea4abece1b8465a2f5a223d4ad446dbfe725dc4a4fb"),
    "Windows": ("liblsl-1.17.7-Win_amd64.zip",
                "1285c4846f705108d417f5b7e57727f7e864941692d936fa18f8e7ab9b7112e1"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Уже загруженный официальный архив")
    parser.add_argument("--allow-base", action="store_true", help="Разрешить установку в Python изолированного CI")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix and not args.allow_base:
        parser.error("Запускайте скрипт внутри виртуального окружения")
    system = platform.system()
    if system not in ASSETS:
        parser.error("На Linux установите liblsl из официального пакета вашей платформы")
    filename, checksum = ASSETS[system]
    spec = find_spec("pylsl")
    if spec is None:
        parser.error("Сначала установите requirements.txt")
    destination = Path(spec.origin).parent / "lib"
    with tempfile.TemporaryDirectory(prefix="hybrid-liblsl-") as folder:
        root = Path(folder)
        archive = args.archive or root / filename
        if not args.archive:
            with urlopen(RELEASE + filename, timeout=60) as response, archive.open("wb") as file:
                shutil.copyfileobj(response, file)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != checksum:
            raise ValueError("SHA256 liblsl не соответствует закреплённому релизу")
        if system == "Darwin":
            expanded = root / "expanded"
            subprocess.run(["pkgutil", "--expand-full", str(archive), str(expanded)], check=True)
            libraries = list(expanded.glob("*/Payload/lsl.framework/Versions/A/lsl"))
            if len(libraries) != 1:
                raise ValueError("Не найдена библиотека в macOS package")
            shutil.copy2(libraries[0], destination / "liblsl.dylib")
            # A framework executable becomes a standalone dylib in the Python wheel.
            subprocess.run(["codesign", "--force", "--sign", "-", str(destination / "liblsl.dylib")], check=True)
        else:
            with zipfile.ZipFile(archive) as zipped:
                libraries = [n for n in zipped.namelist() if Path(n).name.lower() == "lsl.dll"]
                if len(libraries) != 1:
                    raise ValueError("Не найдена lsl.dll в Windows archive")
                with zipped.open(libraries[0]) as source, (destination / "lsl.dll").open("wb") as file:
                    shutil.copyfileobj(source, file)
    print(f"liblsl 1.17.7 установлена в {destination}")


if __name__ == "__main__":
    main()
