"""Автоматическая проверка состава пакета; не проверка научной точности."""

import mne
from pylsl import library_version

from .config import P300Config
from .engine import P300Engine
from .model import ALGORITHMS, make_pipeline, software_versions


def self_check():
    config = P300Config(sfreq=100.0, channel_names=("Cz", "Pz"))
    engine = P300Engine(config)
    # Resolve MNE lazy exports, whose data files/imports must be in the bundle.
    assert callable(mne.EpochsArray) and callable(mne.filter.filter_data)
    for name in ALGORITHMS:
        make_pipeline(name, config)
    return {"ok": True, "versions": software_versions(), "liblsl": library_version(),
            "p300_status": engine.status, "scientific_validation": False}
