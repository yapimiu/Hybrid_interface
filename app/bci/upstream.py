"""Load standalone upstream protocol helpers without mixing the two app packages."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys


@lru_cache(maxsize=None)
def protocol_helper(name):
    if name not in ('experiment_queue', 'stim_control'):
        raise ValueError('Unsupported standalone helper')
    root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[2]))
    path = root / 'vendor/hybrid_bci/experiment_protocol' / (name + '.py')
    module_name = 'hybrid_upstream_' + name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
