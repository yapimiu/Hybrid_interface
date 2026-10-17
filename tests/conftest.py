import os
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('HYBRID_BCI_DATA_DIR', str(Path(__file__).resolve().parents[1] / 'build' / 'test-output'))
# Upstream tests install pylsl stubs only when it has not been imported.
# Integration tests must exercise the real library.
import pylsl
