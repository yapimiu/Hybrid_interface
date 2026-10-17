"""Offline import and numerical smoke check for the packaged application."""
import importlib.metadata
import numpy as np
from pylsl import library_version
from p300_analysis.signal_processing import bandpass_filter
from p300_analysis.online_engine import P300OnlineEngine
from app.bci.panel import P300Panel
from validation.p300_synthetic import run_p300_synthetic_battery
from app.bci.experiment import ExperimentPlan


def self_check():
    values = bandpass_filter(np.zeros((500, 2)), 250)
    assert values.shape == (500, 2) and np.isfinite(values).all()
    P300OnlineEngine()
    validation = run_p300_synthetic_battery(seed=0)
    assert validation['summary']['both_decide'] == validation['summary']['trials']
    plan = ExperimentPlan('self-check', trials=9).targets()
    assert sorted(plan) == list(range(9))
    return {'ok': True, 'lsl': library_version(), 'engine': 'p300_analysis', 'unified_plan': plan, 'synthetic_validation': validation,
            'versions': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pylsl')}}
