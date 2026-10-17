"""Константы онлайн P300-анализатора (LSL, эпохи, UI-метки)."""

EPOCH_DURATION_MS = 800
EPOCH_RESERVE_MS = 50
EEG_KEEP_SECONDS = 10.0
# Minimum epochs per stimulus class before allowing a winner (uses all accumulated epochs).
SAFE_MIN_EPOCHS_TO_DECIDE = 5
# Typical trial length (stimulator sequences); informational — not a hard gate.
IDEAL_EPOCHS_PER_CLASS = 12
# Back-compat alias for UI text tied to protocol sequences.
MIN_EPOCHS_TO_DECIDE = IDEAL_EPOCHS_PER_CLASS
MARKERS_PULL_MAX_SAMPLES = 256
EEG_PULL_MAX_SAMPLES = 2048

SIMULATOR_NAME = "EEG_Simulator"
SIMULATOR_SOURCE_ID = "eeg-simulator-neurospectr"
NEUROSPECTR_MARKER = "neuro"
EEG_STREAM_TYPES = ("EEG", "Signal")

MONITOR_EEG_PLOT_MAX = 2500
MONITOR_MARKER_EVENTS_MAX = 120
MONITOR_LOG_INTERVAL_S = 2.0

WINNER_LABEL_STYLE_IDLE = (
    "QLabel { background-color: #1a1a1a; color: #4dff88; font-size: 18px; "
    "font-weight: bold; padding: 15px; border: 2px solid #333; border-radius: 5px; }"
)

WINNER_LABEL_STYLE_MISMATCH = (
    "QLabel { background-color: #2a1f0a; color: #ffcc66; font-size: 18px; "
    "font-weight: bold; padding: 15px; border: 2px solid #cc8800; border-radius: 5px; }"
)

WINNER_LABEL_STYLE_COLLECTING = (
    "QLabel { background-color: #1a1a1a; color: #ffcc33; font-size: 16px; font-weight: bold; "
    "padding: 15px; border: 2px solid #333; border-radius: 5px; }"
)

WINNER_LABEL_STYLE_MATCH = (
    "QLabel { background-color: #0d2614; color: #4dff88; font-size: 20px; font-weight: bold; "
    "padding: 15px; border: 2px solid #28a745; border-radius: 5px; }"
)
