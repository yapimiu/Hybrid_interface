"""Shared analysis profiles for online and offline P300 runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class AnalysisProfile:
    key: str
    title: str
    button_label: str
    description: str
    baseline_ms: int
    window_x_ms: int
    window_y_ms: int
    roi_channels_0idx: Tuple[int, ...]


ANALYSIS_PROFILE_GENERAL = "general"
ANALYSIS_PROFILE_RECENT = "recent"

# Canonical hybrid-BCI montage (1-based UI labels: 5–8, 19–20).
CANONICAL_P300_ROI_0IDX: Tuple[int, ...] = (4, 5, 6, 7, 18, 19)

_PROFILES: Dict[str, AnalysisProfile] = {
    ANALYSIS_PROFILE_GENERAL: AnalysisProfile(
        key=ANALYSIS_PROFILE_GENERAL,
        title="Общий профиль (рекомендуется)",
        button_label="Общий",
        description=(
            "Каноническая montage: ROI 5–8, 19–20; окно 550–725 мс "
            "(legacy standalone default был ch 4 only)."
        ),
        baseline_ms=100,
        window_x_ms=550,
        window_y_ms=725,
        roi_channels_0idx=CANONICAL_P300_ROI_0IDX,
    ),
    ANALYSIS_PROFILE_RECENT: AnalysisProfile(
        key=ANALYSIS_PROFILE_RECENT,
        title="Профиль последних 7 сессий",
        button_label="Последние 7",
        description=(
            "Та же каноническая montage; альтернативное окно 375–675 мс "
            "(раньше ROI был только 5–6)."
        ),
        baseline_ms=100,
        window_x_ms=375,
        window_y_ms=675,
        roi_channels_0idx=CANONICAL_P300_ROI_0IDX,
    ),
}

DEFAULT_ANALYSIS_PROFILE_KEY = ANALYSIS_PROFILE_GENERAL


def get_analysis_profile(key: str = DEFAULT_ANALYSIS_PROFILE_KEY) -> AnalysisProfile:
    return _PROFILES.get(key, _PROFILES[DEFAULT_ANALYSIS_PROFILE_KEY])


def hybrid_protocol_p300_profile(
    key: str = DEFAULT_ANALYSIS_PROFILE_KEY,
) -> AnalysisProfile:
    """P300 defaults for integrated hybrid protocol (same profile as standalone analyzer)."""
    return get_analysis_profile(key)


def default_roi_channels_0idx(
    count: int,
    profile_key: str = DEFAULT_ANALYSIS_PROFILE_KEY,
) -> List[int]:
    if count <= 0:
        return []
    preferred = [ch for ch in get_analysis_profile(profile_key).roi_channels_0idx if 0 <= ch < count]
    return preferred if preferred else list(range(count))


def format_channels_1idx(channels_0idx: Sequence[int]) -> str:
    return ",".join(str(int(ch) + 1) for ch in channels_0idx)
