from __future__ import annotations

from p300_analysis.analysis_profiles import (
    ANALYSIS_PROFILE_GENERAL,
    ANALYSIS_PROFILE_RECENT,
    CANONICAL_P300_ROI_0IDX,
    default_roi_channels_0idx,
    format_channels_1idx,
    get_analysis_profile,
)


def test_canonical_p300_montage() -> None:
    assert CANONICAL_P300_ROI_0IDX == (4, 5, 6, 7, 18, 19)
    assert format_channels_1idx(CANONICAL_P300_ROI_0IDX) == "5,6,7,8,19,20"


def test_general_profile_uses_canonical_montage() -> None:
    profile = get_analysis_profile(ANALYSIS_PROFILE_GENERAL)

    assert profile.baseline_ms == 100
    assert profile.window_x_ms == 550
    assert profile.window_y_ms == 725
    assert profile.roi_channels_0idx == CANONICAL_P300_ROI_0IDX


def test_recent_profile_same_roi_different_window() -> None:
    profile = get_analysis_profile(ANALYSIS_PROFILE_RECENT)

    assert profile.baseline_ms == 100
    assert profile.window_x_ms == 375
    assert profile.window_y_ms == 675
    assert profile.roi_channels_0idx == CANONICAL_P300_ROI_0IDX


def test_default_roi_channels_falls_back_to_all_if_profile_out_of_range() -> None:
    assert default_roi_channels_0idx(2, ANALYSIS_PROFILE_GENERAL) == [0, 1]
    assert default_roi_channels_0idx(6, ANALYSIS_PROFILE_RECENT) == [4, 5]


def test_format_channels_uses_one_based_indices() -> None:
    assert format_channels_1idx((3,)) == "4"
    assert format_channels_1idx(CANONICAL_P300_ROI_0IDX) == "5,6,7,8,19,20"
