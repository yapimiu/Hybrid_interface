"""Point-wise summation of flash epochs per object (numeric fixtures only)."""
import csv

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
from pylsl import StreamInfo

from app.bci.erp_sums import (format_numbers, format_unified, object_numbers, object_sums,
                              save_sums, unified_array)
from app.bci.panel import P300Panel


@pytest.fixture(scope='module')
def app():
    yield QApplication.instance() or QApplication([])


def test_sums_each_sample_across_flashes_per_channel():
    # Object 2 flashes three times; object 0 twice. Two channels, four samples.
    a = np.arange(8.).reshape(4, 2)
    epochs = {'стимул_2': [a, a * 10, a + 1], 'стимул_0': [np.ones((4, 2)), np.full((4, 2), 2.)]}
    result = object_sums(epochs, [-100, 0, 100, 200], ['Cz', 'Pz'])
    assert result.objects.tolist() == [0, 2]  # sorted by tile number
    assert result.counts.tolist() == [2, 3]
    assert result.sums.shape == (2, 4, 2)
    np.testing.assert_array_equal(result.sums[0], np.full((4, 2), 3.))
    np.testing.assert_array_equal(result.sums[1], a + a * 10 + a + 1)
    assert result.sums[1, 0, 0] == a[0, 0] + 10 * a[0, 0] + a[0, 0] + 1  # first point of each flash
    np.testing.assert_allclose(result.means[1], result.sums[1] / 3)
    assert result.time_ms.tolist() == [-100, 0, 100, 200]


def test_single_channel_epochs_become_one_column():
    result = object_sums({'стимул_5': [np.array([1., 2.]), np.array([3., 4.])]}, [0, 4], ['Cz'])
    np.testing.assert_array_equal(result.sums, [[[4.], [6.]]])


def test_empty_and_inconsistent_inputs_are_diagnosed():
    assert object_sums({}, None, ['Cz']).sums.shape == (0, 0, 1)
    assert format_unified(object_sums({}, None, [])) == ''
    assert object_numbers(object_sums({}, None, []), (0, 100))[1] is None
    with pytest.raises(ValueError, match='ROI'):
        object_sums({'стимул_0': [np.zeros((4, 2))], 'стимул_1': [np.zeros((4, 3))]}, None, [])


def test_unified_array_puts_all_objects_side_by_side():
    epochs = {'стимул_1': [np.full((3, 2), 1.), np.full((3, 2), 2.)], 'стимул_4': [np.arange(6.).reshape(3, 2)]}
    result = object_sums(epochs, [0., 4., 8.], ['Cz', 'Pz'])
    array, columns = unified_array(result)
    assert columns == ['1_Cz', '1_Pz', '4_Cz', '4_Pz']
    assert array.shape == (3, 4)
    np.testing.assert_array_equal(array[:, :2], result.sums[0])
    np.testing.assert_array_equal(array[:, 2:], result.sums[1])
    assert format_unified(result).splitlines()[0] == 't, мс\t1_Cz\t1_Pz\t4_Cz\t4_Pz'


def test_one_number_per_object_uses_baseline_and_window():
    time_ms = np.array([-100., -50., 0., 300., 400., 600.])
    flat = np.full((6, 2), 10.)            # constant offset only: number 0 after baseline
    bump = flat.copy()
    bump[3:5] += 6.                        # +6 µV inside 250–500 ms on both channels
    result = object_sums({'стимул_0': [flat, flat], 'стимул_7': [bump, bump, bump]}, time_ms, ['Cz', 'Pz'])
    numbers, winner = object_numbers(result, (250, 500))
    np.testing.assert_allclose(numbers, [0., 6.])  # mean of sums / flashes, not raw sum
    assert winner == 7
    assert '← максимум' in format_numbers(result, (250, 500)).splitlines()[2]
    equal, none = object_numbers(object_sums({'стимул_0': [flat], 'стимул_1': [flat]}, time_ms, []), (250, 500))
    assert none is None  # identical numbers: no maximum is invented


def test_saved_files_match_both_variants(tmp_path):
    epochs = {'стимул_1': [np.ones((3, 2)), np.ones((3, 2))], 'стимул_4': [np.eye(3, 2)]}
    result = object_sums(epochs, [-4., 4., 8.], ['Cz', 'Pz'])
    numbers, winner = save_sums(result, tmp_path, (0, 10))
    with np.load(tmp_path / 'erp_sums.npz', allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['sums'], result.sums)
        np.testing.assert_array_equal(archive['unified'], unified_array(result)[0])
        np.testing.assert_array_equal(archive['numbers'], numbers)
        assert archive['channels'].tolist() == ['Cz', 'Pz']
    with (tmp_path / 'erp_unified.csv').open(encoding='utf-8-sig') as file:
        rows = list(csv.reader(file))
    assert rows[0] == ['time_ms', '1_Cz', '1_Pz', '4_Cz', '4_Pz']
    assert [float(v) for v in rows[1]] == [-4., 2., 2., 1., 0.]
    with (tmp_path / 'erp_numbers.csv').open(encoding='utf-8-sig') as file:
        rows = list(csv.reader(file))
    assert rows[0] == ['object', 'flashes', 'number_uv', 'is_max']
    assert [int(r[1]) for r in rows[1:]] == [2, 1]
    assert sum(int(r[3]) for r in rows[1:]) == (winner is not None)


def test_panel_view_shows_live_roi_sums(app):
    panel = P300Panel()
    try:
        panel.set_stream(StreamInfo('test', 'EEG', 3, 250, 'double64', 'sums-test'), ['Fz', 'Cz', 'Pz'])
        for index, checkbox in enumerate(panel.channel_checkboxes):
            checkbox.setChecked(index != 0)
        panel.epochs_data = {'стимул_3': [np.ones((5, 2)), np.ones((5, 2))]}
        result = panel.sums_view.refresh()
        assert result.channels == ['Cz', 'Pz']
        np.testing.assert_array_equal(result.sums, np.full((1, 5, 2), 2.))
        assert panel.sums_view.text.toPlainText().splitlines()[0] == 't, мс\t3_Cz\t3_Pz'
        assert '5 × 2' in panel.sums_view.label.text()
        panel.sums_view.variant.setCurrentIndex(1)
        assert panel.sums_view.text.toPlainText().splitlines()[1].startswith('3\t2\t')
        panel.epochs_data = {}
        assert panel.sums_view.refresh().sums.shape[0] == 0
        assert panel.sums_view.text.toPlainText() == ''
    finally:
        panel.begin_shutdown()
