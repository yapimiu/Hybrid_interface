"""Одинаковая конечная обработка каждой эпохи online/offline, без fit по всем данным."""

from dataclasses import dataclass
from fractions import Fraction

import mne
import numpy as np
from scipy.signal import resample_poly

from .buffer import EpochRejected


@dataclass
class Epoch:
    data: np.ndarray  # channels × time, volts
    times: np.ndarray
    event: object
    source_interval: tuple[float, float]
    alignment_error_s: float


class EpochProcessor:
    def __init__(self, config):
        self.config = config
        self.start = round(config.tmin * config.sfreq)
        self.end = round(config.tmax * config.sfreq)
        self.pad = round(config.padding_s * config.sfreq)
        self.ratio = Fraction(config.feature_sfreq / config.sfreq).limit_denominator(10000)
        self.output_sfreq = config.sfreq * self.ratio.numerator / self.ratio.denominator
        if not np.isclose(self.output_sfreq, config.feature_sfreq, rtol=1e-8):
            raise ValueError("Не представимое рациональным отношением feature_sfreq")
        self.info = mne.create_info(list(config.channel_names), self.output_sfreq, "eeg")

    def extract(self, buffer, event):
        c = self.config
        raw, timestamps, alignment = buffer.window(
            event.timestamp, self.start - self.pad, self.end + self.pad,
            c.max_gap_samples, c.max_jitter_samples,
        )
        raw = raw.T * c.scale_to_volts
        if not np.all(np.isfinite(raw)):
            raise EpochRejected("NaN/Inf в EEG")
        core = raw[:, self.pad:self.pad + self.end - self.start + 1]
        ptp = np.ptp(core, axis=1)
        if np.any(ptp > c.reject_uv * 1e-6):
            raise EpochRejected("Превышение порога амплитуды")
        if np.any(ptp < c.flat_uv * 1e-6):
            raise EpochRejected("Плоский/отключённый канал")
        filtered = mne.filter.filter_data(
            raw, c.sfreq, c.l_freq, c.h_freq, method="iir",
            iir_params={"order": c.filter_order, "ftype": "butter", "output": "sos"},
            phase="zero", verbose=False,
        )
        # Resampling uses only this event's padded interval, never a global recording.
        filtered = resample_poly(filtered, self.ratio.numerator, self.ratio.denominator, axis=-1)
        padded_start = (self.start - self.pad) / c.sfreq
        sample_start = int(np.ceil((self.start / c.sfreq - padded_start) * self.output_sfreq - 1e-9))
        sample_end = int(np.floor((self.end / c.sfreq - padded_start) * self.output_sfreq + 1e-9))
        tmin = padded_start + sample_start / self.output_sfreq
        epochs = mne.EpochsArray(filtered[None, :, sample_start:sample_end + 1],
                                 self.info, tmin=tmin, baseline=None, verbose=False)
        epochs.apply_baseline(c.baseline, verbose=False)
        data = epochs.get_data(copy=True)[0]
        if not np.all(np.isfinite(data)) or np.any(np.ptp(data, axis=1) > c.reject_uv * 1e-6):
            raise EpochRejected("Артефакт после фильтрации")
        return Epoch(data, epochs.times.copy(), event,
                     (float(timestamps[0]), float(timestamps[-1])), alignment)


class ERPAverage:
    def __init__(self):
        self.sums = {}
        self.counts = {}
        self.times = None

    def add(self, epoch):
        key = epoch.event.object_id
        if key not in self.sums:
            self.sums[key] = np.zeros_like(epoch.data)
            self.counts[key] = 0
        self.sums[key] += epoch.data
        self.counts[key] += 1
        self.times = epoch.times.copy()

    def snapshot(self):
        return {key: {"count": self.counts[key], "data": value / self.counts[key]}
                for key, value in self.sums.items()}
