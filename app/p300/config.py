"""Явные, сериализуемые параметры обработки; амплитуды ядра — в вольтах."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class P300Config:
    sfreq: float
    channel_names: tuple[str, ...]
    input_unit: str = "uV"
    tmin: float = -0.2
    tmax: float = 0.8
    baseline: tuple[float, float] = (-0.2, 0.0)
    l_freq: float = 0.5
    h_freq: float = 20.0
    filter_order: int = 4
    padding_s: float = 1.0
    feature_sfreq: float = 50.0
    reject_uv: float = 150.0
    flat_uv: float = 0.05
    buffer_s: float = 30.0
    max_gap_samples: float = 1.5
    max_jitter_samples: float = 0.5
    max_pending: int = 1000
    max_calibration_epochs: int = 5000
    min_class_epochs: int = 5
    cv_folds: int = 3
    xdawn_components: int = 2
    min_repetitions: int = 3
    score_margin: float = 0.0
    random_state: int = 42

    def __post_init__(self):
        object.__setattr__(self, "channel_names", tuple(self.channel_names))
        object.__setattr__(self, "baseline", tuple(self.baseline))
        numeric = [getattr(self, key) for key in self.__dataclass_fields__
                   if key not in {"channel_names", "input_unit", "baseline"}]
        if not all(math.isfinite(v) for v in [*numeric, *self.baseline]):
            raise ValueError("Параметры должны быть конечными")
        if not self.channel_names or len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("Нужны уникальные имена выбранных EEG-каналов")
        if self.input_unit not in {"V", "mV", "uV"}:
            raise ValueError("Укажите единицы V, mV или uV")
        if not 0 < self.l_freq < self.h_freq < self.sfreq / 2:
            raise ValueError("Частоты фильтра должны лежать внутри (0, Nyquist)")
        if not 2 * self.h_freq < self.feature_sfreq <= self.sfreq:
            raise ValueError("feature_sfreq должна превышать 2*h_freq и быть <= sfreq")
        if not self.tmin < 0 < self.tmax:
            raise ValueError("Эпоха должна включать время до и после стимула")
        if not self.tmin <= self.baseline[0] < self.baseline[1] <= 0:
            raise ValueError("Базовая линия должна находиться до стимула внутри эпохи")
        if self.padding_s <= 0 or self.buffer_s <= self.tmax - self.tmin + 2 * self.padding_s:
            raise ValueError("Недостаточный буфер/padding")
        if not 0 < self.flat_uv < self.reject_uv:
            raise ValueError("Неверные пороги качества")
        if self.max_gap_samples <= 1 or self.max_jitter_samples <= 0 or self.score_margin < 0:
            raise ValueError("Неверные допуски времени/оценки")
        for key in ("filter_order", "max_pending", "max_calibration_epochs",
                    "min_class_epochs", "cv_folds", "xdawn_components", "min_repetitions"):
            value = getattr(self, key)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{key}: требуется положительное целое")
        if self.cv_folds < 2 or self.min_class_epochs < 2:
            raise ValueError("Для проверки нужны >=2 fold и >=2 эпохи каждого класса")

    @property
    def scale_to_volts(self):
        return {"V": 1.0, "mV": 1e-3, "uV": 1e-6}[self.input_unit]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)
