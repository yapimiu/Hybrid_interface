"""Строгий протокол JSON-маркеров внешнего визуального стимулятора."""

from dataclasses import asdict, dataclass
import json
import math


@dataclass(frozen=True)
class Stimulus:
    event_id: str
    object_id: str
    trial_id: str
    block_id: str
    objects: tuple[str, ...]
    timestamp: float
    source_timestamp: float
    clock_offset: float
    target: bool | None = None

    def __post_init__(self):
        object.__setattr__(self, "objects", tuple(self.objects))
        for key in ("event_id", "object_id", "trial_id", "block_id"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"Маркер: требуется непустой {key}")
        if (not 2 <= len(self.objects) <= 256 or len(set(self.objects)) != len(self.objects)
                or not all(isinstance(x, str) and x for x in self.objects)
                or self.object_id not in self.objects):
            raise ValueError("Маркер: objects должен содержать >=2 уникальных ID, включая object_id")
        if self.target is not None and not isinstance(self.target, bool):
            raise ValueError("Маркер: target должен быть true/false или отсутствовать")
        if not all(math.isfinite(x) for x in
                   (self.timestamp, self.source_timestamp, self.clock_offset)):
            raise ValueError("Неконечное время маркера")
        if not math.isclose(self.timestamp, self.source_timestamp + self.clock_offset, abs_tol=1e-7):
            raise ValueError("Несогласованная коррекция времени маркера")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def parse(cls, payload: str, source_timestamp: float, clock_offset: float):
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("Маркер должен быть JSON-объектом")
        required = {"event_id", "object_id", "trial_id", "block_id", "objects"}
        if not required <= value.keys() or value.keys() - required - {"target"}:
            raise ValueError("Неверные поля маркера")
        if not isinstance(value["objects"], list):
            raise ValueError("objects должен быть массивом ID")
        return cls(**value, timestamp=source_timestamp + clock_offset,
                   source_timestamp=source_timestamp, clock_offset=clock_offset)
