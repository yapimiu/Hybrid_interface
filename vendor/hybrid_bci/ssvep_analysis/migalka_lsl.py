"""LSL-маркеры от Migalka (Arduino Due), совместимы с p300_analysis.marker_parsing."""

from __future__ import annotations

from typing import Optional

from pylsl import StreamInfo, StreamOutlet

STREAM_NAME = "MigalkaStimMarkers"
STREAM_TYPE = "Markers"
SOURCE_ID = "migalka-due-001"


class MigalkaLslSender:
    def __init__(self) -> None:
        self._outlet: Optional[StreamOutlet] = None
        self._init_stream()

    def _init_stream(self) -> None:
        try:
            info = StreamInfo(
                name=STREAM_NAME,
                type=STREAM_TYPE,
                channel_count=1,
                nominal_srate=0,
                channel_format="string",
                source_id=SOURCE_ID,
            )
            self._outlet = StreamOutlet(info)
        except Exception as e:
            print(f"LSL Migalka: не удалось создать стрим: {e}")
            self._outlet = None

    def close(self) -> None:
        """Освободить LSL outlet (иначе MigalkaStimMarkers виден после остановки протокола)."""
        self._outlet = None

    def send_lamp_event(self, lamp_index: int, event: str) -> None:
        """lamp_index 0..5 → маркер 100+index|on|off (как BCI_StimMarkers)."""
        if self._outlet is None:
            return
        if event not in ("on", "off"):
            return
        if not (0 <= lamp_index <= 8):
            return
        marker = f"{100 + lamp_index}|{event}"
        try:
            self._outlet.push_sample([marker])
        except Exception as e:
            print(f"LSL Migalka: ошибка отправки: {e}")
