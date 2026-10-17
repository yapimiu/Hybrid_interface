from typing import Optional
from pylsl import StreamInfo, StreamOutlet

class LslMarkerSender:
    STREAM_NAME = 'BCI_StimMarkers'
    STREAM_TYPE = 'Markers'
    SOURCE_ID = 'stimulus-controller-001'

    def __init__(self) -> None:
        self._outlet: Optional[StreamOutlet] = None
        self._init_stream()

    def _init_stream(self) -> None:
        try:
            info = StreamInfo(name=self.STREAM_NAME, type=self.STREAM_TYPE, channel_count=1, nominal_srate=0, channel_format='string', source_id=self.SOURCE_ID)
            self._outlet = StreamOutlet(info)
        except Exception as e:
            print(f'LSL: не удалось создать стрим: {e}')
            self._outlet = None

    def send(self, tile_id: int, event: str) -> None:
        if self._outlet is None:
            return
        # Avoid conflict: 0 in exported continuous CSV means "no active flash".
        # Encode stimulus tile ids as 100..108, keep service ids negative.
        marker_id = (100 + tile_id) if tile_id >= 0 else tile_id
        marker = f'{marker_id}|{event}'
        try:
            self._outlet.push_sample([marker])
        except Exception as e:
            print(f'LSL: ошибка отправки маркера: {e}')
