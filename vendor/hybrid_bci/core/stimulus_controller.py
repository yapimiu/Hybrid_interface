import random
from typing import Dict, List, Optional

from psychopy import core

from .grid import Grid
from .lsl import LslMarkerSender
from .tile import Tile


class StimulusController:
    """Конечный автомат: cue → ready → стимуляция (блоки вспышек плиток)."""

    def __init__(
        self,
        grid: Grid,
        flash_duration: float = 0.1,
        isi: float = 0.05,
        cue_duration: float = 2.0,
        ready_duration: float = 1.5,
        inter_block_s: float = 0.8,
        cue_color: str = "blue",
        stim_color: str = "white",
    ) -> None:
        self.grid = grid
        self.flash_duration = flash_duration
        self.isi = isi
        self.cue_duration = cue_duration
        self.ready_duration = ready_duration
        self.inter_block_s = inter_block_s
        self.cue_color = cue_color
        self.stim_color = stim_color
        self._clock = core.Clock()
        self._state = "idle"
        self._state_start = 0.0
        self._current_tile: Optional[Tile] = None
        self._running = False
        self._lsl = LslMarkerSender()
        self.sequences: int = 0
        self.target_tile: Optional[Tile] = None
        self._blocks_completed: int = 0
        self._current_block: List[int] = []
        self._block_index: int = 0
        self._substate: Optional[str] = None
        self._substate_start: float = 0.0

    @property
    def lsl(self) -> LslMarkerSender:
        return self._lsl

    def get_target_id(self) -> Optional[int]:
        return self.target_tile.id if self.target_tile else None

    def get_target_color(self) -> Optional[str]:
        if self._state == "cue" and self.target_tile:
            return self.cue_color
        return None

    def get_stim_color(self) -> str:
        return self.stim_color

    def is_running(self) -> bool:
        return self._running

    def start_experiment(self, sequences: int, target_tile_id: Optional[int] = None) -> None:
        if sequences <= 0:
            return
        self.sequences = sequences
        self._blocks_completed = 0
        if target_tile_id is not None and 0 <= target_tile_id < len(self.grid.tiles):
            self.target_tile = self.grid.tiles[target_tile_id]
        else:
            self.target_tile = random.choice(self.grid.tiles)
        self._running = True
        self._state = "cue"
        self._state_start = self._clock.getTime()
        self.grid.reset()
        self._substate = None

    def stop(self) -> None:
        self._running = False
        self.grid.reset()
        self._state = "idle"
        self.target_tile = None
        self._current_block = []
        self._block_index = 0
        self._substate = None

    def _generate_next_block(self) -> None:
        self._current_block = list(range(len(self.grid.tiles)))
        random.shuffle(self._current_block)
        self._block_index = 0

    def _ensure_stim_substate(self, now: float) -> None:
        if self._substate is None:
            self._substate = "isi"
            self._substate_start = now

    def _advance_block_or_finish(self, now: float) -> Optional[Dict[str, object]]:
        """Увеличить счётчик блоков или завершить trial; между раундами — пауза inter_block_s."""
        self._blocks_completed += 1
        if self._blocks_completed >= self.sequences:
            self._state = "end"
            self._substate = None
            self._running = False
            self.grid.reset()
            return {"event": "trial_end"}  # следующий тик: _state «end» → idle
        self._substate = "block_break"
        self._substate_start = now
        return None

    def _try_flash_next_tile(self, now: float) -> Optional[Dict[str, object]]:
        """Перейти к следующей вспышке в блоке или к следующему блоку."""
        if self._block_index >= len(self._current_block):
            return self._advance_block_or_finish(now)
        next_tile_id = self._current_block[self._block_index]
        self._block_index += 1
        self._current_tile = self.grid.tiles[next_tile_id]
        self._current_tile.active = True
        self._substate = "on"
        self._substate_start = now
        return {"tile_id": self._current_tile.id, "event": "on", "timestamp": now}

    def _update_cue(self, now: float) -> Optional[Dict[str, object]]:
        if now - self._state_start >= self.cue_duration:
            cue_target_id = self.target_tile.id if self.target_tile else -1
            self._state = "ready"
            self._state_start = now
            if self.target_tile:
                self.target_tile.active = False
            return {"event": "trial_start", "target": cue_target_id}
        if self.target_tile and not self.target_tile.active:
            self.target_tile.active = True
        return None

    def _update_ready(self, now: float) -> Optional[Dict[str, object]]:
        if now - self._state_start >= self.ready_duration:
            self._state = "stim"
            self._state_start = now
            self._generate_next_block()
            self._substate = None
        return None

    def _update_stim(self, now: float) -> Optional[Dict[str, object]]:
        self._ensure_stim_substate(now)
        if self._substate == "block_break":
            if now - self._substate_start >= self.inter_block_s:
                self._generate_next_block()
                return self._try_flash_next_tile(now)
            return None
        if self._substate == "isi":
            if now - self._substate_start >= self.isi:
                return self._try_flash_next_tile(now)
            return None
        if self._substate == "on":
            assert self._current_tile is not None
            if now - self._substate_start >= self.flash_duration:
                self._current_tile.active = False
                self._substate = "isi"
                self._substate_start = now
                return {
                    "tile_id": self._current_tile.id,
                    "event": "off",
                    "timestamp": now,
                }
        return None

    def _update_end(self) -> Optional[Dict[str, object]]:
        """Один тик после trial_end: перевод в idle (как в исходной логике)."""
        self._state = "idle"
        self._running = False
        return None

    def update(self) -> Optional[Dict[str, object]]:
        if self._state == "end":
            return self._update_end()
        if not self._running:
            return None
        now = self._clock.getTime()
        if self._state == "cue":
            return self._update_cue(now)
        if self._state == "ready":
            return self._update_ready(now)
        if self._state == "stim":
            return self._update_stim(now)
        return None
