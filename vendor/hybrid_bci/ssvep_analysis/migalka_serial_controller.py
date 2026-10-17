from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import serial

from ssvep_analysis.burst_gate import (
    parse_burst_done_line,
    parse_burst_target_ack,
    parse_led_serial_line,
)
from ssvep_analysis.migalka_event_queue import LampEvent, MigalkaEventQueue
from ssvep_analysis.migalka_lsl import MigalkaLslSender

try:
    from experiment_protocol.protocol_log import info as _log_info
except Exception:  # pragma: no cover

    def _log_info(msg: str) -> None:
        print(f"[migalka] {msg}", flush=True)


@dataclass(frozen=True)
class MigalkaConfig:
    port: str
    baudrate: int = 115200
    timeout_s: float = 0.1
    # Migalka protocol: "M <mode>" where 0=continuous, 1=burst (as in migalka.py)
    mode: int = 0
    freqs: Tuple[str, str, str, str, str, str] = ("0", "0", "0", "0", "0", "0")
    # Global burst ON-budget (ms), shared across lamps; sent as ``T <ms>`` after freqs in mode 1
    burst_target_on_ms: Optional[int] = None


_OFF6 = ("0", "0", "0", "0", "0", "0")

# Пауза после M1 до строки частот (прошивка должна успеть сменить mode).
_BURST_MODE_SETTLE_S = 0.35


class MigalkaSerialController:
    """Direct COM control for Arduino Due Migalka + optional LSL marker mirroring."""

    def __init__(
        self,
        *,
        mirror_lsl: bool = True,
        event_queue: Optional[MigalkaEventQueue] = None,
        on_event: Optional[Callable[[LampEvent], None]] = None,
    ) -> None:
        self._ser: Optional[serial.Serial] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._event_queue = event_queue
        self._on_event = on_event if event_queue is None else None
        self._lsl = MigalkaLslSender() if mirror_lsl else None
        self._lock = threading.RLock()
        self._last_mode: Optional[int] = None
        self._burst_done_pending: List[Dict[str, object]] = []
        self._burst_target_ack_ms: Optional[int] = None

    def is_open(self) -> bool:
        return self._ser is not None and bool(getattr(self._ser, "is_open", False))

    def _write_line(self, line: str) -> None:
        if not self.is_open():
            return
        assert self._ser is not None
        payload = (line.strip() + "\n").encode("utf-8")
        self._ser.write(payload)
        self._ser.flush()

    def _send_mode(self, mode: int) -> None:
        m = 0 if int(mode) <= 0 else 1
        self._write_line(f"M {m}")
        self._last_mode = int(m)

    def _send_freqs(self, freqs: Sequence[str]) -> None:
        vals = [str(x).strip().replace(",", ".") for x in freqs]
        if len(vals) != 6:
            raise ValueError("Migalka expects 6 frequency values")
        self._write_line(" ".join(vals))

    def _send_burst_target_ms(self, ms: int) -> None:
        self._write_line(f"T {int(ms)}")
        _log_info(f"burst global ON budget: {int(ms)} ms (shared across lamps)")

    def _apply_config(self, cfg: MigalkaConfig) -> None:
        """Постоянный: M0 + частоты (как migalka.py / MyForm).

        Пакетный: сначала 0×6 (гасит continuous-таймеры), затем M1, затем частоты.
        Без строки нулей между M1 и частотами — иначе в прошивке при M1 сразу
        перезапускаются старые targetFreq[] и лампы мигают как в постоянном режиме.
        """
        m = 0 if int(cfg.mode) <= 0 else 1
        if m == 1:
            self._send_freqs(_OFF6)
            time.sleep(0.15)
            self._send_mode(1)
            time.sleep(_BURST_MODE_SETTLE_S)
            self._send_freqs(cfg.freqs)
            if cfg.burst_target_on_ms is not None and int(cfg.burst_target_on_ms) > 0:
                time.sleep(0.08)
                self._send_burst_target_ms(int(cfg.burst_target_on_ms))
        else:
            self._send_mode(0)
            time.sleep(0.06)
            self._send_freqs(cfg.freqs)
        _log_info(f"apply: M {m}, freqs={' '.join(cfg.freqs)}")

    def halt_lamps(self) -> None:
        """Погасить все лампы (0×6), не закрывая порт — перед stop или перед пакетным."""
        with self._lock:
            if not self.is_open():
                return
            _log_info("halt_lamps: 0×6")
            self._send_freqs(_OFF6)
            time.sleep(0.35)

    def prepare_burst_handoff(self) -> None:
        """После continuous: M1 + нули, порт остаётся открыт (без close/reopen)."""
        with self._lock:
            if not self.is_open():
                return
            _log_info("handoff continuous->burst: 0×6, M1, 0×6 (порт открыт)")
            self._send_freqs(_OFF6)
            time.sleep(0.15)
            self._send_mode(1)
            time.sleep(0.2)
            self._send_freqs(_OFF6)
            time.sleep(0.15)

    def _ensure_reader_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._read_loop, name="MigalkaSerialReader", daemon=True)
        self._thread.start()

    def _open_port(self, cfg: MigalkaConfig) -> None:
        burst = int(cfg.mode) > 0
        _log_info(
            f"открываем COM {cfg.port!r} baud={cfg.baudrate}, mode={cfg.mode} "
            f"({'постоянный' if not burst else 'пакетный'})"
        )
        self._ser = serial.Serial(cfg.port, cfg.baudrate, timeout=cfg.timeout_s)
        self._running = True
        self._ensure_reader_thread()
        if burst:
            # После continuous порт закрыт, но таймеры на Due могут ещё крутиться.
            # Сразу гасим, без 0.5 с тишины в постоянном режиме.
            _log_info("пакетный: сразу 0×6 после открытия COM")
            self._send_freqs(_OFF6)
            time.sleep(0.2)
        else:
            time.sleep(0.5)

    def open_and_start(self, cfg: MigalkaConfig) -> None:
        with self._lock:
            if self.is_open():
                _log_info("COM уже открыт — только apply_config")
                self._apply_config(cfg)
                return
            self._open_port(cfg)
            self._apply_config(cfg)

    def reconfigure(self, cfg: MigalkaConfig) -> None:
        with self._lock:
            if not self.is_open():
                self._open_port(cfg)
            self._apply_config(cfg)

    def standby_burst_between_phases(self) -> None:
        """После последнего continuous-блока: M1, лампы выкл, порт остаётся открыт."""
        with self._lock:
            if not self.is_open():
                _log_info("standby_burst: порт закрыт")
                return
            _log_info("standby_burst: M1, лампы 0, без закрытия COM")
            self._send_mode(1)
            time.sleep(0.06)
            self._send_freqs(_OFF6)

    def stop_and_close(self, *, join_timeout_s: float = 2.0) -> None:
        with self._lock:
            _log_info("stop_and_close")
            self._running = False
            try:
                if self.is_open():
                    self._send_freqs(_OFF6)
                    time.sleep(0.35)
            except Exception:
                pass
            try:
                if self._ser is not None:
                    self._ser.close()
            except Exception:
                pass
            self._ser = None
            self._last_mode = None
            if self._lsl is not None:
                self._lsl.close()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(join_timeout_s)))
        self._thread = None

    def set_mode(self, mode: int) -> None:
        with self._lock:
            self._send_mode(mode)

    def set_freqs(self, freqs: Sequence[str]) -> None:
        with self._lock:
            self._send_freqs(freqs)

    def drain_pending_events(self) -> Tuple[List[LampEvent], Dict[str, float | int]]:
        """Main thread: dequeue COM events and mirror to LSL (if enabled)."""
        if self._event_queue is None:
            return [], {"batch": 0, "pending_after": 0}
        events, stats = self._event_queue.drain()
        if self._lsl is not None:
            for lamp, is_on, _raw in events:
                self._lsl.send_lamp_event(int(lamp), "on" if is_on else "off")
        return events, stats

    def pending_event_count(self) -> int:
        if self._event_queue is None:
            return 0
        return self._event_queue.pending_count()

    def drain_burst_status(self) -> List[Dict[str, object]]:
        """Main thread: BURST DONE / TARGET ack from firmware."""
        with self._lock:
            out = list(self._burst_done_pending)
            self._burst_done_pending.clear()
            return out

    def last_burst_target_ack_ms(self) -> Optional[int]:
        with self._lock:
            return self._burst_target_ack_ms

    def _emit_event(self, lamp: int, is_on: bool, raw_line: str) -> None:
        ev: LampEvent = (int(lamp), bool(is_on), str(raw_line))
        if self._event_queue is not None:
            self._event_queue.enqueue(ev)
            return
        if self._lsl is not None:
            self._lsl.send_lamp_event(int(lamp), "on" if is_on else "off")
        if self._on_event is not None:
            try:
                self._on_event(ev)
            except Exception:
                pass

    def _read_loop(self) -> None:
        while True:
            if not self._running or not self.is_open():
                return
            try:
                assert self._ser is not None
                line = self._ser.readline().decode(errors="replace").strip()
                if not line:
                    continue
                done = parse_burst_done_line(line)
                if done is not None:
                    with self._lock:
                        self._burst_done_pending.append(done)
                    continue
                ack_ms = parse_burst_target_ack(line)
                if ack_ms is not None:
                    with self._lock:
                        self._burst_target_ack_ms = int(ack_ms)
                    continue
                parsed = parse_led_serial_line(line)
                if parsed is None:
                    continue
                lamp, is_on = parsed
                self._emit_event(lamp, is_on, line)
            except Exception:
                time.sleep(0.01)
