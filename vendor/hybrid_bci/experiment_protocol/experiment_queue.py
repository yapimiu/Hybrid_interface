"""Сборка очередей для ручного запуска блоков (P300 / SSVEP continuous / SSVEP burst)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


SSVEP_DEFAULT_BLOCKS_PER_MODE = 16
SSVEP_DEFAULT_EXPERIMENTS_PER_MODE = 4
SSVEP_DEFAULT_GROUP_SIZE = 4


def ssvep_physical_blocks_per_mode(
    experiments_per_mode: int,
    *,
    group_size: int = SSVEP_DEFAULT_GROUP_SIZE,
    physical_blocks_override: int = 0,
) -> int:
    """Logical queue items = experiments × sub-blocks per experiment (default 4×4=16)."""
    explicit = int(physical_blocks_override)
    if explicit > 0:
        return explicit
    return max(0, int(experiments_per_mode)) * max(1, int(group_size))


@dataclass(frozen=True)
class QueueItem:
    kind: str  # "p300" | "ssvep"
    ssvep_mode: Optional[str] = None  # "continuous" | "burst"
    target_lamp_0idx: Optional[int] = None  # для SSVEP
    target_tile_id: Optional[int] = None  # для P300 (0..8), назначается при сборке очереди
    phase: str = "main"  # "calib" | "main"
    # SSVEP grouped-random queue metadata (block_in_group==0 → новая группа / instruction overlay)
    ssvep_group_index: Optional[int] = None
    ssvep_block_in_group: Optional[int] = None
    ssvep_group_size: int = SSVEP_DEFAULT_GROUP_SIZE

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "kind": self.kind,
            "ssvep_mode": self.ssvep_mode,
            "target_lamp_0idx": self.target_lamp_0idx,
            "target_tile_id": self.target_tile_id,
            "phase": self.phase,
        }
        if self.kind == "ssvep" and self.ssvep_group_index is not None:
            lamp = self.target_lamp_0idx
            out.update(
                {
                    "group_index": int(self.ssvep_group_index),
                    "group_target": int(lamp) + 1 if lamp is not None else None,
                    "block_in_group": int(self.ssvep_block_in_group or 0),
                    "group_size": int(self.ssvep_group_size),
                }
            )
        return out

    @property
    def ssvep_group_start(self) -> bool:
        return self.kind == "ssvep" and int(self.ssvep_block_in_group or 0) == 0


def _balanced_lamp_indices(n_items: int, n_lamps: int) -> List[int]:
    """Распределить n_items по лампам 0..n_lamps-1 максимально равномерно."""
    n_lamps = max(1, int(n_lamps))
    n_items = max(0, int(n_items))
    per = n_items // n_lamps
    rem = n_items % n_lamps
    out: List[int] = []
    for lamp in range(n_lamps):
        out.extend([lamp] * (per + (1 if lamp < rem else 0)))
    return out


def _max_same_lamp_run(lamps: Sequence[int]) -> int:
    if not lamps:
        return 0
    best = 1
    run = 1
    for i in range(1, len(lamps)):
        if int(lamps[i]) == int(lamps[i - 1]):
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def _shuffled_balanced_lamp_indices(
    n_items: int,
    n_lamps: int,
    rng: random.Random,
    *,
    max_run: int = 2,
) -> List[int]:
    """
    Сбалансированная последовательность ламп (4+4+4+3 для 15×4) в случайном порядке.

    Ограничение: не более ``max_run`` одинаковых ламп подряд (по умолчанию 2).
    """
    n_items = max(0, int(n_items))
    if n_items == 0:
        return []
    counts = [0] * max(1, int(n_lamps))
    for lamp in _balanced_lamp_indices(n_items, n_lamps):
        counts[int(lamp)] += 1

    lamps_order = list(range(len(counts)))

    def _backtrack(out: List[int], prev: Optional[int], streak: int) -> bool:
        if len(out) >= n_items:
            return True
        order = list(lamps_order)
        rng.shuffle(order)
        for lamp in order:
            if counts[lamp] <= 0:
                continue
            if prev is not None and streak >= int(max_run) and int(lamp) == int(prev):
                continue
            counts[lamp] -= 1
            out.append(int(lamp))
            next_streak = streak + 1 if prev is not None and int(lamp) == int(prev) else 1
            if _backtrack(out, int(lamp), next_streak):
                return True
            out.pop()
            counts[lamp] += 1
        return False

    result: List[int] = []
    if not _backtrack(result, None, 0):
        raise RuntimeError(
            f"cannot build SSVEP lamp queue: n_items={n_items}, n_lamps={n_lamps}, max_run={max_run}"
        )
    return result


def _assign_p300_tile_targets(items: List[QueueItem], *, n_tiles: int, rng: random.Random) -> List[QueueItem]:
    n_tiles = max(1, int(n_tiles))
    p300_idx = [i for i, x in enumerate(items) if x.kind == "p300"]
    tiles = _balanced_lamp_indices(len(p300_idx), n_tiles)
    rng.shuffle(tiles)
    out = list(items)
    for i, tile in zip(p300_idx, tiles):
        old = out[i]
        out[i] = QueueItem(
            kind=old.kind,
            ssvep_mode=old.ssvep_mode,
            target_lamp_0idx=old.target_lamp_0idx,
            target_tile_id=int(tile),
            phase=old.phase,
        )
    return out


def build_p300_block_queue(
    *,
    p300_calib_trials: int,
    calib_target_tile_id: int,
    p300_main_trials: int,
    n_p300_tiles: int = 9,
    shuffle_seed: Optional[int] = None,
) -> Tuple[List[QueueItem], int]:
    """Калибровка подряд, затем main P300 (плитки равномерно, без ССВП)."""
    items: List[QueueItem] = [
        QueueItem(
            kind="p300",
            target_tile_id=int(calib_target_tile_id),
            phase="calib",
        )
        for _ in range(int(p300_calib_trials))
    ]
    items.extend(QueueItem(kind="p300", phase="main") for _ in range(int(p300_main_trials)))
    seed_used = int(shuffle_seed) if shuffle_seed is not None else random.randrange(1 << 31)
    rng = random.Random(seed_used)
    main_idx = [i for i, x in enumerate(items) if x.kind == "p300" and x.phase == "main"]
    tiles = _balanced_lamp_indices(len(main_idx), int(n_p300_tiles))
    rng.shuffle(tiles)
    out = list(items)
    for i, tile in zip(main_idx, tiles):
        old = out[i]
        out[i] = QueueItem(
            kind=old.kind,
            ssvep_mode=old.ssvep_mode,
            target_lamp_0idx=old.target_lamp_0idx,
            target_tile_id=int(tile),
            phase=old.phase,
        )
    return out, seed_used


def default_ssvep_experiment_lamps(n_experiments: int, n_active_lamps: int) -> Tuple[int, ...]:
    """По умолчанию L1, L2, L3, L4, … (0-based), с wrap при n_experiments > n_lamps."""
    n_lamps = max(1, int(n_active_lamps))
    n_exp = max(0, int(n_experiments))
    return tuple(i % n_lamps for i in range(n_exp))


def normalize_ssvep_experiment_lamps(
    lamps: Sequence[int],
    *,
    n_experiments: int,
    n_active_lamps: int,
) -> Tuple[int, ...]:
    """Обрезать/дополнить до ``n_experiments``; clamp lamp indices."""
    n_lamps = max(1, int(n_active_lamps))
    n_exp = max(0, int(n_experiments))
    out: List[int] = []
    for i in range(n_exp):
        if i < len(lamps):
            out.append(int(lamps[i]) % n_lamps)
        else:
            out.append(i % n_lamps)
    return tuple(out)


def shuffle_ssvep_experiment_lamps(
    n_experiments: int,
    n_active_lamps: int,
    rng: random.Random,
) -> Tuple[int, ...]:
    """Случайная перестановка без повторов, если экспериментов ≤ числа ламп; иначе с повторами."""
    n_lamps = max(1, int(n_active_lamps))
    n_exp = max(0, int(n_experiments))
    if n_exp <= n_lamps:
        pool = list(range(n_lamps))
        rng.shuffle(pool)
        return tuple(pool[:n_exp])
    return tuple(rng.randrange(n_lamps) for _ in range(n_exp))


def is_ssvep_experiment_lamp_permutation(
    lamps: Sequence[int],
    *,
    n_experiments: int,
    n_active_lamps: int,
) -> bool:
    """True, если первые ``n_experiments`` ламп — попарно разные (cue sequence без повторов)."""
    n_lamps = max(1, int(n_active_lamps))
    n_exp = max(0, int(n_experiments))
    if n_exp == 0:
        return True
    if n_exp > n_lamps:
        return False
    seq = normalize_ssvep_experiment_lamps(
        lamps, n_experiments=n_exp, n_active_lamps=n_lamps
    )
    return len(set(seq)) == n_exp


def ensure_ssvep_experiment_lamp_permutation(
    lamps: Sequence[int],
    *,
    n_experiments: int,
    n_active_lamps: int,
    rng: random.Random,
) -> Tuple[int, ...]:
    """
    Cue/target sequence: каждый experiment — уникальная лампа, если n_exp ≤ n_lamps.

    Валидная ручная последовательность сохраняется; при дубликатах — детерминированный shuffle.
    """
    n_lamps = max(1, int(n_active_lamps))
    n_exp = max(0, int(n_experiments))
    normalized = normalize_ssvep_experiment_lamps(
        lamps, n_experiments=n_exp, n_active_lamps=n_lamps
    )
    if is_ssvep_experiment_lamp_permutation(
        normalized, n_experiments=n_exp, n_active_lamps=n_lamps
    ):
        return normalized
    return shuffle_ssvep_experiment_lamps(n_exp, n_lamps, rng)


def build_manual_ssvep_schedule(
    experiment_lamps_0idx: Sequence[int],
    *,
    blocks_per_group: int = SSVEP_DEFAULT_GROUP_SIZE,
) -> List[Tuple[int, int, int, int]]:
    """
    Ручная последовательность: один experiment = одна лампа × ``blocks_per_group`` sub-blocks.
    Returns (lamp_0idx, group_index, block_in_group, group_size).
    """
    bpg = max(1, int(blocks_per_group))
    out: List[Tuple[int, int, int, int]] = []
    for gi, lamp in enumerate(experiment_lamps_0idx):
        lamp_i = int(lamp)
        for bi in range(bpg):
            out.append((lamp_i, int(gi), int(bi), int(bpg)))
    return out


def _ssvep_items_from_schedule(
    *,
    mode: str,
    schedule: Sequence[Tuple[int, int, int, int]],
) -> List[QueueItem]:
    return [
        QueueItem(
            kind="ssvep",
            ssvep_mode=str(mode),
            target_lamp_0idx=int(lamp),
            phase="main",
            ssvep_group_index=int(gi),
            ssvep_block_in_group=int(bi),
            ssvep_group_size=int(gs),
        )
        for lamp, gi, bi, gs in schedule
    ]


def build_ssvep_mode_queue(
    *,
    mode: str,
    experiment_lamps_0idx: Sequence[int],
    blocks_per_group: int = SSVEP_DEFAULT_GROUP_SIZE,
) -> List[QueueItem]:
    """ССВП одного режима: ручная последовательность ламп (experiment → group × sub-blocks)."""
    schedule = build_manual_ssvep_schedule(
        experiment_lamps_0idx, blocks_per_group=int(blocks_per_group)
    )
    return _ssvep_items_from_schedule(mode=str(mode), schedule=schedule)


def build_main_queue(
    *,
    p300_trials: int,
    ssvep_continuous_lamps: Sequence[int],
    ssvep_burst_lamps: Sequence[int],
    n_p300_tiles: int = 9,
    shuffle_seed: Optional[int] = None,
    blocks_per_group: int = SSVEP_DEFAULT_GROUP_SIZE,
) -> Tuple[List[QueueItem], int]:
    """
    Собрать и перемешать очередь основного блока.

    SSVEP: лампы задаются явно (``ssvep_*_lamps``); P300 tiles — shuffle по seed.
    Returns (items, seed_used).
    """
    seed_used = int(shuffle_seed) if shuffle_seed is not None else random.randrange(1 << 31)
    rng = random.Random(seed_used)
    items: List[QueueItem] = []
    for _ in range(int(p300_trials)):
        items.append(QueueItem(kind="p300"))
    cont_sched = build_manual_ssvep_schedule(
        ssvep_continuous_lamps, blocks_per_group=int(blocks_per_group)
    )
    items.extend(_ssvep_items_from_schedule(mode="continuous", schedule=cont_sched))
    burst_sched = build_manual_ssvep_schedule(
        ssvep_burst_lamps, blocks_per_group=int(blocks_per_group)
    )
    items.extend(_ssvep_items_from_schedule(mode="burst", schedule=burst_sched))
    rng.shuffle(items)
    items = _assign_p300_tile_targets(items, n_tiles=int(n_p300_tiles), rng=rng)
    return items, seed_used


def queue_summary(items: Sequence[QueueItem]) -> Dict[str, int]:
    n_p300 = sum(1 for x in items if x.kind == "p300")
    n_cont = sum(1 for x in items if x.kind == "ssvep" and x.ssvep_mode == "continuous")
    n_burst = sum(1 for x in items if x.kind == "ssvep" and x.ssvep_mode == "burst")
    return {"p300": n_p300, "ssvep_continuous": n_cont, "ssvep_burst": n_burst, "total": len(items)}
