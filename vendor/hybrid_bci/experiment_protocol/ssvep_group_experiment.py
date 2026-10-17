"""SSVEP grouped experiment: aggregate sub-block (8s) results into one experiment record."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional


def aggregate_ssvep_group_experiment(
    subblocks: List[Dict[str, Any]],
    *,
    target_lamp_0idx: Optional[int],
) -> Dict[str, Any]:
    """Experiment-level summary from ``subblock_results`` (each ~8s logical slice)."""
    if not subblocks:
        return {}

    majors = [
        int(b["winner_majority_1based"])
        for b in subblocks
        if b.get("winner_majority_1based") is not None and not b.get("winner_majority_tied")
    ]
    group_maj: Optional[int] = None
    group_maj_tied = False
    if majors:
        counts = Counter(majors)
        top_val, top_cnt = counts.most_common(1)[0]
        group_maj_tied = sum(1 for c in counts.values() if c == top_cnt) > 1
        group_maj = None if group_maj_tied else int(top_val)

    stabilities: List[float] = []
    for b in subblocks:
        trace = b.get("msi_decision_trace") or {}
        summ = trace.get("summary") if isinstance(trace, dict) else {}
        if isinstance(summ, dict) and summ.get("stability_pct") not in (None, ""):
            try:
                stabilities.append(float(summ["stability_pct"]))
            except (TypeError, ValueError):
                pass

    correct_flags = [b.get("correct") for b in subblocks if b.get("correct") is not None]
    group_correct: Optional[bool] = None
    if target_lamp_0idx is not None and group_maj is not None and not group_maj_tied:
        group_correct = int(group_maj) == int(target_lamp_0idx) + 1
    elif correct_flags:
        group_correct = sum(1 for c in correct_flags if c) >= (len(correct_flags) + 1) // 2

    return {
        "experiment_target_lamp_0idx": target_lamp_0idx,
        "experiment_target_lamp_1based": int(target_lamp_0idx) + 1 if target_lamp_0idx is not None else None,
        "subblock_results": list(subblocks),
        "subblock_count": len(subblocks),
        "group_majority_1based": group_maj,
        "group_majority_tied": bool(group_maj_tied),
        "experiment_stability_pct": round(sum(stabilities) / len(stabilities), 2) if stabilities else None,
        "group_correct": group_correct,
        "ssvep_experiment_semantics": "grouped_lamp_session",
    }
