"""Накопление LDA-оценок отдельно для каждого trial и объекта."""

from collections import OrderedDict
import math


class ScoreAccumulator:
    def __init__(self, min_repetitions=3, margin=0.0):
        self.min_repetitions = min_repetitions
        self.margin = margin
        self.trials = OrderedDict()

    def add(self, event, score):
        if not math.isfinite(score):
            raise ValueError("Неконечная оценка модели")
        key = (event.block_id, event.trial_id)
        state = self.trials.setdefault(key, {"objects": event.objects, "sums": {}, "counts": {}})
        if state["objects"] != event.objects:
            raise ValueError("Изменился набор объектов trial")
        self.trials.move_to_end(key)
        if len(self.trials) > 256:
            self.trials.popitem(last=False)
        obj = event.object_id
        state["sums"][obj] = state["sums"].get(obj, 0.0) + score
        state["counts"][obj] = state["counts"].get(obj, 0) + 1
        counts = {obj: state["counts"].get(obj, 0) for obj in event.objects}
        means = {obj: state["sums"][obj] / counts[obj] for obj in event.objects if counts[obj]}
        result = {"block_id": event.block_id, "trial_id": event.trial_id,
                  "event_id": event.event_id, "score": score, "counts": counts,
                  "mean_scores": means, "candidate": None, "status": "collecting"}
        if min(counts.values()) >= self.min_repetitions and len(set(counts.values())) == 1:
            ranked = sorted(means, key=means.get, reverse=True)
            difference = means[ranked[0]] - means[ranked[1]]
            result["margin"] = difference
            if difference > self.margin:
                result.update(candidate=ranked[0], status="candidate")
            else:
                result["status"] = "ambiguous"
        return result
