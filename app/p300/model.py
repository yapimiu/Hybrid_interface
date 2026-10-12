"""Калибровка с GroupKFold, удалением пересечений и переносимыми числовыми моделями."""

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

from mne.decoding import Vectorizer, XdawnTransformer
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import P300Config


ALGORITHMS = ("shrinkage_lda", "xdawn_lda")


def software_versions():
    return {"python": platform.python_version(), **{
        name: version(name) for name in ("numpy", "scipy", "mne", "scikit-learn", "pylsl")}}


def make_pipeline(algorithm, config):
    if algorithm not in ALGORITHMS:
        raise ValueError("Неизвестный алгоритм")
    steps = []
    if algorithm == "xdawn_lda":
        steps.append(("xdawn", XdawnTransformer(
            n_components=min(config.xdawn_components, len(config.channel_names)), reg="oas")))
    steps.extend([("vectorize", Vectorizer()), ("scale", StandardScaler()),
                  ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto",
                                                     priors=[0.5, 0.5]))])
    return Pipeline(steps)


def purged_group_splits(groups, intervals, n_splits):
    """Ни один исходный отсчёт/padding test не должен присутствовать в train."""
    groups = np.asarray(groups)
    intervals = np.asarray(intervals, float)
    if (intervals.shape != (len(groups), 2) or not np.all(np.isfinite(intervals))
            or np.any(intervals[:, 1] < intervals[:, 0])):
        raise ValueError("Неверные интервалы эпох")
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"Для калибровки нужны >= {n_splits} независимых block_id")
    for train, test in GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups):
        overlap = ((intervals[train, None, 0] <= intervals[None, test, 1] + 1e-9)
                   & (intervals[train, None, 1] >= intervals[None, test, 0] - 1e-9))
        yield train[~np.any(overlap, axis=1)], test


@dataclass
class CalibratedModel:
    """Только числовые параметры, загрузка NPZ без pickle/исполняемого кода."""

    config: P300Config
    algorithm: str
    spatial: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    epoch_shape: tuple[int, int]
    metadata: dict

    @classmethod
    def from_pipeline(cls, pipeline, config, algorithm, shape, metadata):
        channels = shape[0]
        if "xdawn" in pipeline.named_steps:
            basis = np.eye(channels)[:, :, None]
            spatial = pipeline.named_steps["xdawn"].transform(basis)[:, :, 0].T
        else:
            spatial = np.eye(channels)
        scaler, lda = pipeline.named_steps["scale"], pipeline.named_steps["lda"]
        return cls(config, algorithm, spatial, scaler.mean_.copy(), scaler.scale_.copy(),
                   lda.coef_[0].copy(), float(lda.intercept_[0]), tuple(shape), metadata)

    def decision_function(self, data):
        data = np.asarray(data, float)
        if data.ndim == 2:
            data = data[None]
        if data.ndim != 3 or data.shape[1:] != self.epoch_shape or not np.all(np.isfinite(data)):
            raise ValueError("Данные не соответствуют откалиброванной модели")
        features = np.einsum("kc,nct->nkt", self.spatial, data).reshape(len(data), -1)
        return ((features - self.mean) / self.scale) @ self.coef + self.intercept

    def save(self, path):
        metadata = {"schema": 1, "config": self.config.to_dict(), "algorithm": self.algorithm,
                    "epoch_shape": self.epoch_shape, "report": self.metadata}
        with Path(path).open("wb") as file:
            np.savez_compressed(file, metadata=json.dumps(metadata, ensure_ascii=False),
                                spatial=self.spatial, mean=self.mean, scale=self.scale,
                                coef=self.coef, intercept=np.array(self.intercept))

    @classmethod
    def load(cls, path, config=None):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta["schema"] != 1 or meta["algorithm"] not in ALGORITHMS:
                raise ValueError("Неподдерживаемый формат модели")
            saved_config = P300Config.from_dict(meta["config"])
            if config is not None and saved_config != config:
                raise ValueError("Конфигурация/каналы модели отличаются от текущего потока")
            model = cls(saved_config, meta["algorithm"], data["spatial"].copy(), data["mean"].copy(),
                        data["scale"].copy(), data["coef"].copy(), float(data["intercept"]),
                        tuple(meta["epoch_shape"]), meta["report"])
        features = model.spatial.shape[0] * model.epoch_shape[1]
        if (model.spatial.ndim != 2 or model.spatial.shape[1] != len(saved_config.channel_names)
                or model.epoch_shape[0] != len(saved_config.channel_names)
                or any(x.shape != (features,) for x in (model.mean, model.scale, model.coef))
                or np.any(model.scale <= 0)
                or not all(np.all(np.isfinite(x)) for x in
                           (model.spatial, model.mean, model.scale, model.coef, model.intercept))):
            raise ValueError("Повреждённые параметры модели")
        return model


def train_models(epochs, config):
    """Не подбирает победителя по test. Выбор алгоритма задаётся пользователем заранее."""
    if len(epochs) > config.max_calibration_epochs:
        raise ValueError("Превышен предел калибровки")
    if not epochs or any(e.event.target is None for e in epochs):
        raise ValueError("Для обучения нужны реальные эпохи с явными target-метками")
    ids = [e.event.event_id for e in epochs]
    if len(set(ids)) != len(ids):
        raise ValueError("Повторяющиеся event_id в калибровке")
    trials = {}
    for epoch in epochs:
        event = epoch.event
        key = (event.block_id, event.trial_id)
        objects, labels = trials.setdefault(key, (event.objects, {}))
        if event.objects != objects:
            raise ValueError("Изменился набор объектов внутри trial")
        previous = labels.setdefault(event.object_id, event.target)
        if previous != event.target or sum(labels.values()) > 1:
            raise ValueError("Противоречивые target-метки внутри trial")
    X = np.stack([e.data for e in epochs])
    y = np.array([int(e.event.target) for e in epochs])
    if len(np.unique(y)) != 2 or min(np.bincount(y)) < config.min_class_epochs:
        raise ValueError(f"Нужно >= {config.min_class_epochs} принятых эпох каждого класса")
    if X.shape[1] != len(config.channel_names) or not np.all(np.isfinite(X)):
        raise ValueError("Неверная форма/значения калибровки")
    times = np.asarray(epochs[0].times)
    if (times.shape != (X.shape[2],) or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)
            or not np.allclose(np.diff(times), 1 / config.feature_sfreq, rtol=0, atol=1e-9)
            or any(not np.array_equal(e.times, times) for e in epochs)):
        raise ValueError("Несогласованные временные сетки калибровки")
    groups = np.array([e.event.block_id for e in epochs])
    intervals = np.array([e.source_interval for e in epochs])
    splits = list(purged_group_splits(groups, intervals, config.cv_folds))
    folds = []
    for train, test in splits:
        if (len(np.unique(y[train])) != 2 or min(np.bincount(y[train])) < 2
                or len(np.unique(y[test])) != 2):
            raise ValueError("После удаления пересечений каждый fold должен содержать оба класса; "
                             "соберите больше независимых блоков с паузами")
        folds.append({"train_indices": train.tolist(), "test_indices": test.tolist(),
                      "train_event_ids": [ids[i] for i in train],
                      "test_event_ids": [ids[i] for i in test],
                      "train_blocks": np.unique(groups[train]).tolist(),
                      "test_blocks": np.unique(groups[test]).tolist(),
                      "purged_count": len(epochs) - len(train) - len(test)})
    digest = hashlib.sha256(np.ascontiguousarray(X).tobytes())
    digest.update(json.dumps([e.event.to_dict() for e in epochs], sort_keys=True).encode())
    digest.update(intervals.tobytes())
    digest.update(json.dumps(config.to_dict(), sort_keys=True).encode())
    digest.update(np.ascontiguousarray(epochs[0].times).tobytes())
    report = {"schema": 1, "validation": "internal_group_cv_only_not_independent_validation",
              "dataset_sha256": digest.hexdigest(), "versions": software_versions(),
              "config": config.to_dict(), "folds": folds, "algorithms": {},
              "class_counts": np.bincount(y).tolist(), "event_ids": ids,
              "source_intervals": intervals.tolist(), "random_state": config.random_state}
    models = {}
    for algorithm in ALGORITHMS:
        results = []
        for train, test in splits:
            pipeline = make_pipeline(algorithm, config)
            pipeline.fit(X[train], y[train])
            scores = pipeline.decision_function(X[test])
            predicted = (scores >= 0).astype(int)
            results.append({"balanced_accuracy": float(balanced_accuracy_score(y[test], predicted)),
                            "roc_auc": float(roc_auc_score(y[test], scores)),
                            "confusion_matrix": confusion_matrix(y[test], predicted, labels=[0, 1]).tolist(),
                            "test_scores": scores.tolist(), "test_labels": y[test].tolist()})
        report["algorithms"][algorithm] = results
        pipeline = make_pipeline(algorithm, config)
        pipeline.fit(X, y)
        models[algorithm] = CalibratedModel.from_pipeline(pipeline, config, algorithm, X.shape[1:], report)
        # Numerical export must preserve the actual fitted sklearn/MNE predictor.
        if not np.allclose(models[algorithm].decision_function(X),
                           pipeline.decision_function(X), rtol=1e-8, atol=1e-8):
            raise RuntimeError("Числовой экспорт модели не совпадает с обученным Pipeline")
    return models, report
