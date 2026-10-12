"""CLI для обработки архивов, калибровки и оценки фиксированной модели."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from .model import CalibratedModel, train_models
from .offline import preprocess_session
from .storage import load_epochs, save_epochs, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preprocess = commands.add_parser("preprocess", help="Обработать реальный архив raw EEG + маркеры")
    preprocess.add_argument("session", type=Path)
    preprocess.add_argument("output", type=Path)
    train = commands.add_parser("train", help="Обучить две модели на размеченных эпохах NPZ")
    train.add_argument("epochs", type=Path)
    train.add_argument("output_directory", type=Path)
    evaluate = commands.add_parser("evaluate", help="Оценить замороженную модель на отдельном датасете")
    evaluate.add_argument("model", type=Path)
    evaluate.add_argument("epochs", type=Path)
    evaluate.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "preprocess":
        epochs, config, report = preprocess_session(args.session)
        save_epochs(args.output, epochs, config)
        print(f'Принято: {report["accepted"]}, отклонено: {report["rejected"]}')
        return
    epochs, config = load_epochs(args.epochs)
    if args.command == "train":
        models, report = train_models(epochs, config)
        args.output_directory.mkdir(parents=True, exist_ok=False)
        for name, model in models.items():
            model.save(args.output_directory / f"{name}.npz")
        write_json(args.output_directory / "calibration_report.json", report)
        print("Обучены обе модели; отчёт содержит только внутреннюю CV по блокам.")
        return
    model = CalibratedModel.load(args.model, config)
    if not epochs or any(e.event.target is None for e in epochs):
        parser.error("Для оценки нужны явно размеченные эпохи обоих классов")
    if set(e.event.event_id for e in epochs) & set(model.metadata.get("event_ids", [])):
        parser.error("В датасете оценки обнаружены event_id из обучения")
    training_blocks = {block for fold in model.metadata.get("folds", [])
                       for block in [*fold["train_blocks"], *fold["test_blocks"]]}
    if training_blocks & {e.event.block_id for e in epochs}:
        parser.error("Для внешней оценки используйте отдельные экспериментальные block_id")
    y = np.array([int(e.event.target) for e in epochs])
    if len(np.unique(y)) != 2:
        parser.error("Для оценки требуются оба класса")
    scores = model.decision_function(np.stack([e.data for e in epochs]))
    result = {"validation": "external_dataset_evaluation_requires_real_data_provenance",
              "model_dataset_sha256": model.metadata.get("dataset_sha256"),
              "event_ids": [e.event.event_id for e in epochs],
              "balanced_accuracy": float(balanced_accuracy_score(y, scores >= 0)),
              "roc_auc": float(roc_auc_score(y, scores)),
              "confusion_matrix": confusion_matrix(y, scores >= 0, labels=[0, 1]).tolist(),
              "scores": scores.tolist(), "labels": y.tolist()}
    write_json(args.output, result)
    print("Оценка сохранена. Научная валидность зависит от независимости и происхождения данных.")


if __name__ == "__main__":
    main()
