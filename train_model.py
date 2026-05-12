from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.model import DEFAULT_MODEL_PATH, FEATURE_NAMES, GenderMLP

SEED = 42
DATASET_PATH = Path("data/datasets/voice.csv")
METRICS_PATH = Path("artifacts/voice_gender_metrics.json")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: GenderMLP, features: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(features, dtype=torch.float32)).squeeze(1)
        probabilities = torch.sigmoid(logits).cpu().numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)
    accuracy = accuracy_score(labels, predictions)
    return accuracy, probabilities


def main() -> None:
    set_seed(SEED)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    artifacts_dir = DEFAULT_MODEL_PATH.parent
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(DATASET_PATH)
    features = dataset[FEATURE_NAMES].to_numpy(dtype=np.float32)
    labels = (dataset["label"] == "female").astype(np.int64).to_numpy()

    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        labels,
        test_size=0.30,
        random_state=SEED,
        stratify=labels,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = GenderMLP(input_dim=len(FEATURE_NAMES))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_val_accuracy = -1.0
    patience = 20
    stale_epochs = 0

    for epoch in range(1, 201):
        model.train()
        epoch_loss = 0.0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_features)

        val_accuracy, _ = evaluate(model, x_val, y_val)
        train_accuracy, _ = evaluate(model, x_train, y_train)
        mean_loss = epoch_loss / len(train_dataset)
        print(
            f"epoch={epoch:03d} loss={mean_loss:.4f} "
            f"train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            stale_epochs = 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a model state.")

    model.load_state_dict(best_state)
    train_accuracy, train_probabilities = evaluate(model, x_train, y_train)
    val_accuracy, val_probabilities = evaluate(model, x_val, y_val)
    test_accuracy, test_probabilities = evaluate(model, x_test, y_test)
    test_predictions = (test_probabilities >= 0.5).astype(np.int64)

    metrics = {
        "dataset_name": "Gender Recognition by Voice and Speech Analysis (voice.csv)",
        "dataset_path": str(DATASET_PATH),
        "train_size": int(len(x_train)),
        "validation_size": int(len(x_val)),
        "test_size": int(len(x_test)),
        "train_accuracy": float(train_accuracy),
        "validation_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "confusion_matrix": confusion_matrix(y_test, test_predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            test_predictions,
            target_names=["male", "female"],
            output_dict=True,
            zero_division=0,
        ),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    torch.save(
        {
            "input_dim": len(FEATURE_NAMES),
            "hidden_dims": [64, 32],
            "feature_names": FEATURE_NAMES,
            "label_mapping": {"0": "male", "1": "female"},
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "dataset_name": metrics["dataset_name"],
            "trained_at": metrics["trained_at"],
        },
        DEFAULT_MODEL_PATH,
    )

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nSaved model to {DEFAULT_MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(
        "Final metrics: "
        f"train={metrics['train_accuracy']:.4f} "
        f"val={metrics['validation_accuracy']:.4f} "
        f"test={metrics['test_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
