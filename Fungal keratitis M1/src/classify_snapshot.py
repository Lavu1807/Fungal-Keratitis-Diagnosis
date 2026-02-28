"""Snapshot Ensembling for Model M1.
5 snapshots, 10 epochs per snapshot, batch_size=32,
cyclic LR: base_lr=0.001, max_lr=0.01, step_size=5,
hidden layers: 64, 32.
"""
import os, json, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "features", "features_m1.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")


def load_data():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=["filename", "label"]).values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def evaluate(name, y_true, y_pred, y_prob):
    """Compute all metrics matching Table 5 of the paper."""
    acc = accuracy_score(y_true, y_pred)
    f1_n = f1_score(y_true, y_pred, pos_label=0)
    f1_f = f1_score(y_true, y_pred, pos_label=1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    results = {
        "classifier": name,
        "accuracy": round(acc, 4),
        "f1_normal": round(f1_n, 4),
        "f1_fungal": round(f1_f, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
    }

    print(f"  Accuracy:    {acc:.2%}")
    print(f"  F1 Normal:   {f1_n:.4f}")
    print(f"  F1 Fungal:   {f1_f:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  AUROC:       {auroc:.4f}")
    print(f"  AUPRC:       {auprc:.4f}")
    return results


def cosine_annealing_lr(epoch, max_lr=0.01, base_lr=0.001, T=10):
    """Cosine annealing learning rate schedule."""
    return base_lr + 0.5 * (max_lr - base_lr) * (1 + np.cos(np.pi * epoch / T))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()

    print("M1 — Snapshot Ensembling")
    print("=" * 50)

    n_snapshots = 5
    epochs_per_snapshot = 10
    batch_size = 32
    base_lr = 0.001
    max_lr = 0.01

    snapshots = []

    # Create base MLP model
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=1,
        batch_size=batch_size,
        learning_rate_init=max_lr,
        learning_rate="constant",
        warm_start=True,
        random_state=42
    )

    print(f"  Training {n_snapshots} snapshots ({epochs_per_snapshot} epochs each)...")

    for snap in range(n_snapshots):
        for epoch in range(epochs_per_snapshot):
            # Update learning rate via cosine annealing
            lr = cosine_annealing_lr(epoch, max_lr, base_lr, epochs_per_snapshot)
            model.learning_rate_init = lr

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

        # Save snapshot
        snapshot_model = copy.deepcopy(model)
        snapshots.append(snapshot_model)
        snap_acc = accuracy_score(y_test, snapshot_model.predict(X_test))
        print(f"    Snapshot {snap + 1}: acc={snap_acc:.2%}")

    # Ensemble: average predictions from all snapshots
    print("  Combining snapshots...")
    all_probs = np.zeros((len(y_test), 2))
    for snap_model in snapshots:
        all_probs += snap_model.predict_proba(X_test)
    all_probs /= n_snapshots

    y_prob = all_probs[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    results = evaluate("Snapshot Ensembling", y_test, y_pred, y_prob)
    with open(os.path.join(RESULTS_DIR, "snapshot_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/snapshot_results.json")


if __name__ == "__main__":
    main()
