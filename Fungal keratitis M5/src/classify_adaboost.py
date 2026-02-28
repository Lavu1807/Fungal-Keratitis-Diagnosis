"""AdaBoost classifier for Model M5."""
import os, json, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "features", "features_m5.csv")
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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()

    print("M5 — AdaBoost")
    print("=" * 50)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        clf = AdaBoostClassifier(
            n_estimators=50,
            learning_rate=1.0,
            algorithm="SAMME",
            random_state=42
        )
        clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    results = evaluate("AdaBoost", y_test, y_pred, y_prob)
    with open(os.path.join(RESULTS_DIR, "adaboost_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/adaboost_results.json")


if __name__ == "__main__":
    main()
