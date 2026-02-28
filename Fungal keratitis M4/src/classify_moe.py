"""Mixture of Experts (MoE) ensemble for Model M4.
Gating network: MLP with hidden_layers=(10,), max_iter=500, activation=relu.
Experts: LR, Linear SVC, GNB, RF, AdaBoost, RUSBoost, KNN.
"""
import os, json, warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "data", "features", "features_m4.csv")
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


def create_experts():
    """Create all 7 base classifiers (experts) with Table 2 parameters."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        experts = [
            ("LR", LogisticRegression(max_iter=1000, solver="lbfgs", penalty="l2", random_state=42)),
            ("SVC", SVC(kernel="linear", probability=True, C=1.0, random_state=42)),
            ("GNB", GaussianNB(var_smoothing=1e-9)),
            ("RF", RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, criterion="gini", random_state=42)),
            ("AdaBoost", AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm="SAMME", random_state=42)),
            ("RUSBoost", RUSBoostClassifier(n_estimators=50, learning_rate=1.0, sampling_strategy="auto", random_state=42)),
            ("KNN", KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="minkowski")),
        ]
    return experts


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data()

    print("M4 — Mixture of Experts (MoE)")
    print("=" * 50)

    # Step 1: Train all experts
    experts = create_experts()
    print("  Training 7 experts...")
    for name, clf in experts:
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"    {name}: {acc:.2%}")

    # Step 2: Get expert predictions (probabilities) for training gating network
    expert_train_probs = np.column_stack([
        clf.predict_proba(X_train)[:, 1] for _, clf in experts
    ])
    expert_test_probs = np.column_stack([
        clf.predict_proba(X_test)[:, 1] for _, clf in experts
    ])

    # Step 3: Train gating network (MLP)
    print("  Training gating network (MLP)...")
    gating = MLPClassifier(
        hidden_layer_sizes=(10,),
        max_iter=500,
        activation="relu",
        random_state=42
    )
    gating.fit(expert_train_probs, y_train)

    # Step 4: Gating network assigns weights to experts
    gate_probs = gating.predict_proba(expert_test_probs)  # (n_samples, 2)

    # Use gating probabilities as confidence-weighted combination
    # The gating network's output indicates which "class" each expert combination maps to
    # For MoE: weighted combination of expert predictions using gating weights
    # Alternative approach: train gating on expert outputs, use its prediction directly
    y_pred = gating.predict(expert_test_probs)
    y_prob = gating.predict_proba(expert_test_probs)[:, 1]

    results = evaluate("Mixture of Experts", y_test, y_pred, y_prob)
    with open(os.path.join(RESULTS_DIR, "moe_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/moe_results.json")


if __name__ == "__main__":
    main()
