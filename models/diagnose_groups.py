import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from models.features import build_features, FEATURES, CLASSES

DATA_PATH = os.path.join(ROOT, "data", "processed", "india_households.csv")
ART_DIR = os.path.join(ROOT, "models", "artifacts")
INCOME_BINS = [10000, 20000, 50000, 100000, 200000, 400001]
INCOME_LABELS = ["10k-20k", "20k-50k", "50k-1L", "1L-2L", "2L-4L"]


def main():
    raw = pd.read_csv(DATA_PATH)
    X = build_features(raw)[FEATURES]
    y = raw["health_class"]

    # Training split only. The test set is not used here.
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    raw_train = raw.loc[X_train.index]

    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = pd.Series(cross_val_predict(model, X_train, y_train, cv=cv), index=X_train.index)

    # Distance between classes: 0 = same, 1 = neighbours, 2 = Stable vs Distressed
    rank = {c: i for i, c in enumerate(CLASSES)}
    distance = (y_train.map(rank) - pred.map(rank)).abs()

    band = pd.cut(raw_train["monthly_salary"], INCOME_BINS, right=False,
                  labels=INCOME_LABELS).astype(str)
    groups = {
        "employment": pd.Series(raw_train["employment_type"].values, index=X_train.index),
        "income band": pd.Series(band.values, index=X_train.index),
        "loan status": pd.Series(np.where(raw_train["loan_count"] > 0, "has loan", "no loan"),
                                 index=X_train.index),
    }

    rows = []
    for dim, values in groups.items():
        for g in sorted(values.unique()):
            idx = values[values == g].index
            wrong = y_train[idx] != pred[idx]
            n_wrong = int(wrong.sum())
            neighbour = int((distance[idx][wrong] == 1).sum())
            rows.append({
                "dimension": dim,
                "group": g,
                "n": len(idx),
                "accuracy": round(accuracy_score(y_train[idx], pred[idx]), 3),
                "errors": n_wrong,
                "neighbour_error_share": round(neighbour / n_wrong, 3) if n_wrong else None,
                "stable_vs_distressed_errors": n_wrong - neighbour,
            })
    table = pd.DataFrame(rows)

    print("Training-split cross-validated results by group (logistic regression)\n")
    print(table.to_string(index=False))

    for g in ["informal", "salaried"]:
        idx = groups["employment"][groups["employment"] == g].index
        cm = pd.DataFrame(confusion_matrix(y_train[idx], pred[idx], labels=CLASSES,
                                           normalize="true"),
                          index=[f"actual {c}" for c in CLASSES],
                          columns=[f"pred {c}" for c in CLASSES]).round(3)
        print(f"\nRow-normalised confusion matrix, {g}:")
        print(cm.to_string())

    os.makedirs(ART_DIR, exist_ok=True)
    table.to_csv(os.path.join(ART_DIR, "group_error_diagnosis.csv"), index=False)
    print(f"\nSaved to {os.path.join(ART_DIR, 'group_error_diagnosis.csv')}")


if __name__ == "__main__":
    main()