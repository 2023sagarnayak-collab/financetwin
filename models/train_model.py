import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from models.features import build_features, FEATURES, CLASSES

DATA_PATH = os.path.join(ROOT, "data", "processed", "india_households.csv")
ART_DIR = os.path.join(ROOT, "models", "artifacts")

INCOME_BINS = [10000, 20000, 50000, 100000, 200000, 400001]
INCOME_LABELS = ["10k-20k", "20k-50k", "50k-1L", "1L-2L", "2L-4L"]


def candidate_models():
    """The two models compared. The baseline is reported separately as a reference."""
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=42, n_jobs=-1),
    }


def group_report(model, name, X_test, y_test, raw):
    """Accuracy and macro-F1 for each employment type, income band, loan status and city tier."""
    g = raw.loc[X_test.index]
    pred = pd.Series(model.predict(X_test), index=X_test.index)
    band = pd.cut(g["monthly_salary"], INCOME_BINS, right=False, labels=INCOME_LABELS).astype(str)
    slices = {
        "employment": g["employment_type"].values,
        "income band": band.values,
        "loan status": np.where(g["loan_count"] > 0, "has loan", "no loan"),
        "city tier": g["city_tier"].map({1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}).values,
    }
    rows = []
    for dim, values in slices.items():
        values = pd.Series(values, index=X_test.index)
        for v in sorted(values.unique()):
            idx = values[values == v].index
            rows.append({
                "model": name, "dimension": dim, "group": v, "n": len(idx),
                "accuracy": round(accuracy_score(y_test[idx], pred[idx]), 3),
                "macro_f1": round(f1_score(y_test[idx], pred[idx], average="macro"), 3),
            })
    return pd.DataFrame(rows)


def select_model(results):
    """Pick the model with the highest cross-validated macro-F1 on the training data.
    If the top two are within one standard deviation of each other, pick logistic
    regression, the simpler and more interpretable model."""
    ranked = sorted(results, key=lambda k: results[k]["cv_macro_f1_mean"], reverse=True)
    top, runner = ranked[0], ranked[1]
    gap = results[top]["cv_macro_f1_mean"] - results[runner]["cv_macro_f1_mean"]
    noise = max(results[top]["cv_macro_f1_std"], results[runner]["cv_macro_f1_std"])
    if gap < noise and "Logistic regression" in (top, runner):
        reason = (f"Difference in CV macro-F1 ({gap:.3f}) is within one standard deviation "
                  f"({noise:.3f}), so the simpler, more interpretable logistic regression is selected.")
        return "Logistic regression", reason
    reason = f"Highest cross-validated macro-F1 on the training data (gap {gap:.3f} over {runner})."
    return top, reason


def main():
    raw = pd.read_csv(DATA_PATH)
    X = build_features(raw)[FEATURES]
    y = raw["health_class"]

    # Test set is held out. It is used only for the final reported scores, never for selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Reference: a model that ignores the inputs and always predicts the most common class
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    reference = {
        "Majority-class baseline": {
            "test_accuracy": round(float(accuracy_score(y_test, base_pred)), 3),
            "test_macro_f1": round(float(f1_score(y_test, base_pred, average="macro")), 3),
        }
    }

    # Step 1: evaluate both candidates with the same cross-validation on the training split
    results, fitted, group_tables = {}, {}, []
    for name, model in candidate_models().items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "cv_macro_f1_mean": round(float(cv_scores.mean()), 3),
            "cv_macro_f1_std": round(float(cv_scores.std()), 3),
            "test_accuracy": round(float(accuracy_score(y_test, pred)), 3),
            "test_macro_f1": round(float(f1_score(y_test, pred, average="macro")), 3),
        }
        fitted[name] = model
        group_tables.append(group_report(model, name, X_test, y_test, raw))
        print(f"{name:22s} CV macro-F1 {results[name]['cv_macro_f1_mean']:.3f} "
              f"(+/- {results[name]['cv_macro_f1_std']:.3f})   "
              f"test accuracy {results[name]['test_accuracy']:.3f}")

    print(f"{'Majority-class baseline':22s} test accuracy "
          f"{reference['Majority-class baseline']['test_accuracy']:.3f} (reference only)")

    # Step 2: select on the training data only
    best_name, reason = select_model(results)
    best = fitted[best_name]
    print(f"\nSelected model: {best_name}\n{reason}")

    pred = best.predict(X_test)
    print(classification_report(y_test, pred, labels=CLASSES, target_names=CLASSES))
    cm = confusion_matrix(y_test, pred, labels=CLASSES)

    imp = permutation_importance(best, X_test, y_test, n_repeats=10,
                                 random_state=42, scoring="f1_macro")
    importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importance (drop in macro-F1 when shuffled):")
    print(importance.round(4))

    groups = pd.concat(group_tables, ignore_index=True)
    print("\nPerformance by group (test set, both models):")
    print(groups.to_string(index=False))

    os.makedirs(ART_DIR, exist_ok=True)
    joblib.dump(best, os.path.join(ART_DIR, "health_model.joblib"))
    groups.to_csv(os.path.join(ART_DIR, "group_report.csv"), index=False)
    with open(os.path.join(ART_DIR, "metrics.json"), "w") as f:
        json.dump({
            "models": results,
            "reference": reference,
            "selected": best_name,
            "selection_reason": reason,
            "confusion_matrix": cm.tolist(),
            "class_order": CLASSES,
            "importance": importance.round(4).to_dict(),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        }, f, indent=2)
    print(f"\nSaved selected model, metrics and group report to {ART_DIR}")


if __name__ == "__main__":
    main()