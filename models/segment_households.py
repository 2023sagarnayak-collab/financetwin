import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from models.features import build_features, FEATURES

DATA_PATH = os.path.join(ROOT, "data", "processed", "india_households.csv")
ART_DIR = os.path.join(ROOT, "models", "artifacts")
K_RANGE = range(3, 7)
SAMPLE_SIZE = 10000


def segment_profiles(raw_train, y_train, labels):
    """Describe each segment in plain terms, including the health-class mix as a check."""
    df = pd.DataFrame({
        "segment": labels,
        "salary": raw_train["monthly_salary"].values,
        "emi_ratio": (raw_train["monthly_emi"] / raw_train["monthly_salary"]).values,
        "surplus": (raw_train["monthly_salary"] - raw_train["monthly_expenses"]
                    - raw_train["monthly_emi"]).values,
        "has_loan": (raw_train["loan_count"] > 0).values,
        "informal": (raw_train["employment_type"] == "informal").values,
        "stable": (y_train.values == "Stable"),
        "stretched": (y_train.values == "Stretched"),
        "distressed": (y_train.values == "Distressed"),
    })
    prof = df.groupby("segment").agg(
        households=("salary", "size"),
        median_salary=("salary", "median"),
        median_emi_ratio=("emi_ratio", "median"),
        median_surplus=("surplus", "median"),
        share_with_loan=("has_loan", "mean"),
        share_informal=("informal", "mean"),
        share_stable=("stable", "mean"),
        share_stretched=("stretched", "mean"),
        share_distressed=("distressed", "mean"),
    ).round(2)
    prof["share_of_households"] = (prof["households"] / prof["households"].sum()).round(3)
    return prof


def main():
    raw = pd.read_csv(DATA_PATH)
    X = build_features(raw)[FEATURES]
    y = raw["health_class"]

    # Same training split as the health model. Segments are formed from training households only.
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    raw_train = raw.loc[X_train.index]

    Z = StandardScaler().fit_transform(X_train)
    rng = np.random.default_rng(42)
    sample = rng.choice(len(Z), size=SAMPLE_SIZE, replace=False)

    k_scores = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Z)
        k_scores[k] = round(float(silhouette_score(Z[sample], km.labels_[sample])), 3)
        print(f"k={k}: silhouette = {k_scores[k]:.3f}")

    best_k = max(k_scores, key=k_scores.get)
    print(f"\nSelected k = {best_k} (highest silhouette score)\n")

    model = make_pipeline(StandardScaler(), KMeans(n_clusters=best_k, random_state=42, n_init=10))
    model.fit(X_train)
    labels = model.predict(X_train)

    profiles = segment_profiles(raw_train, y_train, labels)
    print(profiles.to_string())

    os.makedirs(ART_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(ART_DIR, "segment_model.joblib"))
    profiles.to_csv(os.path.join(ART_DIR, "segment_profiles.csv"))
    with open(os.path.join(ART_DIR, "segment_metrics.json"), "w") as f:
        json.dump({"k_scores": k_scores, "selected_k": best_k}, f, indent=2)
    print(f"\nSaved segment model, profiles and metrics to {ART_DIR}")


if __name__ == "__main__":
    main()