import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Build Recency, Frequency, Monetary features per customer."""
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Snapshot date = one day after the last transaction in the dataset
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    return rfm


def build_feature_matrix(rfm: pd.DataFrame):
    features = pd.DataFrame({
        "Recency": rfm["Recency"],
        "Frequency": np.log1p(rfm["Frequency"]),
        "Monetary": np.log1p(rfm["Monetary"]),
    })
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, features.columns

def find_best_k(X, k_range=range(2, 9)):
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = score
        print(f"k={k}: silhouette score = {score:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"\nBest k by silhouette score: {best_k}")
    return best_k


def profile_clusters(rfm: pd.DataFrame, labels: np.ndarray):
    rfm = rfm.copy()
    rfm["cluster"] = labels

    print("\nCluster sizes:")
    print(rfm["cluster"].value_counts().sort_index())

    print("\nMean Recency / Frequency / Monetary by cluster:")
    print(rfm.groupby("cluster")[["Recency", "Frequency", "Monetary"]].mean())

    return rfm


if __name__ == "__main__":
    retail = pd.read_csv(os.path.join(PROCESSED_DIR, "retail_clean.csv"))
    rfm = build_rfm(retail)
    print(f"Unique customers: {len(rfm)}")
    print(rfm.describe())

    X, feature_names = build_feature_matrix(rfm)
    best_k = find_best_k(X)

    final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = final_model.fit_predict(X)

    profiled = profile_clusters(rfm, labels)
    profiled.to_csv(os.path.join(PROCESSED_DIR, "customer_clustered.csv"), index=False)
    print(f"\nSaved clustered RFM data to customer_clustered.csv")