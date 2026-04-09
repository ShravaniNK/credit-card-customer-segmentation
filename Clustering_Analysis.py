# src/clustering_analysis.py

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = Path("data/Credit Card Dataset for Clustering.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_LABELED = OUTPUT_DIR / "credit_card_customers_labeled.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "credit_card_cluster_summary.json"
OUTPUT_MODEL_COMPARISON = OUTPUT_DIR / "clustering_model_comparison.csv"
OUTPUT_CLUSTER_PROFILES = OUTPUT_DIR / "winning_cluster_profiles.csv"

RANDOM_STATE = 42
N_CLUSTERS = 4

FEATURES_ALL = [
    "BALANCE",
    "BALANCE_FREQUENCY",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "CASH_ADVANCE_TRX",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT",
    "TENURE",
    "RECENCY_PROXY",
]

SEGMENT_FEATURES = ["PURCHASES", "PURCHASES_FREQUENCY", "RECENCY_PROXY"]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].median())
    df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].median())
    df["RECENCY_PROXY"] = 1 - df["BALANCE_FREQUENCY"].fillna(df["BALANCE_FREQUENCY"].median())
    return df


def build_preprocessor(feature_names):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, feature_names),
        ]
    )


def fit_models(X_scaled):
    models = {
        "KMeans": KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=RANDOM_STATE),
        "Hierarchical": AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward"),
        "GaussianMixture": GaussianMixture(
            n_components=N_CLUSTERS,
            covariance_type="full",
            random_state=RANDOM_STATE,
        ),
        "DBSCAN": DBSCAN(eps=0.9, min_samples=12),
    }

    labels = {}
    labels["KMeans"] = models["KMeans"].fit_predict(X_scaled)
    labels["Hierarchical"] = models["Hierarchical"].fit_predict(X_scaled)
    labels["GaussianMixture"] = models["GaussianMixture"].fit_predict(X_scaled)

    dbscan_labels = models["DBSCAN"].fit_predict(X_scaled)
    labels["DBSCAN"] = dbscan_labels

    return labels


def evaluate_clustering(X_scaled, labels):
    valid_mask = labels != -1
    X_eval = X_scaled[valid_mask]
    y_eval = labels[valid_mask]

    unique_clusters = np.unique(y_eval)

    if len(unique_clusters) < 2:
        return {
            "silhouette_score": np.nan,
            "davies_bouldin_index": np.nan,
            "n_clusters": int(len(unique_clusters)),
            "noise_points": int(np.sum(labels == -1)),
        }

    return {
        "silhouette_score": float(silhouette_score(X_eval, y_eval)),
        "davies_bouldin_index": float(davies_bouldin_score(X_eval, y_eval)),
        "n_clusters": int(len(unique_clusters)),
        "noise_points": int(np.sum(labels == -1)),
    }


def choose_winner(scores_df):
    valid = scores_df.dropna(subset=["silhouette_score", "davies_bouldin_index"]).copy()
    valid["silhouette_rank"] = valid["silhouette_score"].rank(ascending=False, method="min")
    valid["db_rank"] = valid["davies_bouldin_index"].rank(ascending=True, method="min")
    valid["combined_rank"] = valid["silhouette_rank"] + valid["db_rank"]

    winner = (
        valid.sort_values(
            ["combined_rank", "silhouette_score", "davies_bouldin_index"],
            ascending=[True, False, True],
        )
        .iloc[0]["algorithm"]
    )
    return winner


def assign_persona(row):
    if row["PURCHASES"] >= 3000 and row["PURCHASES_FREQUENCY"] >= 0.75:
        return "Premium Power Users"
    if row["PURCHASES_FREQUENCY"] >= 0.70 and row["PURCHASES"] < 700:
        return "Engaged Low Spenders"
    if row["RECENCY_PROXY"] >= 0.12 and row["PURCHASES"] < 600:
        return "At-Risk Dormant Customers"
    return "Everyday Revolving Spenders"


def marketing_strategy(persona):
    strategies = {
        "Premium Power Users": "Offer premium rewards, loyalty perks, exclusive benefits, and higher credit line upsell campaigns.",
        "Engaged Low Spenders": "Use basket-building offers, category bundles, and spend-threshold cashback to convert engagement into higher spend.",
        "At-Risk Dormant Customers": "Launch win-back campaigns with limited-time credits, reminders, and personalized reactivation incentives.",
        "Everyday Revolving Spenders": "Promote installment plans, autopay nudges, and recurring everyday partner offers.",
    }
    return strategies[persona]


def summarize_clusters(df, cluster_col):
    summary = (
        df.groupby(cluster_col)
        .agg(
            customers=("CUST_ID", "count"),
            BALANCE=("BALANCE", "mean"),
            PURCHASES=("PURCHASES", "mean"),
            CREDIT_LIMIT=("CREDIT_LIMIT", "mean"),
            PURCHASES_FREQUENCY=("PURCHASES_FREQUENCY", "mean"),
            RECENCY_PROXY=("RECENCY_PROXY", "mean"),
            TENURE=("TENURE", "mean"),
        )
        .reset_index()
        .sort_values("PURCHASES", ascending=False)
    )

    summary["persona"] = summary.apply(assign_persona, axis=1)
    summary["marketing_strategy"] = summary["persona"].apply(marketing_strategy)

    return summary


def add_spending_labels(df):
    def spending_band(purchases):
        if purchases >= 3000:
            return "High Spending"
        if purchases < 500:
            return "Low Spending"
        return "Mid Spending"

    def hidden_segment(row):
        if row["PURCHASES_FREQUENCY"] >= 0.80 and row["PURCHASES"] < 700:
            return "High Engagement, Low Spending"
        if row["BALANCE_FREQUENCY"] < 0.80 and row["PURCHASES"] < 500:
            return "Dormant Low Spenders"
        if row["PURCHASES"] >= 3000:
            return "Premium Spenders"
        return "Core Transactors"

    df["spend_band"] = df["PURCHASES"].apply(spending_band)
    df["hidden_segment"] = df.apply(hidden_segment, axis=1)
    return df


def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)
    df = add_spending_labels(df)

    print("Preprocessing features...")
    preprocessor = build_preprocessor(FEATURES_ALL)
    X_scaled = preprocessor.fit_transform(df[FEATURES_ALL])

    print("Running clustering models...")
    labels_dict = fit_models(X_scaled)

    score_rows = []
    for algorithm, labels in labels_dict.items():
        metrics = evaluate_clustering(X_scaled, labels)
        score_rows.append({"algorithm": algorithm, **metrics})

    scores_df = pd.DataFrame(score_rows)
    winning_algorithm = choose_winner(scores_df)
    print(f"Winning algorithm: {winning_algorithm}")

    for algorithm, labels in labels_dict.items():
        df[f"{algorithm.lower()}_cluster"] = labels

    df["winning_cluster"] = labels_dict[winning_algorithm]

    print("Running PCA for 2D visualization...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_coords = pca.fit_transform(X_scaled)
    df["pca_x"] = pca_coords[:, 0]
    df["pca_y"] = pca_coords[:, 1]

    cluster_profiles = summarize_clusters(df, "winning_cluster")

    scores_df.to_csv(OUTPUT_MODEL_COMPARISON, index=False)
    cluster_profiles.to_csv(OUTPUT_CLUSTER_PROFILES, index=False)
    df.to_csv(OUTPUT_LABELED, index=False)

    summary_payload = {
        "winning_algorithm": winning_algorithm,
        "scores": scores_df.to_dict(orient="records"),
        "cluster_profiles": cluster_profiles.to_dict(orient="records"),
        "files_generated": {
            "labeled_dataset": str(OUTPUT_LABELED),
            "model_comparison": str(OUTPUT_MODEL_COMPARISON),
            "cluster_profiles": str(OUTPUT_CLUSTER_PROFILES),
        },
        "notes": [
            "RECENCY_PROXY was derived as 1 - BALANCE_FREQUENCY because no direct recency field existed in the dataset.",
            "DBSCAN may produce fewer than 4 effective clusters and may mark some records as noise (-1).",
            "Final cluster labeling uses the winning algorithm selected from Silhouette Score and Davies-Bouldin Index.",
        ],
    }

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\nModel Comparison")
    print(scores_df.sort_values(["silhouette_score", "davies_bouldin_index"], ascending=[False, True]))

    print("\nCluster Profiles")
    print(cluster_profiles[["winning_cluster", "customers", "persona", "marketing_strategy"]])

    print("\nFiles saved:")
    print(f"- {OUTPUT_MODEL_COMPARISON}")
    print(f"- {OUTPUT_CLUSTER_PROFILES}")
    print(f"- {OUTPUT_LABELED}")
    print(f"- {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()