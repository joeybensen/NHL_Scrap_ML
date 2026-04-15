import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

DATA_PATHS = {
    "forwards": "files/forwards_data.csv",
    "current": "files/skater_current_stats.csv",
    "personal": "files/personal_data.csv"
}

NUMERIC_FEATURES = [
    "goals_per_60",
    "assists_per_60",
    "points_per_60",
    "shots_per_60",
    "avgToi",
    "powerPlayPoints"
]

WEIGHTS = np.array([2, 2, 3, 1.5, 1, 1.5])

def load_data(paths):
    forwards = pd.read_csv(paths["forwards"])
    current = pd.read_csv(paths["current"])
    personal = pd.read_csv(paths["personal"])
    return forwards, current, personal

def build_feature_matrix(df, features, scaler=None):
    X = df[features].fillna(0).values

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


def apply_weights(X, weights):
    return X * weights


def build_knn(X, n_neighbors=6):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    knn.fit(X)
    return knn


def get_similar_players(knn, X, idx):
    distances, indices = knn.kneighbors([X[idx]])
    return indices[0], distances[0]

def run_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return model, labels


def elbow_method(X, max_k=6):
    inertias = []

    for k in range(1, max_k):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        inertias.append(model.inertia_)

    plt.plot(range(1, max_k), inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show()

    return inertias


CLUSTER_LABELS_3 = {
    0: "Top-line scorer",
    1: "Secondary / Two-way",
    2: "Depth / Grinder"
}

CLUSTER_LABELS_5 = {
    0: "Top-line scorer",
    1: "Depth / Middle-six",
    2: "Second-line scorer",
    3: "Bottom-six / depth",
    4: "Third-line / secondary scoring"
}


def assign_labels(df, labels, label_map, col_name="cluster"):
    df[col_name] = labels
    df["cluster_label"] = df[col_name].map(label_map)
    return df

def main():

    forwards_df, current_df, personal_df = load_data(DATA_PATHS)

    # --- features ---
    X_raw, scaler = build_feature_matrix(forwards_df, NUMERIC_FEATURES)
    X_weighted = apply_weights(X_raw, WEIGHTS)

    # --- KNN ---
    knn = build_knn(X_weighted)

    target_idx = 326 
    neighbors, distances = get_similar_players(knn, X_weighted, target_idx)

    pids = forwards_df.iloc[neighbors]["playerId"].values

    print("Similar players:", neighbors)
    print("Distances:", distances)

    print(personal_df[personal_df["playerId"].isin(pids)])
    print(current_df[current_df["playerId"].isin(pids)])

    # --- CLUSTERING (choose ONE system) ---
    k = 3
    model, labels = run_kmeans(X_weighted, k)

    forwards_df = assign_labels(
        forwards_df,
        labels,
        CLUSTER_LABELS_3
    )

    forwards_df["cluster"] = labels

    # --- SAVE ---
    forwards_df.to_csv("files/forwards_data.csv", index=False)

    # --- OPTIONAL EVALUATION ---
    elbow_method(X_weighted, max_k=6)


if __name__ == "__main__":
    main()