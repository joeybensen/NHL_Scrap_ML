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
    return indices[0][1:], distances[0][1:]

def run_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return model, labels


def elbow_method(X, max_k=4):
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


def assign_labels(df, labels, label_map, col_name="cluster"):
    df[col_name] = labels
    df["cluster_label"] = df[col_name].map(label_map)
    return df

CLUSTER_LABELS_3 = [
    "Top-line scorer", 
    "Secondary / Two-way", 
    "Depth / Grinder"]

def main():

    forwards_df, current_df, personal_df = load_data(DATA_PATHS)

    # --- features ---
    X_raw, scaler = build_feature_matrix(forwards_df, NUMERIC_FEATURES)
    X_weighted = apply_weights(X_raw, WEIGHTS)

    # --- KNN ---
    knn = build_knn(X_weighted)

    target_idx = 326

    print(f'target player: {forwards_df.iloc[target_idx,1]} {forwards_df.iloc[target_idx,2]}')

    neighbors, distances = get_similar_players(knn, X_weighted, target_idx)

    pids = forwards_df.iloc[neighbors]["playerId"].values


    print("Similar player - Distance:")
    for i, neighbor_idx in enumerate(neighbors):
        print(f'{forwards_df.iloc[neighbor_idx,1]} {forwards_df.iloc[neighbor_idx,2]} - {round(distances[i],3)}')

    print(personal_df[personal_df["playerId"].isin(pids)])
    print(current_df[current_df["playerId"].isin(pids)])

    # --- CLUSTERING (choose ONE system) ---
    k = 3
    model, labels = run_kmeans(X_weighted, k)

    centroid_df = pd.DataFrame(model.cluster_centers_, columns=NUMERIC_FEATURES)
    sorted_clusters = centroid_df["points_per_60"].sort_values(ascending=False).index
    label_map = dict(zip(sorted_clusters, CLUSTER_LABELS_3))

    forwards_df = assign_labels(forwards_df, labels, label_map)

    # --- SAVE ---
    forwards_df.to_csv("files/forwards_data.csv", index=False)

    # --- OPTIONAL EVALUATION ---
    elbow_method(X_weighted, max_k=6)

    centroid_df = pd.DataFrame(model.cluster_centers_, columns=NUMERIC_FEATURES)
    sorted_clusters = centroid_df["points_per_60"].sort_values(ascending=False).index

    print(centroid_df)


if __name__ == "__main__":
    main()