import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Загрузка данных
iris = load_iris()
X = iris.data

# Стандартизация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Подбор оптимального k по коэффициенту силуэта
k_values = [2, 3, 4, 5, 6]
best_k = None
best_silhouette = -1

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    if sil_score > best_silhouette:
        best_silhouette = sil_score
        best_k = k

# Итоговая кластеризация с оптимальным k
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
final_labels = kmeans_final.fit_predict(X_scaled)
final_silhouette = silhouette_score(X_scaled, final_labels)

print(f"METRIC: {final_silhouette:.6f}")
