from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Загрузка датасета
iris = load_iris()
X = iris.data

# Стандартизация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Перебор числа кластеров и выбор оптимального по коэффициенту силуэта
best_k = 2
best_silhouette = -1
for k in [2, 3, 4, 5, 6]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)
    if silhouette > best_silhouette:
        best_silhouette = silhouette
        best_k = k

# Кластеризация с оптимальным k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# Вычисление итогового коэффициента силуэта
final_silhouette = silhouette_score(X_scaled, labels)

print(f"METRIC: {final_silhouette}")
