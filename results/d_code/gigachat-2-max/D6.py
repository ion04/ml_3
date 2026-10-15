from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Загрузка данных
data = load_iris()
X = data.data

# Стандартизация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Определение возможных значений k
k_values = [2, 3, 4, 5, 6]
best_k = None
max_silhouette = -1

# Поиск оптимального числа кластеров
for k in k_values:
    # Кластеризация
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Вычисление коэффициента силуэта
    silhouette_avg = silhouette_score(X_scaled, labels)

    # Обновление лучшего результата
    if silhouette_avg > max_silhouette:
        max_silhouette = silhouette_avg
        best_k = k

# Финальная кластеризация с оптимальным числом кластеров
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
final_labels = final_kmeans.fit_predict(X_scaled)

# Вычисление финальной метрики
final_silhouette = silhouette_score(X_scaled, final_labels)

# Печать результата
print(f"METRIC: {final_silhouette:.4f}")

# Ответ и уверенность
ОТВЕТ: код выше
УВЕРЕННОСТЬ: 100
