# D6 (эталон): load_iris без меток, StandardScaler + KMeans, выбор k из [2..6] по силуэту, силуэт итоговой кластеризации.
# Разбиения на train/test нет (обучение без учителя), поэтому воспроизводимость обеспечивают random_state=42 и n_init=10
# в самом KMeans. Силуэт считается в том же (стандартизированном) пространстве, где шла кластеризация.
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(load_iris().data)   # метки load_iris().target не используются

scores = {}
for k in [2, 3, 4, 5, 6]:
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    scores[k] = silhouette_score(X, labels)

best_k = max(scores, key=scores.get)
print("силуэт по k:", {k: round(s, 4) for k, s in scores.items()}, "лучшее k:", best_k)
print(f"METRIC: {scores[best_k]:.4f}")
