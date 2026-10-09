# A7: kNN-регрессия, k=3, евклидово расстояние, прогноз = среднее y трёх соседей.
import numpy as np

X = np.array([[1, 1], [2, 1], [4, 3], [5, 4], [1, 3], [3, 5]], dtype=float)
y = np.array([2.0, 3.0, 6.0, 7.0, 4.0, 8.0])
query = np.array([2.0, 2.0])
dist = np.linalg.norm(X - query, axis=1)
order = np.argsort(dist)
print("расстояния по возрастанию:", np.round(dist[order], 3))   # проверка, что нет ничьей на 3-м месте
print(f"REFERENCE: {y[order[:3]].mean():.4f}")
