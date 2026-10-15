from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Загрузка датасета
wine = load_wine()
X, y = wine.data, wine.target

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Подбор оптимального k с помощью 5-блочной кросс-валидации
best_k = 1
best_score = 0
for k in [1, 3, 5, 7, 9, 11, 13, 15]:
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5, scoring='accuracy')
    mean_score = np.mean(scores)
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

# Обучение модели с лучшим k на всей обучающей выборке
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Вычисление accuracy на тестовой выборке
test_accuracy = knn.score(X_test, y_test)
print(f"METRIC: {test_accuracy:.1f}")
