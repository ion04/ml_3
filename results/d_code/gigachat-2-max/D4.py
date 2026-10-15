from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Загрузка данных
data = load_wine()
X, y = data.data, data.target

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Определение возможных значений k
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
best_k = None
best_accuracy = 0

# Поиск лучшего значения k через кросс-валидацию
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    mean_accuracy = np.mean(scores)

    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_k = k

# Обучение модели с оптимальным k на всей обучающей выборке
best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(X_train, y_train)

# Вычисление точности на тестовой выборке
test_accuracy = best_model.score(X_test, y_test)

# Печать результата
print(f'METRIC: {test_accuracy}')
