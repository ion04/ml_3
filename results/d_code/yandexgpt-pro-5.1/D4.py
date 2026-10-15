from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Загрузка данных
data = load_wine()
X, y = data.data, data.target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Подбор оптимального числа соседей k
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
best_k = None
best_score = 0

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    mean_score = scores.mean()
    if mean_score > best_score:
        best_score = mean_score
        best_k = k

# Обучение модели с лучшим значением k на всей обучающей выборке
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

# Вычисление accuracy на тестовой выборке
y_pred = knn_best.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Вывод итоговой метрики
print(f'METRIC: {accuracy}')
