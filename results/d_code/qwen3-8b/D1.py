from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# Загрузка датасета
data = load_breast_cancer()
X, y = data.data, data.target

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Определение параметров для поиска
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto']
}

# Создание модели SVM с RBF-ядром
svm = SVC(kernel='rbf')

# Кросс-валидация для подбора параметра C
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Получение лучшей модели
best_model = grid_search.best_estimator_

# Предсказание на тестовой выборке
y_pred = best_model.predict(X_test)

# Вычисление и вывод точности
accuracy = accuracy_score(y_test, y_pred)
print(f"Итоговая точность (accuracy): {accuracy:.4f}")
