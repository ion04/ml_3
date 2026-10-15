import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Загрузка данных
data = load_breast_cancer()
X, y = data.data, data.target

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Пайплайн: стандартизация + SVM с RBF-ядром
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', random_state=42))
])

# Сетка параметров для подбора C
param_grid = {
    'svm__C': [0.01, 0.1, 1, 10, 100]
}

# Кросс-валидация для подбора C
grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Лучшая модель и предсказание на тестовой выборке
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Итоговая точность
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
