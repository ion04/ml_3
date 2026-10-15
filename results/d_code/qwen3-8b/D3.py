from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

# Генерация данных
X, y = make_classification(n_samples=5000, n_features=20, n_informative=5, 
                           weights=[0.95, 0.05], random_state=42)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Определение параметров для поиска
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Инициализация модели и GridSearchCV
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

# Обучение модели
grid_search.fit(X_train, y_train)

# Получение лучшей модели
best_model = grid_search.best_estimator_

# Предсказание на тестовой выборке
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Вычисление ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Вывод результата
print(f"ROC AUC на тестовой выборке: {roc_auc:.4f}")
