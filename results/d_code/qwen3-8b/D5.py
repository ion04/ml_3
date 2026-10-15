from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import numpy as np

# Загрузка датасета
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Определение параметров для поиска
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}

# 5-блочная кросс-валидация для подбора alpha
grid_search = GridSearchCV(Ridge(), param_grid, scoring='neg_mean_absolute_error', cv=5)
grid_search.fit(X_train, y_train)

# Получение лучшего alpha
best_alpha = grid_search.best_params_['alpha']
best_model = Ridge(alpha=best_alpha)
best_model.fit(X_train, y_train)

# Вычисление MAE на тестовой выборке
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"METRIC: {mae}")
