import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# Загрузка данных
data = load_diabetes()
X, y = data.data, data.target

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Набор значений alpha
alphas = [0.01, 0.1, 1, 10, 100]

# Подбор оптимального alpha по кросс-валидации (MAE)
best_alpha = None
best_mae_cv = float('inf')

for alpha in alphas:
    model = Ridge(alpha=alpha)
    # Используем negative MAE, так как cross_val_score максимизирует метрику
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    mae_cv = -scores.mean()
    if mae_cv < best_mae_cv:
        best_mae_cv = mae_cv
        best_alpha = alpha

# Обучение модели с лучшим alpha на всей обучающей выборке
final_model = Ridge(alpha=best_alpha)
final_model.fit(X_train, y_train)

# Предсказание и оценка на тестовой выборке
y_pred = final_model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)

# Вывод метрики в требуемом формате
print(f"METRIC: {test_mae:.4f}")
