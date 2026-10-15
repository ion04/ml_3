from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# Загрузка данных
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Определение списка альфа-значений для подбора
alphas = [0.01, 0.1, 1, 10, 100]
best_alpha = None
best_mae = float('inf')

# Поиск оптимального значения alpha через кросс-валидацию
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    maes = -cross_val_score(ridge, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
    avg_mae = maes.mean()

    if avg_mae < best_mae:
        best_mae = avg_mae
        best_alpha = alpha

# Обучение модели с оптимальным alpha на всей обучающей выборке
final_ridge = Ridge(alpha=best_alpha)
final_ridge.fit(X_train, y_train)

# Вычисление MAE на тестовой выборке
y_pred = final_ridge.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)

# Печать результата
print(f"METRIC: {test_mae:.3f}")
