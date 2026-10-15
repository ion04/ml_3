from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import numpy as np

# Загрузка данных
data = load_diabetes()
X, y = data.data, data.target

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Определение набора значений alpha для поиска оптимального
alphas = [0.01, 0.1, 1, 10, 100]

# Инициализация объекта для 5-блочной кросс-валидации
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Поиск оптимального alpha
best_alpha = None
best_mae = np.inf
for alpha in alphas:
    model = Ridge(alpha=alpha)
    mae_scores = -cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
    mean_mae = np.mean(mae_scores)
    if mean_mae < best_mae:
        best_mae = mean_mae
        best_alpha = alpha

# Обучение модели с лучшим alpha на всей обучающей выборке
best_model = Ridge(alpha=best_alpha)
best_model.fit(X_train, y_train)

# Прогноз на тестовой выборке и вычисление MAE
y_pred = best_model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)

# Вывод итоговой метрики
print(f'METRIC: {test_mae}')
