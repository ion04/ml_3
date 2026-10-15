from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Генерация синтетических данных
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=5,
    weights=[0.95, 0.05],
    random_state=42
)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Подбор параметров модели через кросс-валидацию
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Лучшая модель
best_model = grid_search.best_estimator_

# Предсказания вероятностей класса 1
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Вычисление ROC-AUC на тестовой выборке
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Печать результата
print(f'ROC-AUC на тестовой выборке: {roc_auc:.4f}')
