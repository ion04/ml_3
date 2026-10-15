from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Генерация данных
X, y = make_classification(n_samples=5000, n_features=20, n_informative=5, weights=[0.95, 0.05], random_state=42)

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Настройка гиперпараметров с помощью GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Получение лучшего классификатора
best_clf = grid_search.best_estimator_

# Прогнозирование на тестовой выборке
y_pred_proba = best_clf.predict_proba(X_test)[:, 1]

# Расчет ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Вывод результата
print(f'ROC AUC на тестовой выборке: {roc_auc}')
