# D3 (эталон): несбалансированные данные (5% положительного класса), RandomForest,
# подбор гиперпараметров на кросс-валидации по ROC AUC (а не по accuracy), ROC AUC на тесте.
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

X, y = make_classification(n_samples=5000, n_features=20, n_informative=5,
                           weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

grid = {"n_estimators": [100, 300], "max_depth": [None, 10]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), grid, cv=cv, scoring="roc_auc")
search.fit(X_train, y_train)

print("лучшие параметры:", search.best_params_)
proba = search.predict_proba(X_test)[:, 1]   # вероятность редкого класса
print(f"METRIC: {roc_auc_score(y_test, proba):.4f}")
