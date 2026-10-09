# dev-D1 (эталон): load_iris, дерево решений, подбор max_depth на 5-fold CV, accuracy на тесте.
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(DecisionTreeClassifier(random_state=42), {"max_depth": [1, 2, 3, 4, 5, None]}, cv=cv)
search.fit(X_train, y_train)

print("лучшая глубина:", search.best_params_["max_depth"])
print(f"METRIC: {search.score(X_test, y_test):.4f}")
