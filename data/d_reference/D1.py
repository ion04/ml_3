# D1 (эталон): load_breast_cancer, SVC с RBF-ядром, подбор C на кросс-валидации, accuracy на тесте.
# Методологически корректно: разбиение до любых преобразований, скейлер внутри Pipeline,
# подбор C только на обучающей части (GridSearchCV), стратификация, фиксированный random_state.
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
grid = {"svc__C": [0.01, 0.1, 1, 10, 100]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy").fit(X_train, y_train)

print("лучшее C:", search.best_params_["svc__C"])
print(f"METRIC: {search.score(X_test, y_test):.4f}")
