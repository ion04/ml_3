# D4 (эталон): load_wine, kNN, подбор k из [1, 3, ..., 15] на 5-блочной кросс-валидации, accuracy на тесте.
# Методологически корректно: разбиение до любых преобразований, скейлер внутри Pipeline
# (kNN считает расстояния, а признаки wine в разных масштабах), подбор k только на обучающей части,
# стратификация, фиксированный random_state.
from sklearn.datasets import load_wine
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier())
grid = {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy").fit(X_train, y_train)

print("лучшее k:", search.best_params_["kneighborsclassifier__n_neighbors"])
print(f"METRIC: {search.score(X_test, y_test):.4f}")
