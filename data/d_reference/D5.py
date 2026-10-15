# D5 (эталон): load_diabetes, Ridge, подбор alpha из [0.01, 0.1, 1, 10, 100] на 5-блочной кросс-валидации по MAE,
# MAE на тесте. Методологически корректно: разбиение до любых преобразований, скейлер внутри Pipeline,
# подбор alpha только на обучающей части, фиксированный random_state (стратификация для регрессии не нужна).
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = make_pipeline(StandardScaler(), Ridge())
grid = {"ridge__alpha": [0.01, 0.1, 1, 10, 100]}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipe, grid, cv=cv, scoring="neg_mean_absolute_error").fit(X_train, y_train)

mae = -search.score(X_test, y_test)          # scorer возвращает −MAE
print("лучшее alpha:", search.best_params_["ridge__alpha"])
print(f"METRIC: {mae:.4f}")
