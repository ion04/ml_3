import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Загрузка данных
X, y = load_digits(return_X_y=True)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Создание пайплайна: PCA (95% дисперсии) + логистическая регрессия
pipeline = Pipeline([
    ('pca', PCA(n_components=0.95, random_state=42)),
    ('logreg', LogisticRegression(max_iter=10000, random_state=42))
])

# Обучение модели
pipeline.fit(X_train, y_train)

# Предсказание и оценка точности
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
