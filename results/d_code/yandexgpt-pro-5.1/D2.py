from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Загрузка данных
digits = load_digits()
X, y = digits.data, digits.target

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Создание пайплайна: PCA + логистическая регрессия
pipeline = make_pipeline(PCA(n_components=0.95), LogisticRegression(max_iter=10000, random_state=42))

# Обучение модели
pipeline.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = pipeline.predict(X_test)

# Вычисление accuracy
accuracy = accuracy_score(y_test, y_pred)

# Вывод результата
print(f"Accuracy: {accuracy}")
