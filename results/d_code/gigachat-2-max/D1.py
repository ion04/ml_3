from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Загружаем датасет
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Делим на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Создаем модель SVM с RBF-ядром
model = SVC(kernel='rbf')

# Подбираем параметр C через кросс-валидацию
param_grid = {'C': [0.1, 1, 10, 100]}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Лучшая найденная модель
best_model = grid_search.best_estimator_

# Предсказания на тестовой выборке
y_pred = best_model.predict(X_test)

# Вычисляем точность модели
accuracy = accuracy_score(y_test, y_pred)

# Печатаем результат
print(f"Точность модели на тестовой выборке: {accuracy:.3f}")

# Ответ и уверенность
ОТВЕТ: код выше
УВЕРЕННОСТЬ: 100
