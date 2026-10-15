from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Загрузка датасета
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Настройка SVM с RBF-ядром и подбор параметра C через кросс-валидацию
param_grid = {'C': [0.1, 1, 10, 100]}
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Определение лучшего значения C и обучение модели с этим параметром
best_C = grid_search.best_params_['C']
best_model = SVC(kernel='rbf', C=best_C)
best_model.fit(X_train, y_train)

# Оценка точности на тестовой выборке
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Вывод точности
print(f'Точность на тестовой выборке: {accuracy:.4f}')
