# <ID> (эталон): <одна строка — что делает решение>.
# Правила эталона: разбиение до любых преобразований, препроцессинг внутри Pipeline,
# гиперпараметры подбираются только на обучающей части, stratify для классификации,
# random_state=42 везде, где есть случайность. Последняя строка вывода: METRIC: <число>.
from sklearn.model_selection import train_test_split

# X, y = ...
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
# ...
# print(f"METRIC: {metric:.4f}")
