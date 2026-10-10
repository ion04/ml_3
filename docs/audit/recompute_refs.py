"""Независимый пересчёт эталонов банка задач (аудит).

Каждый эталон считается здесь заново, «с нуля», без data/refs/*.py и data/d_reference/*.py.
Потом результат сравнивается с полем reference (A) или reference_metric (D) из банка
с тем же относительным допуском tolerance, что и при автопроверке.

Для D-задач считается одно «основное» решение и несколько альтернативных корректных решений
(со stratify / без, другая сетка гиперпараметров). Все они должны попасть в допуск — иначе
правильный код модели может быть засчитан как неверный.

Запуск: .venv/bin/python docs/audit/recompute_refs.py   (около минуты)
"""
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    """Задачи из JSONL-файла в словарь id → задача."""
    with open(ROOT / path, encoding="utf-8") as f:
        return {t["id"]: t for t in map(json.loads, f)}


# ---------- Тип A: расчёт по формулам ----------

def a1_bayes():
    # Формула Байеса: P(D|+) = P(+|D)P(D) / (P(+|D)P(D) + P(+|¬D)P(¬D))
    p_d, p_pos_d, p_pos_nd = 0.01, 0.95, 0.05
    return [p_pos_d * p_d / (p_pos_d * p_d + p_pos_nd * (1 - p_d))]


def gini(counts):
    n = sum(counts)
    return 1 - sum((c / n) ** 2 for c in counts)


def a2_gini_gain():
    parent, left, right = (40, 10), (35, 2), (5, 8)
    n = sum(parent)
    weighted = sum(left) / n * gini(left) + sum(right) / n * gini(right)
    return [gini(parent) - weighted]


def a3_confusion():
    tp, fp, fn, tn = 30, 10, 20, 940
    acc = (tp + tn) / (tp + fp + fn + tn)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    return [acc, prec, rec, f1]


def a4_svm():
    w, b, x = np.array([3, 4]), -2, np.array([1, 0])
    width = 2 / np.linalg.norm(w)          # расстояние между w·x+b = 1 и w·x+b = −1
    score = w @ x + b                      # = 1: точка ровно на границе зазора
    return [width, 1 if score > 0 else -1]


def a5_ols():
    x = np.array([1, 2, 3, 4, 5]); y = np.array([2, 4, 5, 4, 5])
    b1, b0 = np.polyfit(x, y, 1)           # МНК для прямой
    return [b0, b1, b0 + b1 * 6]


def a6_logreg():
    b0, b1 = -4.1, 1.5
    p3 = 1 / (1 + math.exp(-(b0 + b1 * 3)))
    return [p3, -b0 / b1]                  # p = 0,5 ⇔ b0 + b1·h = 0


def a7_knn():
    X = np.array([[1, 1], [2, 1], [4, 3], [5, 4], [1, 3], [3, 5]])
    y = np.array([2, 3, 6, 7, 4, 8])
    d = np.linalg.norm(X - np.array([2, 2]), axis=1)
    order = np.argsort(d)
    assert d[order[2]] < d[order[3]], "ничья на третьем соседе — ответ не единственный"
    return [y[order[:3]].mean()]


def a8_pca():
    lam = np.sort(np.linalg.eigvalsh(np.array([[2, 1], [1, 2]])))[::-1]
    return [lam[0], lam[1], lam[0] / lam.sum()]


def dev_a1_adaboost():
    eps = 0.2
    return [0.5 * math.log((1 - eps) / eps)]


A_TASKS = {"A1": a1_bayes, "A2": a2_gini_gain, "A3": a3_confusion, "A4": a4_svm, "A5": a5_ols,
           "A6": a6_logreg, "A7": a7_knn, "A8": a8_pca, "dev-A1": dev_a1_adaboost}


# ---------- Тип D: свои решения по условию + альтернативы ----------

def split(X, y, test_size, stratify):
    return train_test_split(X, y, test_size=test_size, random_state=42,
                            stratify=y if stratify else None)


def d1_variants():
    """SVM (RBF) на load_breast_cancer, подбор C на CV, accuracy на тесте 25%."""
    X, y = load_breast_cancer(return_X_y=True)
    res = {}
    for strat in (True, False):
        Xtr, Xte, ytr, yte = split(X, y, 0.25, strat)
        for grid_name, grid in (("C=0.01..1000", np.logspace(-2, 3, 11)), ("C=0.1,1,10,100", [0.1, 1, 10, 100])):
            # скейлер внутри Pipeline: масштаб считается только на обучающих фолдах
            pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
            gs = GridSearchCV(pipe, {"svc__C": grid}, cv=5).fit(Xtr, ytr)
            res[f"stratify={strat}, {grid_name}"] = accuracy_score(yte, gs.predict(Xte))
    return res


def d2_variants():
    """PCA(95% дисперсии) + логистическая регрессия на load_digits, accuracy на тесте 25%."""
    X, y = load_digits(return_X_y=True)
    res = {}
    for strat in (True, False):
        Xtr, Xte, ytr, yte = split(X, y, 0.25, strat)
        for scale in (False, True):
            steps = ([StandardScaler()] if scale else []) + [PCA(n_components=0.95, random_state=42),
                                                             LogisticRegression(max_iter=5000)]
            model = make_pipeline(*steps).fit(Xtr, ytr)   # PCA обучается только на train
            res[f"stratify={strat}, scaler={scale}"] = accuracy_score(yte, model.predict(Xte))
    return res


def d3_variants():
    """RandomForest на несбалансированных данных, подбор на CV по ROC AUC, ROC AUC на тесте."""
    X, y = make_classification(n_samples=5000, n_features=20, n_informative=5,
                               weights=[0.95, 0.05], random_state=42)
    grids = {
        "depth+leaf": {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 5]},
        "n_est+depth+feat": {"n_estimators": [100, 300], "max_depth": [None, 10],
                             "max_features": ["sqrt", 0.5]},
    }
    res = {}
    for strat in (True, False):
        Xtr, Xte, ytr, yte = split(X, y, 0.25, strat)
        for grid_name, grid in grids.items():
            cv = StratifiedKFold(5, shuffle=True, random_state=42)
            gs = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), grid,
                              scoring="roc_auc", cv=cv).fit(Xtr, ytr)
            # ROC AUC по вероятностям редкого класса, а не по predict
            res[f"stratify={strat}, {grid_name}"] = roc_auc_score(yte, gs.predict_proba(Xte)[:, 1])
    return res


def dev_d1_variants():
    """Дерево решений на load_iris, подбор max_depth на 5-fold CV, accuracy на тесте 30%."""
    X, y = load_iris(return_X_y=True)
    res = {}
    for strat in (True, False):
        Xtr, Xte, ytr, yte = split(X, y, 0.30, strat)
        for depths in (range(1, 11), [1, 2, 3, 4, 5, None]):
            gs = GridSearchCV(DecisionTreeClassifier(random_state=42), {"max_depth": list(depths)}, cv=5)
            gs.fit(Xtr, ytr)
            res[f"stratify={strat}, depths={list(depths)[-1]}"] = accuracy_score(yte, gs.predict(Xte))
    return res


D_TASKS = {"D1": d1_variants, "D2": d2_variants, "D3": d3_variants, "dev-D1": dev_d1_variants}


def close(a, b, tol):
    return abs(a - b) <= max(tol * abs(b), 1e-9)


def main():
    bank = {**load("data/tasks.jsonl"), **load("data/tasks_dev.jsonl")}
    print(f"{'id':8} {'мой расчёт':38} {'эталон банка':34} совпало")
    for tid, fn in A_TASKS.items():
        t = bank[tid]
        mine = fn()
        ref = t["reference"] if isinstance(t["reference"], list) else [t["reference"]]
        ok = len(mine) == len(ref) and all(close(m, r, t["tolerance"]) for m, r in zip(mine, ref))
        print(f"{tid:8} {str([round(float(m), 4) for m in mine]):38} {str(ref):34} {'да' if ok else 'НЕТ'}")
    for tid, fn in D_TASKS.items():
        t = bank[tid]
        start = time.time()
        variants = fn()
        ref, tol = t["reference_metric"], t["tolerance"]
        print(f"{tid:8} reference_metric = {ref}, допуск {tol} (отн.) → [{ref * (1 - tol):.4f}; {ref * (1 + tol):.4f}]"
              f"   ({time.time() - start:.1f} с)")
        for name, value in variants.items():
            print(f"{'':8}   {name:40} {value:.4f}  {'да' if close(value, ref, tol) else 'НЕТ'}")


if __name__ == "__main__":
    main()
