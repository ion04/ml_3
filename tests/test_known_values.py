"""Проверка формул анализа на известных значениях (посчитаны вручную в комментариях).

Дополняет tests/test_analysis.py, test_parse.py, test_grade.py, test_typeE.py, test_typeE_metrics.py:
там уже есть базовые случаи, здесь — ручные расчёты, граничные случаи и сверка с независимой реализацией
(numpy, scipy, sklearn, statsmodels).
"""

import inspect
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

from src import analysis as an, blind, typeE, typeE_metrics
from src.grade import close, grade_auto, number_matches
from src.parse import parse_answer, pick_number, to_float


# ---------- Бутстреп ----------

def test_bootstrap_defaults_are_1000_and_seed_42():
    # Задание: 1000 ресэмплов. Перцентили 2,5/97,5 проверяются ниже сверкой с numpy.
    for f in (an.cluster_bootstrap, an.bootstrap_stat):
        params = inspect.signature(f).parameters
        assert params["n_boot"].default == 1000 and params["seed"].default == 42


def test_cluster_bootstrap_matches_independent_numpy():
    # Задачи с разным числом строк (ответов). Независимая реализация: те же номера задач от того же
    # генератора, потом явно собираем все строки выбранных задач и берём среднее.
    df = pd.DataFrame({"task_id": ["a", "a", "a", "b", "c", "c", "d", "e"],
                       "correct": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0]})
    tasks = sorted(df["task_id"].unique())                     # groupby сортирует задачи
    idx = np.random.default_rng(7).integers(0, len(tasks), size=(1000, len(tasks)))
    means = []
    for row in idx:
        picked = pd.concat([df[df["task_id"] == tasks[i]] for i in row])
        means.append(picked["correct"].mean())
    expected = np.percentile(means, [2.5, 97.5])
    assert an.cluster_bootstrap(df, "correct", seed=7) == pytest.approx(tuple(expected))


def test_cluster_bootstrap_resamples_tasks_not_answers():
    # Задача A: 9 верных ответов, задача B: 1 неверный. Бутстреп задач даёт только три значения:
    # AA → 1, BB → 0, AB → 9/10 = 0,9. P(BB) = P(AA) = 1/4 > 2,5%, поэтому CI = [0; 1].
    # Если бы ресэмплировались 10 ответов, CI был бы узким около 0,9 (примерно [0,7; 1]).
    df = pd.DataFrame({"task_id": ["A"] * 9 + ["B"], "correct": [1.0] * 9 + [0.0]})
    assert an.cluster_bootstrap(df, "correct") == (0.0, 1.0)


def test_degenerate_bootstrap_all_correct():
    # Все ответы верны: любой ресэмпл даёт 1 → CI [1; 1]. Все неверны → [0; 0].
    df = pd.DataFrame({"task_id": list("abcde"), "correct": 1.0})
    assert an.cluster_bootstrap(df, "correct") == (1.0, 1.0)
    assert an.cluster_bootstrap(df.assign(correct=0.0), "correct") == (0.0, 0.0)
    assert an.bootstrap_stat([1, 1, 1], np.mean) == (1.0, 1.0)


def test_bootstrap_stat_matches_independent_numpy():
    # Та же схема, что в bootstrap_stat: на каждой итерации n номеров с возвращением, потом перцентили 2,5/97,5.
    units = [0.2, 0.9, 0.4, 0.7, 0.1, 0.6]
    rng = np.random.default_rng(3)
    vals = [np.mean([units[i] for i in rng.integers(0, len(units), size=len(units))]) for _ in range(1000)]
    assert an.bootstrap_stat(units, np.mean, seed=3) == pytest.approx(tuple(np.percentile(vals, [2.5, 97.5])))


def test_acc_table_accuracy_and_ci():
    # Модель a: 3 задачи, верно 2 → accuracy 2/3; модель b: всё верно → CI [1; 1].
    # Отказ (abstain) в a: accuracy_answered = 2/2 = 1 (отказ не в знаменателе).
    df = pd.DataFrame({"model_key": ["a"] * 3 + ["b"] * 3, "task_id": ["T1", "T2", "T3"] * 2,
                       "correct": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                       "refusal": [False, False, True] + [False] * 3,
                       "abstain": [False, False, True] + [False] * 3})
    t = an.acc_table(df, ["model_key"], 1000, 42).set_index("model_key")
    assert t.loc["a", "accuracy"] == pytest.approx(2 / 3) and t.loc["a", "accuracy_answered"] == 1.0
    assert t.loc["a", "refusal_rate"] == pytest.approx(1 / 3)
    assert (t.loc["a", "ci_low"], t.loc["a", "ci_high"]) == an.cluster_bootstrap(df[df["model_key"] == "a"], "correct")
    assert (t.loc["b", "ci_low"], t.loc["b", "ci_high"]) == (1.0, 1.0)


# ---------- Макнемар ----------

def binom_two_sided(b, c):
    """Точный двусторонний биномиальный тест вручную: 2·P(X ≤ min(b, c)), X ~ Bin(b + c, 0,5), не больше 1."""
    n = b + c
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


@pytest.mark.parametrize("b,c", [(6, 0), (5, 1), (3, 3), (7, 2), (0, 0), (1, 0), (10, 4)])
def test_mcnemar_p_value_by_hand(b, c):
    # Пример (7, 2): n = 9, P(X ≤ 2) = (1 + 9 + 36)/512 = 46/512 → p = 92/512 = 0,1797.
    # (3, 3): 2·P(X ≤ 3) > 1 → обрезается до 1. (0, 0): разногласий нет → p = 1.
    # Совпадающие пары (оба верно 4, оба неверно 2) на p не влияют.
    x = [1] * b + [0] * c + [1] * 4 + [0] * 2
    y = [0] * b + [1] * c + [1] * 4 + [0] * 2
    t = an.mcnemar_test(x, y)
    assert (t["only_first"], t["only_second"], t["both_correct"], t["both_wrong"]) == (b, c, 4, 2)
    assert t["p_value"] == pytest.approx(binom_two_sided(b, c))
    assert t["p_value"] == pytest.approx(mcnemar([[4, b], [c, 2]], exact=True).pvalue)
    if b + c:
        assert t["p_value"] == pytest.approx(stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    assert binom_two_sided(7, 2) == pytest.approx(92 / 512)


def test_mcnemar_excludes_nan_and_half():
    # Пары, где хоть одно значение NaN или 0,5 (проверяющие разошлись), выбрасываются целиком.
    t = an.mcnemar_test([1, 1, 0.5, np.nan, 1, 0], [0, 0, 0, 1, 0.5, 1])
    assert t["n"] == 3 and (t["only_first"], t["only_second"]) == (2, 1)
    assert t["acc_first"] == pytest.approx(2 / 3) and t["acc_second"] == pytest.approx(1 / 3)
    empty = an.mcnemar_test([0.5, np.nan], [1, 1])
    assert empty["n"] == 0 and np.isnan(empty["p_value"])


def test_mcnemar_pairs_holm_by_hand():
    # 8 задач. a верна везде, b неверна в T0–T5, c неверна в T0–T2; d — только 0,5 (нет пар для теста).
    # a–b: только a верна 6 раз → p = 2·0,5^6 = 0,03125
    # a–c: только a верна 3 раза → p = 2·0,5^3 = 0,25
    # b–c: только c верна в T3–T5 (3 раза) → p = 0,25
    # Холм (m = 3, пары с d не считаются): по возрастанию 0,03125·3 = 0,09375; 0,25·2 = 0,5; 0,25·1 = 0,25 →
    # монотонность (не меньше предыдущего): 0,09375; 0,5; 0,5.
    marks = {"a": [1] * 8, "b": [0] * 6 + [1] * 2, "c": [0] * 3 + [1] * 5, "d": [0.5] * 8}
    df = pd.DataFrame([{"model_key": m, "task_id": f"T{i}", "correct": float(v)}
                       for m, vs in marks.items() for i, v in enumerate(vs)])
    t = an.mcnemar_pairs(df, ["a", "b", "c", "d"]).set_index(["model_a", "model_b"])
    assert t.loc[("a", "b"), "p_value"] == pytest.approx(0.03125)
    assert t.loc[("a", "c"), "p_value"] == pytest.approx(0.25)
    assert t.loc[("b", "c"), "p_value"] == pytest.approx(0.25)
    assert t.loc[("a", "b"), "p_holm"] == pytest.approx(0.09375)
    assert t.loc[("a", "c"), "p_holm"] == pytest.approx(0.5)
    assert t.loc[("b", "c"), "p_holm"] == pytest.approx(0.5)
    assert t.loc[("a", "d"), "n"] == 0 and np.isnan(t.loc[("a", "d"), "p_holm"])


# ---------- ECE ----------

def test_ece_ten_answers_with_boundaries():
    # Корзины [0; 20), [20; 40), [40; 60), [60; 80), [80; 100]; границы 20/40/60/80 — в верхнюю корзину, 100 — в последнюю.
    # 0: {10}          ср. уверенность 0,10, верно 0/1 → |0,10 − 0|   = 0,10, вес 1/10 → 0,010
    # 1: {20, 30}      0,25, верно 1/2            → |0,25 − 0,5| = 0,25, вес 2/10 → 0,050
    # 2: {40, 55}      0,475, верно 2/2           → 0,525,       вес 2/10 → 0,105
    # 3: {60, 70}      0,65, верно 1/2            → 0,15,        вес 2/10 → 0,030
    # 4: {80, 90, 100} 0,90, верно 3/3            → 0,10,        вес 3/10 → 0,030
    # ECE = 0,010 + 0,050 + 0,105 + 0,030 + 0,030 = 0,225
    conf = [10, 20, 30, 40, 55, 60, 70, 80, 90, 100]
    correct = [0, 0, 1, 1, 1, 0, 1, 1, 1, 1]
    value, table = an.ece(conf, correct)
    assert value == pytest.approx(0.225)
    assert table["n"].tolist() == [1, 2, 2, 2, 3]
    assert table["mean_confidence"].tolist() == pytest.approx([0.10, 0.25, 0.475, 0.65, 0.90])
    assert table["accuracy"].tolist() == pytest.approx([0.0, 0.5, 1.0, 0.5, 1.0])
    assert an.ece(conf, correct, table=False) == pytest.approx(0.225)


def test_ece_expects_percent_scale():
    # ece принимает уверенность в шкале 0–100 (так её отдаёт парсер: «0.9» → 90, см. ниже).
    # Доля 0,9 была бы понята как 0,9% → корзина 0 и ECE ≈ 1 вместо 0,1.
    assert an.ece([90], [1], table=False) == pytest.approx(0.1)
    assert an.ece([0.9], [1], table=False) == pytest.approx(1 - 0.009)
    assert parse_answer("ANSWER: 1\nCONFIDENCE: 0.9", "number", "en").confidence == 90
    assert parse_answer("ОТВЕТ: 1\nУВЕРЕННОСТЬ: 0,4", "number", "ru").confidence == pytest.approx(40)
    # Доля, переведённая парсером в проценты, попадает в ту же корзину, что и целое число (0,4 → 40 → корзина 2).
    for share in ["0.2", "0.4", "0.6", "0.8", "1.0"]:
        pct = parse_answer(f"ANSWER: 1\nCONFIDENCE: {share}", "number", "en").confidence
        _, t = an.ece([pct], [1])
        assert t["n"].tolist().index(1) == min(int(float(share) * 5), 4)


def test_ece_empty_and_perfect():
    assert np.isnan(an.ece([], [], table=False))
    # Уверенность 100 и всё верно, уверенность 0 и всё неверно → ECE = 0.
    assert an.ece([100, 100, 0], [1, 1, 0], table=False) == pytest.approx(0.0)


# ---------- Каппа Коэна ----------

def test_kappa_by_hand():
    # Баллы 0–2 двух проверяющих, 8 ответов; совпали 6 → po = 6/8 = 0,75.
    # Частоты у обоих: «0» — 2, «1» — 3, «2» — 3 → pe = (2·2 + 3·3 + 3·3)/8² = 22/64 = 0,34375.
    # κ = (0,75 − 0,34375)/(1 − 0,34375) = 0,40625/0,65625 = 13/21 ≈ 0,619.
    a = ["0", "1", "2", "2", "1", "0", "2", "1"]
    b = ["0", "1", "2", "1", "1", "0", "2", "2"]
    assert an._kappa(a, b) == pytest.approx(13 / 21)
    assert an._kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))


def test_kappa_degenerate():
    # Оба всегда ставят «2»: po = 1 и pe = 1 → κ = 0/0 не определена → NaN (sklearn даёт nan с предупреждением).
    # В kappa_table согласие 1,0, каппа NaN, below_min = False (NaN < 0,6 ложно) — смотреть на agreement.
    assert np.isnan(an._kappa(["2"] * 5, ["2"] * 5))
    # Один проверяющий всегда «1», другой нет: pe = po → κ = 0 (определена).
    assert an._kappa(["1", "1", "1", "1"], ["1", "0", "1", "0"]) == pytest.approx(0.0)
    assert np.isnan(an._kappa([], []))
    col = blind.SCORE_COL
    m = pd.DataFrame({"type": ["C"] * 3, "required": [[col]] * 3, f"{col}_r1": ["2"] * 3, f"{col}_r2": ["2"] * 3})
    t = an.kappa_table(m).set_index("type").loc["C"]
    assert t["n_pairs"] == 3 and t["agreement"] == 1.0 and np.isnan(t["kappa"]) and not t["below_min"]


def test_kappa_table_by_hand():
    # Тип B: те же 8 пар, что в test_kappa_by_hand, плюс одна строка, где r2 не заполнил (None) — не входит.
    col = blind.SCORE_COL
    a = ["0", "1", "2", "2", "1", "0", "2", "1", "2"]
    b = ["0", "1", "2", "1", "1", "0", "2", "2", None]
    # dtype=object, как в load_manual: иначе pandas 3 превратит None в NaN.
    m = pd.DataFrame({"type": ["B"] * 9, "required": [[col]] * 9, f"{col}_r1": a, f"{col}_r2": b}, dtype=object)
    t = an.kappa_table(m).set_index("type").loc["B"]
    assert t["n_pairs"] == 8 and t["agreement"] == pytest.approx(0.75) and t["kappa"] == pytest.approx(13 / 21)
    assert t["below_min"] == (13 / 21 < an.KAPPA_MIN)


# ---------- F1 и balanced accuracy (тип E) ----------

# 10 строк: 3 отказа (1), 7 «нет отказа» (0). Предсказано: TP = 2, FN = 1, FP = 1, TN = 6.
Y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
P = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])


def test_f1_and_balanced_accuracy_by_hand():
    # F1 = 2·TP/(2·TP + FP + FN) = 4/6 = 0,6667; precision = 2/3, recall = 2/3.
    # BA = (TP/(TP+FN) + TN/(TN+FP))/2 = (2/3 + 6/7)/2 = 32/42 = 0,7619; accuracy = 8/10.
    m = typeE.metrics(Y, P)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 6)
    assert m["f1_failure"] == pytest.approx(4 / 6) and an.f1_failure(Y, P) == pytest.approx(4 / 6)
    assert m["balanced_accuracy"] == pytest.approx(32 / 42) and an.balanced_acc(Y, P) == pytest.approx(32 / 42)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["precision_failure"] == pytest.approx(2 / 3) and m["recall_failure"] == pytest.approx(2 / 3)
    assert m["f1_failure"] == pytest.approx(f1_score(Y, P, pos_label=1))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(Y, P))


def test_constant_no_failure_baseline():
    # Константа «нет отказа»: accuracy = доля нулей = 7/10, BA = (0 + 1)/2 = 0,5, F1 = 0 (TP = 0).
    m = typeE.metrics(Y, np.zeros_like(Y))
    assert m["accuracy"] == pytest.approx(0.7) and m["balanced_accuracy"] == 0.5 and m["f1_failure"] == 0.0


def test_llm_group_non_answer_is_error_by_hand():
    # Ответы LLM: тот же P, но строка 2 (y = 1) — «нельзя определить», строка 3 (y = 0) — не разобрано.
    # Неответ = ошибка: на y = 1 → предсказание 0 (FN), на y = 0 → 1 (FP). Получается ровно P → те же метрики.
    # Только ответившие (8 строк: y = 1,1,0,0,0,0,0,0; p = 1,1,0,0,0,0,0,0) → F1 = 1, BA = 1, accuracy = 1.
    values = [1, 1, "cannot_determine", None, 0, 0, 0, 0, 0, 0]
    group = [{"y_true": int(y), "value": v, "error": None, "tokens_in": 10, "tokens_out": 1, "latency_s": 1.0,
              "model_reported": "m", "prompt_version": "v1"} for y, v in zip(Y, values)]
    rec = typeE_metrics.llm_group(group)
    assert (rec["tp"], rec["fn"], rec["fp"], rec["tn"]) == (2, 1, 1, 6)
    assert rec["f1_failure"] == pytest.approx(4 / 6) and rec["balanced_accuracy"] == pytest.approx(32 / 42)
    assert rec["coverage"] == pytest.approx(0.8)
    assert rec["f1_failure_answered"] == 1.0 and rec["balanced_accuracy_answered"] == 1.0
    assert rec["accuracy_answered"] == 1.0


# ---------- ₽ за правильный ответ ----------

MODELS = {"m": {"role": "ru1", "provider": "gigachat", "enabled": True, "model": "x"},
          "free": {"role": "local", "provider": "ollama", "enabled": True, "model": "q"},
          "noprice": {"role": "big", "provider": "openai_compat", "enabled": True, "model": "y"}}
PRICING = {"models": {"m": {"rub_per_1m_in": 150.0, "rub_per_1m_out": 600.0},
                      "free": {"rub_per_1m_in": 0.0, "rub_per_1m_out": 0.0},
                      "noprice": {"rub_per_1m_in": None, "rub_per_1m_out": 1.0}}}


def cost_rows(model, correct, tin=2000, tout=500):
    return pd.DataFrame([{"model_key": model, "task_id": f"T{i}", "correct": c, "tokens_in": tin,
                          "tokens_out": tout, "latency_s": 1.0} for i, c in enumerate(correct)])


def test_cost_by_hand():
    # Один ответ: 2000·150/1e6 + 500·600/1e6 = 0,3 + 0,3 = 0,6 ₽. 4 ответа → 2,4 ₽; верных 1 + 1 + 0,5 = 2,5
    # (0,5 — проверяющие разошлись) → 2,4/2,5 = 0,96 ₽ за правильный ответ.
    assert an.price_for("m", PRICING, MODELS) == (150.0, 600.0)
    df = cost_rows("m", [1.0, 1.0, 0.5, 0.0])
    assert an.add_cost(df, PRICING, MODELS)["cost_rub"].tolist() == pytest.approx([0.6] * 4)
    t = an.cost_table(df, PRICING, MODELS, 1000, 42).iloc[0]
    assert t["cost_rub_total"] == pytest.approx(2.4) and t["cost_rub_per_task"] == pytest.approx(0.6)
    assert t["n_correct"] == 2.5 and t["rub_per_correct"] == pytest.approx(0.96)
    assert t["rub_per_correct_ci_low"] <= 0.96 <= t["rub_per_correct_ci_high"]


def test_cost_zero_correct_and_missing_price():
    # 0 верных: деление на 0 не выполняется → rub_per_correct = NaN и CI (NaN, NaN), а не inf.
    t = an.cost_table(cost_rows("m", [0.0, 0.0]), PRICING, MODELS, 100, 42).iloc[0]
    assert t["cost_rub_total"] == pytest.approx(1.2) and np.isnan(t["rub_per_correct"])
    assert np.isnan(t["rub_per_correct_ci_low"]) and np.isnan(t["rub_per_correct_ci_high"])
    # Локальная модель с нулевым тарифом: 0 ₽ за правильный ответ.
    assert an.cost_table(cost_rows("free", [1.0, 0.0]), PRICING, MODELS, 100, 42).iloc[0]["rub_per_correct"] == 0.0
    # Нет одной из цен → price_for = None → стоимость NaN, ₽ за правильный ответ NaN.
    assert an.price_for("noprice", PRICING, MODELS) is None
    t = an.cost_table(cost_rows("noprice", [1.0]), PRICING, MODELS, 100, 42).iloc[0]
    assert np.isnan(t["cost_rub_total"]) and np.isnan(t["rub_per_correct"])


# ---------- Clopper–Pearson ----------

@pytest.mark.parametrize("k,n", [(0, 10), (3, 10), (10, 10), (6, 12), (1, 200)])
def test_clopper_pearson_matches_scipy_and_beta(k, n):
    # Точный CI: нижняя граница — квантиль 2,5% Beta(k, n − k + 1), верхняя — 97,5% Beta(k + 1, n − k);
    # при k = 0 нижняя 0, при k = n верхняя 1.
    lo = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    assert an.clopper_pearson(k, n) == pytest.approx((lo, hi))
    assert an.clopper_pearson(k, n) == pytest.approx((ci.low, ci.high))


def test_clopper_pearson_empty():
    assert all(np.isnan(an.clopper_pearson(0, 0)))


# ---------- Разбор чисел и допуск ----------

@pytest.mark.parametrize("text,lang,value,pct", [
    ("16%", "ru", 0.16, True), ("0,16", "ru", 0.16, False), ("0.161", "ru", 0.161, False),
    ("1 551", "ru", 1551.0, False), ("1,551", "ru", 1.551, False), ("1,551", "en", 1551.0, False),
    ("1.5e-3", "ru", 0.0015, False), ("1,5e-3", "ru", 0.0015, False),
    ("−0,5", "ru", -0.5, False), ("≈ 0.97", "ru", 0.97, False), ("≈ 0.97", "en", 0.97, False),
])
def test_parse_number_forms(text, lang, value, pct):
    label = "ОТВЕТ" if lang == "ru" else "ANSWER"
    r = parse_answer(f"{label}: {text}", "number", lang)
    assert r.value == pytest.approx(value) and r.had_percent is pct


def test_pick_number_and_to_float():
    # pick_number берёт первое число после последнего «=», «≈», «→» или «:»; проценты → доли.
    assert pick_number("P = 0,0095/0,059 ≈ 0,161", "ru")[0] == pytest.approx(0.161)
    assert pick_number("16%", "ru")[:2] == (pytest.approx(0.16), True)
    assert to_float("-1,551", "en") == -1551.0 and to_float("-1,551", "ru") == pytest.approx(-1.551)


@pytest.mark.parametrize("answer,reference,ok", [
    # Ровно 1% входит в допуск при любом представлении дробей в компьютере (ошибка была: 0.505 vs 0.5 — нет).
    (0.505, 0.5, True), (0.495, 0.5, True), (1.01, 1.0, True), (0.99, 1.0, True), (101, 100, True),
    (0.16261, 0.161, True),            # 0,161·1,01
    (0.5051, 0.5, False), (0.4949, 0.5, False), (101.01, 100, False),
    (0.01, 0, True), (-0.01, 0, True), (0.0101, 0, False),     # эталон 0 → допуск абсолютный ±0,01
    (-0.5, -0.5, True), (-0.506, -0.5, False),
])
def test_close_one_percent_boundary(answer, reference, ok):
    assert close(answer, reference, 0.01) is ok


def test_percent_vs_fraction_matching():
    # «16%» парсер даёт как 0,16 → совпадает с эталоном 0,16; при эталоне в процентах (16) сравнивается 16.
    assert number_matches(0.16, 0.16, 0.01, had_percent=True)
    assert number_matches(0.16, 16, 0.01, had_percent=True)
    assert not number_matches(16, 0.16, 0.01, had_percent=False)     # 16 без % при эталоне 0,16 — неверно
    task = {"id": "X", "type": "A", "answer_kind": "number", "reference": 0.5, "tolerance": 0.01}
    assert grade_auto(task, parse_answer("ОТВЕТ: 50,5%", "number", "ru"))["correct"] is True    # ровно +1%
    assert grade_auto(task, parse_answer("ОТВЕТ: 50,6%", "number", "ru"))["correct"] is False


# ---------- CI у каждой точности (шаг 13): условия гипотез, тип D, тип E ----------

def test_compare_conditions_ci_for_each_condition():
    # 4 задачи; условие a верно на 3, b — на 1. CI каждой точности — тот же бутстреп задач, что в acc_table.
    pairs = pd.DataFrame({"model_key": "m", "task_id": ["T1", "T2", "T3", "T4"],
                          "correct_a": [1.0, 1.0, 1.0, 0.0], "correct_b": [1.0, 0.0, 0.0, 0.0]})
    t = an.compare_conditions(pairs, None, 1000, 42).iloc[0]
    assert t["acc_a"] == 0.75 and t["acc_b"] == 0.25
    a = pairs.rename(columns={"correct_a": "correct"})
    assert (t["acc_a_ci_low"], t["acc_a_ci_high"]) == an.cluster_bootstrap(a, "correct")
    assert t["acc_b_ci_low"] <= 0.25 <= t["acc_b_ci_high"]
    named = an.name_conditions(an.compare_conditions(pairs, None, 100, 42), "cot", "main")
    assert {"acc_cot", "acc_cot_ci_low", "acc_main_ci_high", "diff_cot_minus_main",
            "only_cot_correct", "only_main_correct"} <= set(named.columns)


def test_typeD_table_ci():
    # 4 программы на 4 задачах: запустились 3 → pass@1 = 0,75 с CI бутстрепа задач.
    d = pd.DataFrame({"type": "D", "model_key": "m", "task_id": ["D1", "D2", "D3", "D4"],
                      "pass_at_1": [True, True, True, False], "correct": [1.0, 0.0, 1.0, 0.0],
                      "any_error_mean": [0.0, 1.0, np.nan, 0.5],
                      **{f"{c}_mean": np.nan for c in blind.CHECKLIST}})
    t = an.typeD_table(d).set_index("model_key").loc["все модели"]
    assert t["pass_at_1"] == 0.75 and t["correct"] == 0.5
    assert (t["pass_at_1_ci_low"], t["pass_at_1_ci_high"]) == an.cluster_bootstrap(
        d.assign(p=d["pass_at_1"].astype(float)), "p")
    assert t["method_error_any_ci_low"] <= t["method_error_any"] <= t["method_error_any_ci_high"]


def test_typeE_ci_table_by_hand():
    # Те же 10 строк, что выше: F1 = 2/3, balanced accuracy = (2/3 + 6/7)/2. CI — бутстреп строк.
    preds = pd.DataFrame({"model_key": "llm", "shots": 5, "row_idx": range(10), "y_true": Y, "pred": P})
    t = an.typeE_ci_table(preds, 1000, 42).iloc[0]
    assert t["f1_failure"] == pytest.approx(2 / 3)
    assert t["balanced_accuracy"] == pytest.approx((2 / 3 + 6 / 7) / 2)
    assert t["f1_ci_low"] <= 2 / 3 <= t["f1_ci_high"] and t["ba_ci_low"] <= t["balanced_accuracy"] <= t["ba_ci_high"]
    units = list(range(10))
    assert (t["f1_ci_low"], t["f1_ci_high"]) == an.bootstrap_stat(units, lambda s: an.f1_failure(Y[s], P[s]), 1000, 42)
    # Предсказания LogReg по строкам → тот же вид таблицы, метки берутся из ответов LLM.
    lr = an.logreg_rows({5: {i: 0 for i in range(10)}}, preds)
    assert len(lr) == 10 and lr["y_true"].tolist() == Y.tolist() and set(lr["model_key"]) == {"LogisticRegression"}
