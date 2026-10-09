"""Тесты src/grade.py: автопроверка A, B (вердикт), E, F. Сеть не нужна."""

import pytest

from src import bank
from src.grade import close, grade_auto
from src.parse import parse_answer


def task(**over):
    t = {"id": "X1", "type": "A", "answer_kind": "number", "reference": 0.161, "tolerance": 0.01}
    t.update(over)
    return t


def grade(t, text, lang="ru"):
    return grade_auto(t, parse_answer(text, t["answer_kind"], lang))


# ---------- Числа ----------

@pytest.mark.parametrize("text,correct", [
    ("ОТВЕТ: 0.161", True),
    ("ОТВЕТ: 0,161", True),
    ("ОТВЕТ: 0.16", True),             # 0.6% — в допуске ±1%
    ("ОТВЕТ: 16,1%", True),            # проценты ↔ доли
    ("ОТВЕТ: 16 %", True),
    ("**ОТВЕТ:** $\\approx 0{,}161$", True),
    ("ОТВЕТ: 0.15", False),            # 6.8% — вне допуска
    ("ОТВЕТ: 16.1", False),            # без знака % не пересчитываем
    ("ОТВЕТ: 0.95", False),            # типичная ошибка: P(+|D) вместо P(D|+)
])
def test_number(text, correct):
    assert grade(task(), text)["correct"] is correct


def test_percent_reference_in_percent_units():
    # Если эталон записан в процентах (16.1), ответ «16,1%» тоже засчитывается.
    assert grade(task(reference=16.1), "ОТВЕТ: 16,1%")["correct"] is True


def test_tolerance_from_task():
    t = task(reference=0.8994, tolerance=0.03)
    assert grade(t, "ОТВЕТ: 0.88")["correct"] is True
    assert grade(t, "ОТВЕТ: 0.85")["correct"] is False


def test_percent_of_fraction_is_not_scaled():
    # «0,161%» — это 0.00161, а не 0.161.
    assert grade(task(), "ОТВЕТ: 0,161%")["correct"] is False
    t = task(answer_kind="numbers", reference=[3, 1, 0.75])
    assert grade(t, "ОТВЕТ: 3; 1; 0,75%")["correct"] is False


def test_zero_tolerance_kept():
    assert grade(task(reference=3, tolerance=0), "ОТВЕТ: 3.01")["correct"] is False
    assert grade(task(reference=3, tolerance=0), "ОТВЕТ: 3")["correct"] is True


def test_default_tolerance_when_null():
    assert grade(task(tolerance=None), "ОТВЕТ: 0.1605")["correct"] is True


def test_zero_reference_absolute_tolerance():
    assert close(0.005, 0, 0.01) and not close(0.02, 0, 0.01)


def test_numbers_all_must_match():
    t = task(answer_kind="numbers", reference=[0.97, 0.75, 0.6, 0.6667])
    ok = grade(t, "ОТВЕТ: 0,97; 0,75; 0,6; 0,667")
    assert ok["correct"] is True and ok["score"] == 1.0
    bad = grade(t, "ОТВЕТ: 0,97; 0,75; 0,6; 0,7")
    assert bad["correct"] is False and bad["score"] == 0.75
    assert "0.7 vs 0.6667" in bad["reason"]


def test_numbers_count_mismatch():
    t = task(answer_kind="numbers", reference=[0.4, 1])
    r = grade(t, "ОТВЕТ: 0.4")
    assert r["correct"] is False and "чисел в ответе 1, нужно 2" in r["reason"]


def test_numbers_order_matters():
    t = task(answer_kind="numbers", reference=[2.2, 0.6, 5.8])
    assert grade(t, "ОТВЕТ: 0.6; 2.2; 5.8")["correct"] is False


def test_numbers_percent_list():
    t = task(answer_kind="numbers", reference=[0.97, 0.75, 0.6, 0.6667])
    assert grade(t, "ОТВЕТ: 97%; 75%; 60%; 66,67%")["correct"] is True


def test_refusal_on_task_with_answer_is_wrong():
    r = grade(task(), "ОТВЕТ: нельзя определить")
    assert r["correct"] is False and "нельзя определить" in r["reason"]
    assert grade(task(), "Не люблю менять тему разговора, но вот сейчас тот самый случай.")["correct"] is False


def test_no_answer_is_wrong():
    r = grade(task(), "Интересная задача.")
    assert r["correct"] is False and r["reason"] == "ответ не найден"


def test_heuristic_marked_in_reason():
    r = grade(task(), "Итак, вероятность примерно 0,161.")
    assert r["correct"] is True and "эвристикой" in r["reason"]


# ---------- Недостающие данные (F, cannot_determine) ----------

CD = dict(type="F", answer_kind="cannot_determine", reference="cannot_determine", tolerance=None)


@pytest.mark.parametrize("text,correct", [
    ("ОТВЕТ: нельзя определить\nУВЕРЕННОСТЬ: 80", True),
    ("ОТВЕТ: данных недостаточно", True),
    ("Не задан b0, поэтому вычислить вероятность невозможно.", True),    # без метки, но отказ от числа
    ("Я не могу ответить на этот вопрос.", True),                        # отказ = не дал число
    ("ОТВЕТ: 0.5987", False),                                            # «додумал» b0
    ("ОТВЕТ: 0.5; 2.7", False),
    ("Интересная задача.", False),
])
def test_cannot_determine(text, correct):
    assert grade(task(**CD), text)["correct"] is correct


def test_cannot_determine_reason_distinguishes_refusal():
    assert "нельзя определить" in grade(task(**CD), "ОТВЕТ: нельзя определить")["reason"]
    assert "отказ" in grade(task(**CD), "Я не могу ответить на этот вопрос.")["reason"]


def test_cannot_determine_en():
    assert grade(task(**CD), "ANSWER: cannot be determined", "en")["correct"] is True


# ---------- B: только вердикт ----------

B = dict(type="B", answer_kind="bool", tolerance=None)


@pytest.mark.parametrize("ref,text,correct", [
    (False, "ОТВЕТ: неверно", True),
    (False, "ОТВЕТ: верно", False),
    (True, "ОТВЕТ: **Верно**", True),
    (True, "ОТВЕТ: частично верно", False),
    (True, "ОТВЕТ: нельзя однозначно определить", False),
])
def test_bool(ref, text, correct):
    r = grade(task(reference=ref, **B), text)
    assert r["correct"] is correct
    assert "обоснование оценивают люди" in r["reason"]


# ---------- E: метка класса ----------

def test_label():
    t = task(type="E", answer_kind="label", reference=1, tolerance=None)
    assert grade(t, "ОТВЕТ: 1")["correct"] is True
    assert grade(t, "ОТВЕТ: 0 (нет отказа)")["correct"] is False
    assert grade(t, "ОТВЕТ: отказ")["correct"] is True


# ---------- C и D: люди ----------

@pytest.mark.parametrize("over,text", [
    (dict(type="C", answer_kind="text", reference="kNN"), "ОТВЕТ: kNN"),
    (dict(type="D", answer_kind="code", reference=None), "```python\nprint(1)\n```\nОТВЕТ: код выше"),
])
def test_human_graded(over, text):
    r = grade(task(**over), text)
    assert r == {"correct": None, "score": None, "reason": "проверяют люди (слепая проверка)"}


# ---------- Настоящий банк: идеальный ответ засчитывается, сдвиг на 5% — нет ----------

def fmt(x):
    """Эталон так, как его записала бы модель: десятичная запятая."""
    return f"{x:.4f}".rstrip("0").rstrip(".").replace(".", ",")


def numeric_tasks():
    tasks = bank.load_tasks(bank.BANK_PATH) + bank.load_tasks(bank.DEV_PATH)
    return [t for t in tasks if t["answer_kind"] in ("number", "numbers") and t["reference"] is not None]


@pytest.mark.parametrize("t", numeric_tasks(), ids=lambda t: t["id"])
def test_bank_reference_roundtrip(t):
    refs = t["reference"] if isinstance(t["reference"], list) else [t["reference"]]
    good = "Решение...\nОТВЕТ: " + "; ".join(fmt(x) for x in refs) + "\nУВЕРЕННОСТЬ: 90"
    assert grade(t, good)["correct"] is True, good
    shifted = "ОТВЕТ: " + "; ".join(fmt(x * 1.05 if x else 0.5) for x in refs)
    assert grade(t, shifted)["correct"] is False, shifted


@pytest.mark.parametrize("t", [t for t in bank.load_tasks(bank.BANK_PATH) + bank.load_tasks(bank.DEV_PATH)
                               if t["answer_kind"] == "bool" and t["reference"] is not None],
                         ids=lambda t: t["id"])
def test_bank_bool_roundtrip(t):
    word = "верно" if t["reference"] else "неверно"
    assert grade(t, f"Обоснование.\nОТВЕТ: {word}\nУВЕРЕННОСТЬ: 80")["correct"] is True
