"""Автопроверка ответов (шаг 8 хода работы): типы A, B (только вердикт), E и F.

grade_auto(task, parsed) -> {"correct": bool | None, "score": float | None, "reason": str}
  correct  True/False; None — автоматически не проверяется (C, D, обоснование B — проверяют люди вслепую)
  score    доля верных чисел (для numbers), иначе 1.0 / 0.0; None, если correct is None
  reason   короткое пояснение по-русски (попадает в таблицу результатов и в выборочную проверку парсера)

Правила:
- Числа сравниваются с относительным допуском task["tolerance"] (по умолчанию 0.01 = ±1%).
  Если эталон равен 0, допуск абсолютный (тот же tolerance).
- Проценты: парсер переводит «16%» в 0.16, поэтому эталон-доля 0.16 совпадает. Если в ответе были
  проценты и эталон больше 1 (значит, он записан в процентах), сравниваем исходное число (16).
  Число без знака % (например, 16 при эталоне 0.16) не засчитывается.
- numbers: верно, только если количество чисел совпало и все числа в допуске.
- cannot_determine: верно, если модель отказалась дать число: «нельзя определить» или отказ отвечать.
  Так в PDF: «доля вопросов без ответа или с ответом „не могу определить“ — в задачах с недостающими
  данными такой ответ правильный». В reason видно, какой из двух случаев.
- Отказ / «нельзя определить» в задаче, где ответ есть, — неверно.
- parse_ok = False (значение найдено эвристикой) не меняет оценку, но отмечается в reason.
"""

from src.parse import CANNOT, ParsedAnswer

DEFAULT_TOLERANCE = 0.01
# Запас на погрешность дробных чисел в компьютере: 0.505 − 0.5 = 0.0050000000000000044 > 0.005,
# хотя это ровно 1%. Без запаса граница допуска засчитывалась бы то так, то нет (101 vs 100 — да, 0.505 vs 0.5 — нет).
FLOAT_SLACK = 1e-9


def close(answer: float, reference: float, tol: float) -> bool:
    """Совпадение с относительным допуском (абсолютным при эталоне 0). Граница входит в допуск."""
    if reference == 0:
        return abs(answer) <= tol + FLOAT_SLACK
    return abs(answer - reference) <= tol * abs(reference) * (1 + FLOAT_SLACK)


def number_matches(answer: float, reference: float, tol: float, had_percent: bool) -> bool:
    """Сравнение одного числа. Парсер уже перевёл «16%» в 0.16. Если эталон > 1, он не доля,
    а проценты: тогда сравниваем исходное число (16). При эталоне ≤ 1 ×100 не пробуем,
    иначе «0,161%» засчитался бы за 0.161."""
    if close(answer, reference, tol):
        return True
    return had_percent and abs(reference) > 1 and close(answer * 100, reference, tol)


def _result(correct, reason, score=None):
    if correct is None:
        return {"correct": None, "score": None, "reason": reason}
    if score is None:
        score = 1.0 if correct else 0.0
    return {"correct": correct, "score": score, "reason": reason}


def grade_numbers(task: dict, parsed: ParsedAnswer) -> dict:
    ref = task["reference"]
    tol = task.get("tolerance")
    if tol is None:
        tol = DEFAULT_TOLERANCE

    if ref == CANNOT:
        if parsed.value == CANNOT:
            return _result(True, "верно: «нельзя определить»")
        if parsed.refused:
            return _result(True, "верно: отказ отвечать (число не дано)")
        if parsed.value is None:
            return _result(False, "ответ не найден")
        return _result(False, f"неверно: дан ответ {parsed.value}, а данных недостаточно")

    if parsed.refused:
        what = "«нельзя определить»" if parsed.value == CANNOT else "отказ отвечать"
        return _result(False, f"неверно: {what}, а ответ есть")
    if parsed.value is None:
        return _result(False, "ответ не найден")

    refs = ref if isinstance(ref, list) else [ref]
    answers = parsed.value if isinstance(parsed.value, list) else [parsed.value]
    if len(answers) != len(refs):
        return _result(False, f"неверно: чисел в ответе {len(answers)}, нужно {len(refs)}", score=0.0)
    ok = [number_matches(a, r, tol, parsed.had_percent) for a, r in zip(answers, refs)]
    score = sum(ok) / len(ok)
    if all(ok):
        return _result(True, f"верно в пределах ±{tol:g} (отн.)", score)
    bad = [f"{a:g} vs {r:g}" for a, r, good in zip(answers, refs, ok) if not good]
    return _result(False, "неверно: " + ", ".join(bad), score)


def grade_exact(task: dict, parsed: ParsedAnswer, what: str) -> dict:
    """bool (вердикт B) и label (тип E): точное совпадение."""
    if parsed.refused:
        return _result(False, "неверно: отказ / «нельзя определить»")
    if parsed.value is None:
        return _result(False, f"{what} не найден")
    ref = task["reference"]
    if what == "класс":
        ref = int(ref)
    if parsed.value == ref:
        return _result(True, f"верно: {what} совпал")
    return _result(False, f"неверно: {what} {parsed.value}, эталон {ref}")


def grade_auto(task: dict, parsed: ParsedAnswer) -> dict:
    kind = task["answer_kind"]
    if task["type"] in ("C", "D") or kind in ("text", "code"):
        return _result(None, "проверяют люди (слепая проверка)")
    if kind in ("number", "numbers", "cannot_determine"):
        res = grade_numbers(task, parsed)
    elif kind == "bool":
        res = grade_exact(task, parsed, "вердикт")
        if task["type"] == "B":
            res["reason"] += "; обоснование оценивают люди"
    elif kind == "label":
        res = grade_exact(task, parsed, "класс")
    else:
        raise ValueError(f"{task.get('id')}: неизвестный answer_kind {kind!r}")
    found = parsed.value is not None or parsed.refused
    if not parsed.parse_ok and found and res["correct"] is not None:
        res["reason"] += " (формат не соблюдён, ответ найден эвристикой)"
    return res
