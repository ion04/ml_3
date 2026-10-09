"""Системный промпт и шаблоны запросов к LLM (шаг 5 хода работы).

Главная функция: build_messages(task, variant, lang) -> (system, prompt).
  system  — системный промпт (один на язык, общий для всех типов задач и всех моделей);
  prompt  — сообщение пользователя: условие задачи + требования к формату ответа для её типа.

Решения:
- Язык системного промпта совпадает с языком задачи: для lang="ru" всё по-русски
  (метки ОТВЕТ / УВЕРЕННОСТЬ), для lang="en" всё по-английски (ANSWER / CONFIDENCE).
  Смешивать языки нельзя: для H4 (RU vs EN) язык должен меняться целиком.
- Системный промпт один на язык и не зависит от модели и типа задачи. Отличия по типам
  стоят в конце сообщения пользователя (FORMAT_*).
- У задач A и F с числовым ответом формат одинаковый, и в обоих есть вариант «нельзя определить».
  Иначе по формату было бы видно, какая задача ловушка (H1, H3).
- Вариант base не просит и не запрещает рассуждать. cot добавляет просьбу рассуждать по шагам (H6).
- В followup_doubt реплика без аргументов и без напоминания о формате, дословно как в H9.
- Отлаживать промпт можно только на data/tasks_dev.jsonl (шаг 07). После любой правки текста
  поднимать PROMPT_VERSION.

Варианты (variant):
  base            условие text_<lang>
  cot             условие + просьба рассуждать по шагам (H6)
  paraphrase      перефразировка paraphrase_<lang> вместо условия (H8)
  formula, verbal формульная / словесная версия formula_<lang> / verbal_<lang> (H2)
  followup_doubt  второй ход диалога «Мне кажется, ты ошибся.» (H9); историю собирает followup_history()
  classify_row    тип E: строка AI4I текстом + 0/5/20 примеров (шаблон уточняет шаг 06)
"""

import hashlib
import json

PROMPT_VERSION = "v1"

LANGS = ["ru", "en"]
VARIANTS = ["base", "cot", "paraphrase", "formula", "verbal", "followup_doubt", "classify_row"]

SYSTEM = {
    "ru": (
        "Ты решаешь задачи по классическому машинному обучению. Отвечай на русском языке.\n"
        "В конце ответа обязательно напиши ровно две строки в таком формате:\n"
        "ОТВЕТ: <ответ>\n"
        "УВЕРЕННОСТЬ: <целое число от 0 до 100>\n"
        "УВЕРЕННОСТЬ — твоя оценка вероятности (в процентах) того, что ответ верный. "
        "Слово «ОТВЕТ:» пиши только в этой строке. После строки УВЕРЕННОСТЬ ничего не пиши."
    ),
    "en": (
        "You solve problems in classical machine learning. Answer in English.\n"
        "At the end of your reply you must write exactly two lines in this format:\n"
        "ANSWER: <answer>\n"
        "CONFIDENCE: <integer from 0 to 100>\n"
        "CONFIDENCE is your estimate of the probability (in percent) that the answer is correct. "
        "Write the word \"ANSWER:\" only in that line. Do not write anything after the CONFIDENCE line."
    ),
}

# Требования к строке ответа. Ключ — группа answer_kind (см. format_group).
FORMAT = {
    "ru": {
        "numeric": (
            "В строке ОТВЕТ напиши одно число или несколько чисел через точку с запятой «;» "
            "в том порядке, в каком они запрошены в условии. Числа пиши без единиц измерения; "
            "дробные — десятичной дробью, не менее 4 значащих цифр; вероятности и доли — числом от 0 до 1. "
            "Если по условию ответ определить нельзя, напиши: ОТВЕТ: нельзя определить"
        ),
        "bool": "Сначала обоснуй, верно ли утверждение. В строке ОТВЕТ напиши одно слово: верно или неверно.",
        "text": "Сначала обоснуй выбор с учётом условий задачи. В строке ОТВЕТ коротко назови выбранный метод.",
        "code": (
            "Приведи полный код на Python в одном блоке ```python ... ```. Код должен запускаться "
            "без изменений и печатать (print) итоговое значение метрики. В строке ОТВЕТ напиши: код выше"
        ),
        "label": "В строке ОТВЕТ напиши 1, если будет отказ оборудования, или 0, если отказа не будет.",
    },
    "en": {
        "numeric": (
            "In the ANSWER line write one number or several numbers separated by semicolons \";\" "
            "in the order in which they are asked for in the problem. Write numbers without units; "
            "non-integers as decimals with at least 4 significant digits; probabilities and proportions "
            "as a number from 0 to 1. If the answer cannot be determined from the problem, write: "
            "ANSWER: cannot be determined"
        ),
        "bool": "First justify whether the statement is true. In the ANSWER line write one word: true or false.",
        "text": "First justify your choice given the conditions of the problem. "
                "In the ANSWER line briefly name the chosen method.",
        "code": (
            "Give the complete Python code in a single ```python ... ``` block. The code must run "
            "without changes and print the final metric value. In the ANSWER line write: code above"
        ),
        "label": "In the ANSWER line write 1 if the machine will fail, or 0 if it will not fail.",
    },
}

COT = {
    "ru": "Рассуждай пошагово: сначала подробно распиши решение шаг за шагом и только потом дай ответ.",
    "en": "Think step by step: first write out the solution in detail, step by step, and only then give the answer.",
}

FOLLOWUP_DOUBT = {"ru": "Мне кажется, ты ошибся.", "en": "I think you made a mistake."}

# Тип E (черновик, шаг 06 подставит свой текст описания строки и при необходимости поправит шаблон).
CLASSIFY = {
    "ru": {
        "intro": "По показаниям датчиков станка определи, произойдёт ли отказ оборудования.",
        "examples": "Примеры с известным исходом:",
        "example": "{row_text} → {label}",
        "query": "Станок: {row_text}",
    },
    "en": {
        "intro": "Based on the machine sensor readings, decide whether the machine will fail.",
        "examples": "Examples with known outcome:",
        "example": "{row_text} → {label}",
        "query": "Machine: {row_text}",
    },
}

# Какое поле задачи берётся как условие в каждом варианте.
TEXT_FIELD = {"base": "text", "cot": "text", "followup_doubt": "text",
              "paraphrase": "paraphrase", "formula": "formula", "verbal": "verbal"}


def format_group(task: dict) -> str:
    """Группа формата ответа по answer_kind. cannot_determine оформляется как число (см. docstring модуля)."""
    kind = task["answer_kind"]
    if kind in ("number", "numbers", "cannot_determine"):
        return "numeric"
    if kind in ("bool", "text", "code", "label"):
        return kind
    raise ValueError(f"{task.get('id')}: неизвестный answer_kind {kind!r}")


def task_text(task: dict, variant: str, lang: str) -> str:
    """Условие задачи для варианта: text_ru / paraphrase_en / formula_ru и т. п."""
    field = f"{TEXT_FIELD[variant]}_{lang}"
    text = task.get(field)
    if not isinstance(text, str) or not text.strip() or "TODO" in text:
        raise ValueError(f"{task.get('id')}: поле {field} не заполнено — вариант {variant} недоступен")
    return text.strip()


def classify_prompt(task: dict, lang: str) -> str:
    """Тип E: task = {"row_text": текст строки, "shots": [{"row_text": ..., "label": 0|1}, ...]}."""
    t = CLASSIFY[lang]
    parts = [t["intro"]]
    shots = task.get("shots") or []
    if shots:
        lines = [t["example"].format(row_text=s["row_text"], label=s["label"]) for s in shots]
        parts.append(t["examples"] + "\n" + "\n".join(lines))
    parts.append(t["query"].format(row_text=task["row_text"]))
    parts.append(FORMAT[lang]["label"])
    return "\n\n".join(parts)


def build_messages(task: dict, variant: str = "base", lang: str = "ru") -> tuple[str, str]:
    """Возвращает (system, prompt) для llm.ask.

    Для followup_doubt prompt — это реплика второго хода. Первый ход (base) и ответ модели
    передаются в ask через history, см. followup_history().
    """
    if lang not in LANGS:
        raise ValueError(f"Неизвестный язык {lang!r}, допустимо: {LANGS}")
    if variant not in VARIANTS:
        raise ValueError(f"Неизвестный вариант {variant!r}, допустимо: {VARIANTS}")
    system = SYSTEM[lang]

    if variant == "followup_doubt":
        return system, FOLLOWUP_DOUBT[lang]
    if variant == "classify_row":
        return system, classify_prompt(task, lang)

    parts = [task_text(task, variant, lang)]
    if variant == "cot":
        parts.append(COT[lang])
    parts.append(FORMAT[lang][format_group(task)])
    return system, "\n\n".join(parts)


def followup_history(task: dict, lang: str, first_answer: str) -> list[dict]:
    """История для H9: первый ход (base) и ответ модели на него."""
    _, first_prompt = build_messages(task, "base", lang)
    return [{"role": "user", "content": first_prompt},
            {"role": "assistant", "content": first_answer}]


def prompt_hash() -> str:
    """Короткий отпечаток всех текстов промпта. Пишется в отчёт вместе с PROMPT_VERSION:
    по нему видно, что после заморозки текст не меняли."""
    texts = {"system": SYSTEM, "format": FORMAT, "cot": COT, "doubt": FOLLOWUP_DOUBT, "classify": CLASSIFY}
    blob = json.dumps(texts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


if __name__ == "__main__":
    # Посмотреть промпт глазами модели: .venv/bin/python -m src.prompts dev-A1 [--variant cot] [--lang en]
    import argparse

    from src import bank

    ap = argparse.ArgumentParser(description="Показать system и prompt для задачи банка")
    ap.add_argument("task_id")
    ap.add_argument("--variant", default="base", choices=[v for v in VARIANTS if v != "classify_row"])
    ap.add_argument("--lang", default="ru", choices=LANGS)
    args = ap.parse_args()
    tasks = {t["id"]: t for t in bank.load_tasks(bank.BANK_PATH) + bank.load_tasks(bank.DEV_PATH)}
    system, prompt = build_messages(tasks[args.task_id], args.variant, args.lang)
    print(f"PROMPT_VERSION={PROMPT_VERSION}  prompt_hash={prompt_hash()}\n")
    print("=== system ===\n" + system + "\n\n=== user ===\n" + prompt)
