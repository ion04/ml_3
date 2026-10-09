"""Тесты src/parse.py на реалистичных ответах моделей. Сеть не нужна.

Ответы ниже написаны «в стиле» GigaChat, YandexGPT, DeepSeek и Qwen: markdown, LaTeX, десятичная
запятая, проценты, повтор «ОТВЕТ» в рассуждении, текст после ответа, <think>, отказы.
"""

import pytest

from src.parse import CANNOT, clean, extract_code, parse_answer, to_float

# ---------- Числа: (ответ модели, ожидаемое значение) ----------

NUMBER_CASES = [
    # Формат соблюдён буквально
    ("Решение по формуле Байеса...\nОТВЕТ: 0.161\nУВЕРЕННОСТЬ: 90", 0.161),
    ("ОТВЕТ: 0,161\nУВЕРЕННОСТЬ: 90", 0.161),
    # GigaChat: жирные метки, десятичная запятая, проценты
    ("**Решение:**\n\nP(D|+) = 0,95·0,01 / 0,059 ≈ 0,161\n\n**ОТВЕТ:** 16,1%\n**УВЕРЕННОСТЬ:** 85%", 0.161),
    ("ОТВЕТ: 16 %", 0.16),
    ("ОТВЕТ: 16,1 процента", 0.161),
    ("ОТВЕТ: ≈0,16", 0.16),
    ("ОТВЕТ: ~0.16", 0.16),
    # YandexGPT: выражение и пояснение в скобках
    ("ОТВЕТ: P(D|+) = 0,0095/0,059 ≈ 0,161 (16,1%)\nУВЕРЕННОСТЬ: 80", 0.161),
    ("ОТВЕТ: 0.161 (округлено до трёх знаков)", 0.161),
    # Дроби и LaTeX (DeepSeek, Qwen)
    ("ОТВЕТ: 4/25", 0.16),
    ("ОТВЕТ: $\\frac{4}{25}$", 0.16),
    ("ОТВЕТ: \\(\\dfrac{4}{25} = 0.16\\)", 0.16),
    ("ОТВЕТ: $\\boxed{0.6931}$\nУВЕРЕННОСТЬ: 95", 0.6931),
    ("ОТВЕТ: $\\alpha = \\tfrac{1}{2}\\ln 4 \\approx 0{,}6931$", 0.6931),
    ("ОТВЕТ: $1.5 \\cdot 10^{-3}$", 0.0015),
    ("ОТВЕТ: 1,5×10⁻³", 0.0015),
    ("ОТВЕТ: 1e-3", 0.001),
    # Разделители тысяч и единицы измерения
    ("ОТВЕТ: 1 551 об/мин", 1551.0),
    ("ОТВЕТ: 1 551", 1551.0),
    ("ОТВЕТ: 2,7333 часа\nУВЕРЕННОСТЬ: 90", 2.7333),
    ("ОТВЕТ: 5.8 ед.", 5.8),
    # Обозначения с цифрами не путаются с ответом
    ("ОТВЕТ: y(6) = 5.8", 5.8),
    ("ОТВЕТ: x_6 → 5,8", 5.8),
    ("ОТВЕТ: F1 = 0.6667", 0.6667),
    ("ОТВЕТ: прогноз при x = 6: 5.8", 5.8),
    # Отрицательные числа: юникодный минус, тире перед цифрой
    ("ОТВЕТ: −1", -1.0),
    ("ОТВЕТ: –0,5", -0.5),
    # Ответ и уверенность в одной строке, текст после ответа
    ("ОТВЕТ: 0.161 УВЕРЕННОСТЬ: 90", 0.161),
    ("ОТВЕТ: 0.161\nУВЕРЕННОСТЬ: 90\n\nЕсли нужно, могу пояснить подробнее!", 0.161),
    # Повторный «ОТВЕТ» внутри рассуждения: берётся последний
    ("Формат: ОТВЕТ: <число>\nСначала ОТВЕТ: 0.5 — нет, ошибка.\nПересчитаем...\nОТВЕТ: 0.6931\nУВЕРЕННОСТЬ: 95",
     0.6931),
    # Метка на отдельной строке, значение ниже (Qwen с display-math)
    ("Итак:\n\n**ОТВЕТ:**\n$$\n0.161\n$$\n**УВЕРЕННОСТЬ:** 90", 0.161),
    # Заголовок markdown
    ("### ОТВЕТ: 0,1212\n### УВЕРЕННОСТЬ: 75", 0.1212),
]


@pytest.mark.parametrize("text,expected", NUMBER_CASES)
def test_number(text, expected):
    r = parse_answer(text, "number", "ru")
    assert r.parse_ok, r.notes
    assert not r.refused
    assert r.value == pytest.approx(expected), (r.value, r.notes)


def test_percent_flag():
    assert parse_answer("ОТВЕТ: 16,1%", "number", "ru").had_percent
    assert not parse_answer("ОТВЕТ: 0,161", "number", "ru").had_percent


# ---------- Английские ответы ----------

EN_CASES = [
    ("Using Bayes' theorem...\nANSWER: 0.161\nCONFIDENCE: 90", 0.161),
    ("**ANSWER:** 16.1%\n**CONFIDENCE:** 80", 0.161),
    ("ANSWER: 1,551 rpm", 1551.0),          # в английском «1,551» — тысячи
    ("ANSWER: 0,161", 0.161),               # нулевая целая часть — всё равно дробь
    ("**Final Answer:** 0.6931", 0.6931),
    ("ANSWER: $\\approx 0.693$\nCONFIDENCE: 95%", 0.693),
]


@pytest.mark.parametrize("text,expected", EN_CASES)
def test_number_en(text, expected):
    r = parse_answer(text, "number", "en")
    assert r.value == pytest.approx(expected), (r.value, r.notes)


def test_ru_comma_with_three_digits_is_decimal():
    assert parse_answer("ОТВЕТ: 1,551", "number", "ru").value == pytest.approx(1.551)


# ---------- Списки чисел ----------

NUMBERS_CASES = [
    ("ОТВЕТ: 0,97; 0,75; 0,6; 0,6667\nУВЕРЕННОСТЬ: 95", [0.97, 0.75, 0.6, 0.6667]),
    ("ОТВЕТ: accuracy = 0.97; precision = 0.75; recall = 0.6; F1 = 0.6667", [0.97, 0.75, 0.6, 0.6667]),
    ("ОТВЕТ: accuracy = 0.97, precision = 0.75, recall = 0.6, F1 = 0.667", [0.97, 0.75, 0.6, 0.667]),
    ("ОТВЕТ: 97%; 75%; 60%; 66,67%", [0.97, 0.75, 0.6, 0.6667]),
    ("ОТВЕТ: 0,4; +1", [0.4, 1.0]),
    ("ОТВЕТ: 0,4; 1", [0.4, 1.0]),
    ("ОТВЕТ: ширина зазора 0.4; класс −1", [0.4, -1.0]),
    ("ОТВЕТ: 2/‖w‖ = 0.4; класс (+1)", [0.4, 1.0]),
    ("ОТВЕТ: $b_0 = 2{,}2$; $b_1 = 0{,}6$; $\\hat y(6) = 5{,}8$", [2.2, 0.6, 5.8]),
    ("ОТВЕТ: [3, 1, 0.75]", [3.0, 1.0, 0.75]),
    ("ОТВЕТ: 3 и 1, доля 0.75", [3.0, 1.0, 0.75]),
    ("ОТВЕТ: 0.5987; 2.7333 (часа)", [0.5987, 2.7333]),
    # Метка на отдельной строке, значения списком ниже (GigaChat)
    ("**ОТВЕТ:**\n- b0 = 2.2\n- b1 = 0.6\n- ŷ(6) = 5.8\n\n**УВЕРЕННОСТЬ:** 90", [2.2, 0.6, 5.8]),
    ("ОТВЕТ:\n1. 3\n2. 1\n3. 0,75\nУВЕРЕННОСТЬ: 90", [3.0, 1.0, 0.75]),
]


@pytest.mark.parametrize("text,expected", NUMBERS_CASES)
def test_numbers(text, expected):
    r = parse_answer(text, "numbers", "ru")
    assert r.parse_ok, r.notes
    assert r.value == pytest.approx(expected), (r.value, r.notes)


# ---------- Верно / неверно ----------

BOOL_CASES = [
    ("Обоснование...\nОТВЕТ: неверно\nУВЕРЕННОСТЬ: 80", False),
    ("ОТВЕТ: Неверно.", False),
    ("ОТВЕТ: не верно", False),
    ("ОТВЕТ: **Верно**", True),
    ("ОТВЕТ: верно (утверждение истинно)", True),
    ("ОТВЕТ: утверждение ложно", False),
    ("ОТВЕТ: Нет, неверно", False),
    ("ОТВЕТ: Да", True),
    ("ОТВЕТ: Утверждение неверно — правильно было бы сказать иначе", False),
    ("ANSWER: false\nCONFIDENCE: 90", False),
    ("ANSWER: **True**", True),
    ("ANSWER: Incorrect", False),
    ("ANSWER: The statement is not true.", False),
]


@pytest.mark.parametrize("text,expected", BOOL_CASES)
def test_bool(text, expected):
    lang = "en" if "ANSWER" in text else "ru"
    r = parse_answer(text, "bool", lang)
    assert r.parse_ok, r.notes
    assert r.value is expected


def test_bool_partial_is_not_a_verdict():
    r = parse_answer("ОТВЕТ: частично верно", "bool", "ru")
    assert r.value is None and not r.parse_ok


def test_bool_reasoning_words_ignored_when_format_followed():
    # В рассуждении есть «верно», но строка ОТВЕТ решает.
    text = "Верно ли, что k растёт — ошибка падает? Нет: при больших k растёт смещение.\nОТВЕТ: неверно\nУВЕРЕННОСТЬ: 90"
    assert parse_answer(text, "bool", "ru").value is False


def test_bool_fallback_first_line():
    r = parse_answer("Неверно.\n\nПри увеличении k растёт смещение, и ошибка на тесте может вырасти.", "bool", "ru")
    assert r.value is False and not r.parse_ok


# ---------- «Нельзя определить» и отказы ----------

CANNOT_CASES = [
    ("ОТВЕТ: нельзя определить\nУВЕРЕННОСТЬ: 90", "ru"),
    ("ОТВЕТ: Нельзя определить", "ru"),
    ("ОТВЕТ: **невозможно однозначно определить**", "ru"),
    ("ОТВЕТ: данных недостаточно", "ru"),
    ("ОТВЕТ: недостаточно данных для расчёта", "ru"),
    ("ОТВЕТ: не хватает данных (не задан b0)", "ru"),
    ("ОТВЕТ: вероятность найти нельзя без b0", "ru"),
    ("ANSWER: cannot be determined\nCONFIDENCE: 85", "en"),
    ("ANSWER: It cannot be uniquely determined", "en"),
    ("ANSWER: insufficient data", "en"),
    ("ANSWER: not enough information", "en"),
]


@pytest.mark.parametrize("text,lang", CANNOT_CASES)
@pytest.mark.parametrize("kind", ["number", "numbers", "cannot_determine"])
def test_cannot_determine(text, lang, kind):
    r = parse_answer(text, kind, lang)
    assert r.value == CANNOT and r.refused and r.parse_ok


def test_cannot_determine_without_label():
    text = "Не задан свободный член b0, поэтому вычислить вероятность невозможно."
    r = parse_answer(text, "number", "ru")
    assert r.value == CANNOT and r.refused and not r.parse_ok


def test_gini_impurity_is_not_cannot_determine():
    text = ("Неопределённость Джини родителя: 1 − 0,8² − 0,2² = 0,32. Неопределённость потомков — взвешенная.\n"
            "ОТВЕТ: 0,1212\nУВЕРЕННОСТЬ: 85")
    r = parse_answer(text, "number", "ru")
    assert r.value == pytest.approx(0.1212) and not r.refused


@pytest.mark.parametrize("text", [
    "Не люблю менять тему разговора, но вот сейчас тот самый случай.",                     # GigaChat
    "Что-то в вашем вопросе меня смущает. Может, поговорим на другую тему?",                 # GigaChat
    "В интернете есть много сайтов с информацией на эту тему. Посмотрите, что нашлось в поиске",  # YandexGPT
    "Я не могу ответить на этот вопрос.",
    "I'm sorry, but I cannot help with that.",
])
def test_refusal(text):
    r = parse_answer(text, "number", "en" if text[0] == "I" else "ru")
    assert r.refused and r.value is None


def test_refusal_in_answer_line():
    r = parse_answer("ОТВЕТ: не могу ответить", "number", "ru")
    assert r.refused and r.value is None and r.parse_ok


def test_long_reasoning_with_phrase_is_not_refusal():
    # «не могу сразу ответить» в середине длинного решения — не отказ.
    text = "Сразу не могу ответить, посчитаем. " + "Шаг. " * 200 + "\nОТВЕТ: 0.161\nУВЕРЕННОСТЬ: 90"
    r = parse_answer(text, "number", "ru")
    assert r.value == pytest.approx(0.161) and not r.refused


# ---------- Запасная эвристика ----------

def test_fallback_boxed():
    r = parse_answer("Следовательно, $$\\boxed{\\frac{4}{25}}$$", "number", "ru")
    assert r.value == pytest.approx(0.16) and not r.parse_ok


def test_fallback_last_line():
    r = parse_answer("Считаем по Байесу.\nИтак, вероятность дефекта примерно 0,161.", "number", "ru")
    assert r.value == pytest.approx(0.161) and not r.parse_ok
    assert any("эвристика" in n for n in r.notes)


def test_fallback_lowercase_inline():
    r = parse_answer("Решение...\nИтак, ответ: 0.6931", "number", "ru")
    assert r.value == pytest.approx(0.6931) and not r.parse_ok


def test_loose_label_accepted():
    r = parse_answer("Решение...\n**Ответ:** 0,6931\n**Уверенность:** 90", "number", "ru")
    assert r.parse_ok and r.value == pytest.approx(0.6931) and r.confidence == 90


def test_placeholder_answer_falls_back():
    # Последняя метка — повтор шаблона без значения: значение берётся эвристикой из строки выше.
    text = "ОТВЕТ: 0.69\nУВЕРЕННОСТЬ: 90\n(формат: ОТВЕТ: <число>)"
    r = parse_answer(text, "number", "ru")
    assert r.value == pytest.approx(0.69) and not r.parse_ok


def test_nothing_found():
    r = parse_answer("Это интересная задача про деревья решений.", "number", "ru")
    assert r.value is None and not r.parse_ok and not r.refused


@pytest.mark.parametrize("text", [None, "", "   \n"])
def test_empty(text):
    r = parse_answer(text, "number", "ru")
    assert r.value is None and not r.parse_ok and r.notes == ["пустой ответ"]


# ---------- <think> (qwen3) ----------

def test_think_block_removed():
    text = "<think>\nМожет, ОТВЕТ: 5? Нет.\n</think>\n\nОТВЕТ: 0.693\nУВЕРЕННОСТЬ: 90"
    r = parse_answer(text, "number", "ru")
    assert r.value == pytest.approx(0.693) and "вырезан блок <think>" in r.notes


def test_closing_think_only():
    r = parse_answer("рассуждение ОТВЕТ: 1\n</think>\nОТВЕТ: 2", "number", "ru")
    assert r.value == 2


# ---------- Уверенность ----------

@pytest.mark.parametrize("text,expected", [
    ("ОТВЕТ: 1\nУВЕРЕННОСТЬ: 90", 90),
    ("ОТВЕТ: 1\nУВЕРЕННОСТЬ: 90%", 90),
    ("ОТВЕТ: 1\n**УВЕРЕННОСТЬ:** 85", 85),
    ("ОТВЕТ: 1\nУВЕРЕННОСТЬ — 70", 70),
    ("ОТВЕТ: 1\nУверенность: 75.5", 75.5),
    ("ANSWER: 1\nCONFIDENCE: 0.9", 90),
    ("ANSWER: 1\nCONFIDENCE: 100", 100),
    ("ОТВЕТ: 1\nУВЕРЕННОСТЬ: 150", None),
    ("ОТВЕТ: 1\nУВЕРЕННОСТЬ: высокая", None),
    ("ОТВЕТ: 1", None),
])
def test_confidence(text, expected):
    assert parse_answer(text, "number", "ru").confidence == expected


def test_confidence_does_not_break_answer():
    r = parse_answer("ОТВЕТ: 0,4; 1\nУВЕРЕННОСТЬ: 95", "numbers", "ru")
    assert r.value == [0.4, 1.0] and r.confidence == 95


# ---------- Тип C (текст) ----------

def test_text_method_name():
    text = "Признаков больше, чем объектов, нужна интерпретируемость...\nОТВЕТ: **Логистическая регрессия с L1-регуляризацией.**\nУВЕРЕННОСТЬ: 75"
    r = parse_answer(text, "text", "ru")
    assert r.value == "Логистическая регрессия с L1-регуляризацией" and r.parse_ok


def test_text_no_label():
    r = parse_answer("Я бы выбрал kNN.", "text", "ru")
    assert r.value is None and not r.parse_ok


# ---------- Тип D (код) ----------

CODE = "from sklearn.datasets import load_iris\nprint(0.97)\n"


def test_code_single_block():
    text = f"Вот решение:\n```python\n{CODE}```\nОТВЕТ: код выше\nУВЕРЕННОСТЬ: 80"
    r = parse_answer(text, "code", "ru")
    assert r.value == CODE and r.parse_ok and r.confidence == 80


def test_code_kept_verbatim_and_answer_inside_code_ignored():
    code = "# ОТВЕТ: это комментарий\nx = 1  # **жирный**\nprint('ANSWER:', x)\n"
    text = f"```python\n{code}```\nОТВЕТ: код выше"
    r = parse_answer(text, "code", "ru")
    assert r.value == code
    assert r.raw_answer_line == "ОТВЕТ: код выше"


def test_code_several_blocks_longest():
    text = f"```python\n{CODE}```\nВывод:\n```\n0.97\n```\nКоротко:\n```python\nprint(1)\n```\nОТВЕТ: код выше"
    r = parse_answer(text, "code", "ru")
    assert r.value == CODE
    assert any("блоков кода 2" in n for n in r.notes)


def test_code_unclosed_block():
    r = parse_answer("```python\nimport numpy as np\nprint(1", "code", "ru")
    assert r.value == "import numpy as np\nprint(1"
    assert any("не закрыт" in n for n in r.notes)


def test_code_without_language():
    code, notes = extract_code("```\nprint(1)\n```")
    assert code == "print(1)\n" and any("не указан язык" in n for n in notes)


def test_code_missing():
    r = parse_answer("Используйте GridSearchCV.\nОТВЕТ: код выше", "code", "ru")
    assert r.value is None and not r.parse_ok


# ---------- Тип E (метка) ----------

@pytest.mark.parametrize("text,expected", [
    ("ОТВЕТ: 1\nУВЕРЕННОСТЬ: 70", 1),
    ("ОТВЕТ: 0 (нет отказа)", 0),
    ("ОТВЕТ: нет отказа", 0),
    ("ОТВЕТ: отказ", 1),
    ("ANSWER: no failure", 0),
    ("ANSWER: **1**", 1),
    ("ОТВЕТ: **Нет отказа.**\nУВЕРЕННОСТЬ: 85", 0),
    ("ОТВЕТ: Отказ (вероятно, из-за перегрева)\nУВЕРЕННОСТЬ: 60", 1),
    ("ОТВЕТ: отказ маловероятен", 0),
    ("ОТВЕТ: отказа не будет", 0),
    ("ОТВЕТ: станок не откажет", 0),
    ("ANSWER: Failure\nCONFIDENCE: 70", 1),
    ("ANSWER: No failure.", 0),
    ("ANSWER: failure is unlikely", 0),
    ("Крутящий момент высокий, износ большой.\nОТВЕТ: отказ\nУВЕРЕННОСТЬ: 75", 1),
])
def test_label(text, expected):
    r = parse_answer(text, "label", "en" if "ANSWER" in text else "ru")
    assert r.value == expected and r.parse_ok


@pytest.mark.parametrize("text,expected", [
    # отрицание после слова «отказ» (случаи из ревью субагентом)
    ("ОТВЕТ: Отказа нет", 0), ("ОТВЕТ: Отказ: нет", 0), ("ОТВЕТ: отказ — нет", 0),
    ("ОТВЕТ: Отказ оборудования не произойдёт", 0), ("ОТВЕТ: Отказа оборудования не будет", 0),
    ("ОТВЕТ: отказ не случится", 0), ("ОТВЕТ: отказ не наблюдается", 0), ("ОТВЕТ: отказ отсутствует", 0),
    ("ОТВЕТ: Отсутствие отказа", 0), ("ОТВЕТ: отказ невозможен", 0), ("ОТВЕТ: не отказ", 0),
    ("ОТВЕТ: низкая вероятность отказа", 0), ("ОТВЕТ: риск отказа низкий", 0), ("ОТВЕТ: вероятность отказа мала", 0),
    ("ОТВЕТ: нормальная работа", 0), ("ОТВЕТ: станок откажет", 1), ("ОТВЕТ: отказ не исключён", 1),
    ("ОТВЕТ: высокая вероятность отказа", 1), ("ОТВЕТ: отказ вероятен", 1),
    ("ANSWER: unlikely to fail", 0), ("ANSWER: Not a failure", 0), ("ANSWER: failure not expected", 0),
    ("ANSWER: Failure is not expected", 0), ("ANSWER: failure is not likely", 0),
    ("ANSWER: The machine is not expected to fail", 0), ("ANSWER: low risk of failure", 0),
    ("ANSWER: failure probability is low", 0), ("ANSWER: failure: no", 0), ("ANSWER: high risk of failure", 1),
])
def test_label_negations(text, expected):
    r = parse_answer(text, "label", "en" if "ANSWER" in text else "ru")
    assert r.value == expected and any("не дословная" in n for n in r.notes)


@pytest.mark.parametrize("text,expected", [("ОТВЕТ: Нет отказа.", 0), ("ОТВЕТ: «отказ»", 1),
                                           ("ANSWER: Failure.", 1), ("ОТВЕТ: Исход: нет отказа.", 0)])
def test_label_exact_has_no_notes(text, expected):
    r = parse_answer(text, "label", "en" if "ANSWER" in text else "ru")
    assert r.value == expected and not any("не дословная" in n for n in r.notes)


def test_label_first_class_wins():
    r = parse_answer("ОТВЕТ: отказ (нет сомнений)", "label", "ru")
    assert r.value == 1 and any("оба класса" in n for n in r.notes)


@pytest.mark.parametrize("text", ["ОТВЕТ: отказ | нет отказа", "ОТВЕТ: отказ или нет отказа",
                                  "ANSWER: failure / no failure"])
def test_label_template_echo_not_parsed(text):
    r = parse_answer(text, "label", "en" if "ANSWER" in text else "ru")
    assert r.value is None and not r.parse_ok


# ---------- Случаи из ревью субагентом ----------

@pytest.mark.parametrize("text,kind,expected", [
    # «нельзя …» внутри обоснования вердикта / метода / метки — это не ответ «нельзя определить»
    ("ОТВЕТ: неверно — нельзя сказать, что ошибка всегда уменьшается", "bool", False),
    ("ОТВЕТ: Неверно: оптимальное k нельзя определить заранее", "bool", False),
    ("ANSWER: False — it cannot be determined in advance which k is best", "bool", False),
    ("ОТВЕТ: kNN, однако без масштабирования вычислить расстояния корректно невозможно", "text",
     "kNN, однако без масштабирования вычислить расстояния корректно невозможно"),
    ("ОТВЕТ: 0 — данных недостаточно, чтобы уверенно предсказать отказ", "label", 0),
    # условие и пояснение рядом со значением
    ("ОТВЕТ: 2,2; 0,6; 5,8 при x = 6", "numbers", [2.2, 0.6, 5.8]),
    ("ОТВЕТ: 0,5987 при h = 3; 2,7333", "numbers", [0.5987, 2.7333]),
    ("ОТВЕТ: при h = 3 вероятность 0,5987; порог h = 2,7333", "numbers", [0.5987, 2.7333]),
    ("ОТВЕТ: 3, так как соседи — точки с y = 2, 3, 4", "number", 3),
    # список под меткой и фраза после него
    ("ОТВЕТ:\n- 0,97\n- 0,75\n- 0,6\n- 0,6667\nВсе значения округлены до 4 знаков.\nУВЕРЕННОСТЬ: 90", "numbers",
     [0.97, 0.75, 0.6, 0.6667]),
    # тройные кавычки внутри строки ответа не считаются блоком кода
    ("ОТВЕТ: ```0.161```\nУВЕРЕННОСТЬ: 90", "number", 0.161),
])
def test_review_cases(text, kind, expected):
    lang = "en" if "ANSWER" in text else "ru"
    r = parse_answer(text, kind, lang)
    assert r.parse_ok and not r.refused, r.notes
    assert r.value == (pytest.approx(expected) if isinstance(expected, (int, float, list)) and kind != "label"
                       else expected)


@pytest.mark.parametrize("text", [
    "ANSWER: It is not possible to determine", "ANSWER: can’t be determined", "ANSWER: undeterminable",
    "ОТВЕТ: Число объектов не может быть определено", "ОТВЕТ: однозначно не определяется",
    "ОТВЕТ: неизвестно (данных не хватает)",
])
def test_more_cannot_phrasings(text):
    r = parse_answer(text, "number", "en" if "ANSWER" in text else "ru")
    assert r.value == CANNOT and r.refused


def test_bool_cannot_when_no_verdict():
    r = parse_answer("ОТВЕТ: нельзя однозначно определить", "bool", "ru")
    assert r.value == CANNOT and r.refused


@pytest.mark.parametrize("text,expected", [
    ("ОТВЕТ: 1\n**Уверенность**: высокая (90%)", 90),
    ("ANSWER: 1\nCONFIDENCE: ~95", 95),
    ("ОТВЕТ: 1\nМоя уверенность в ответе: 80", 80),
])
def test_confidence_with_prefix(text, expected):
    assert parse_answer(text, "number", "ru").confidence == expected


def test_code_in_list_is_dedented():
    text = "1. Код:\n    ```python\n    import numpy as np\n    print(1)\n    ```\nОТВЕТ: код выше"
    r = parse_answer(text, "code", "ru")
    assert r.value == "import numpy as np\nprint(1)\n"
    assert any("отступ" in n for n in r.notes)


def test_fence_inside_string_literal():
    code = 'print("```")\nprint(0.97)\n'
    r = parse_answer(f"```python\n{code}```\nОТВЕТ: код выше", "code", "ru")
    assert r.value == code


# ---------- Вспомогательные функции ----------

@pytest.mark.parametrize("token,lang,expected", [
    ("0,161", "ru", 0.161), ("0,161", "en", 0.161), ("1,551", "ru", 1.551), ("1,551", "en", 1551),
    ("1,234.5", "en", 1234.5), ("1,234,567", "en", 1234567), ("2,5", "en", 2.5), ("1.5e-3", "ru", 0.0015),
    ("+1", "ru", 1.0), ("-0,5", "ru", -0.5), ("1.2.3", "ru", None),
])
def test_to_float(token, lang, expected):
    assert to_float(token, lang) == (pytest.approx(expected) if expected is not None else None)


def test_clean_strips_markup():
    assert clean("**$\\frac{1}{2}$**") == "1/2"
    assert clean("`0{,}5`") == "0,5"
