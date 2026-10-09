"""Разбор ответа модели: строка ОТВЕТ / УВЕРЕННОСТЬ → значение нужного вида.

parse_answer(text, answer_kind, lang) -> ParsedAnswer(value, raw_answer_line, confidence, refused, parse_ok, notes)

Что лежит в value в зависимости от answer_kind:
  number            float
  numbers           list[float] в порядке из ответа
  bool              True (верно) / False (неверно)
  label             1 (отказ) / 0 (нет отказа), тип E
  text              строка: название метода (тип C)
  code              код из блока ```python (тип D), без правок
  cannot_determine  разбирается как number/numbers (модель не знает, что данных не хватает)
Если модель ответила «нельзя определить», то value = CANNOT и refused = True.
Если модель отказалась отвечать вообще, то value = None и refused = True.

Порядок разбора:
  1. Вырезаем <think>…</think> (qwen3 иногда выдаёт рассуждения в тексте). Сырой ответ в журнале не трогаем.
  2. Ищем ПОСЛЕДНЕЕ вхождение «ОТВЕТ:» / «ANSWER:» в верхнем регистре (как просит промпт).
     Если его нет, ищем «Ответ:», «Итоговый ответ:», «Final answer:» в начале строки.
     Метки внутри блоков кода не считаются.
  3. Берём текст после метки до конца строки (или следующие строки, если на этой пусто).
  4. Числовые задачи: сначала проверяем «нельзя определить» / отказ, потом разбираем числа.
     B, C, E: сначала вердикт / метод / метку, и только если их нет — «нельзя определить» / отказ
     (иначе «неверно — нельзя сказать, что …» потеряло бы вердикт).
  5. Если метки нет, работает запасная эвристика (\\boxed{…}, последние строки ответа).
     В этом случае parse_ok = False: такие ответы смотрит человек при выборочной проверке (G2).
"""

import re
import textwrap
from dataclasses import dataclass, field

CANNOT = "cannot_determine"


@dataclass
class ParsedAnswer:
    value: object = None
    raw_answer_line: str | None = None   # строка с меткой ОТВЕТ как есть (или кусок текста для эвристики)
    confidence: float | None = None      # 0..100
    refused: bool = False                # «нельзя определить» или отказ отвечать
    parse_ok: bool = False               # формат соблюдён и значение разобрано
    notes: list = field(default_factory=list)
    had_percent: bool = False            # в ответе были проценты (переведены в доли), см. grade.py


# ---------- Регулярные выражения ----------

# Строгая метка: заглавными буквами, затем (возможно после ** или _) двоеточие.
ANSWER_STRICT = re.compile(r"(?<!\w)(?:ОТВЕТ|ANSWER)(?!\w)[\s*_]*[:：]")
# Мягкая метка: в начале строки, любой регистр, допускаются «итоговый/окончательный/final».
ANSWER_LOOSE = re.compile(
    r"^[ \t>#*_\-•|]*(?:(?:итоговый|окончательный|финальный|final)\s+)?(?:ответ|answer)(?!\w)[\s*_]*[:：]",
    re.I | re.M)
CONF_LABEL = re.compile(r"(?<!\w)(?:УВЕРЕННОСТЬ|CONFIDENCE)(?!\w)", re.I)
# Между меткой и числом — до 25 символов без цифр: «: 90», «**: 85», «: высокая (90%)», «: ~95».
CONF_RE = re.compile(r"(?<!\w)(?:УВЕРЕННОСТЬ|CONFIDENCE)(?!\w)[^\d\n]{0,25}?(\d+(?:[.,]\d+)?)", re.I)

# Блок кода: ```python … ``` (язык может быть не указан). Ограды ``` — только в начале строки.
CODE_RE = re.compile(r"^[ \t]*```[ \t]*([\w+-]*)[^\n]*\n(.*?)^[ \t]*```", re.S | re.M)
CODE_UNCLOSED = re.compile(r"^[ \t]*```[ \t]*([\w+-]*)[^\n]*\n(.*)$", re.S | re.M)

NUM = r"[-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?"
# Число, возможно дробь a/b, возможно со знаком процента.
TOKEN_RE = re.compile(rf"({NUM})(?:\s*/\s*({NUM}))?(\s*(?:%|процент\w*|percent|pct)(?!\w))?", re.I)
# Разделители «что после чего»: берём первое число после последнего из них (x = 6 → 5.8).
ARROWS = ["=", "≈", "→", "=>", ":"]
# Условие рядом со значением («5,8 при x = 6») и пояснение после него («3, так как …») — не ответ.
CONDITION_RE = re.compile(rf"(?<!\w)(?:при|если|for|at|when)\s+[\w()]+\s*=\s*{NUM}", re.I)
BECAUSE_RE = re.compile(r"[\s,;—-]+(?:так\s+как|потому\s+что|поскольку|because|since)(?!\w).*$", re.I | re.S)

# «Нельзя определить» (правильный ответ на задачи с недостающими данными).
_VERB_RU = r"(?:определить|вычислить|рассчитать|посчитать|найти|сказать|ответить|установить|получить)"
CANNOT_RE = re.compile(
    rf"(?:нельзя|невозможно)\s+(?:[\w-]+\s+){{0,2}}?{_VERB_RU}"
    rf"|{_VERB_RU}\s+(?:[\w-]+\s+){{0,3}}?(?:нельзя|невозможно)"
    # «не могу ответить» — это отказ (REFUSAL_RE), а «не могу определить» — ответ «нельзя определить».
    r"|не\s+(?:могу|можем|удаётся|удается|получится)\s+(?:[\w-]+\s+)?(?:определить|вычислить|рассчитать|посчитать|найти)"
    r"|недостаточно\s+(?:\w+\s+)?(?:данных|информации|сведений)"
    r"|(?:данных|информации|сведений)\s+(?:\w+\s+)?недостаточно"
    r"|не\s+хватает\s+(?:\w+\s+)?(?:данных|информации|сведений)"
    r"|не\s+определ(?:ено|ена|ён|ен|ены|яется|яются)(?!\w)"
    r"|не\s+(?:может|могут)\s+быть\s+(?:\w+\s+)?(?:определ|вычисл|найд|рассчита|посчита)\w*"
    r"|(?<!\w)неизвестн(?:о|ы)(?!\w)"
    r"|(?:cannot|can['’]t|can\s+not)\s+be\s+(?:\w+\s+)?(?:determined|calculated|computed|found|answered)"
    r"|(?:cannot|can['’]t|can\s+not|unable\s+to)\s+(?:\w+\s+)?(?:determine|calculate|compute)"
    r"|(?:insufficient|not\s+enough)\s+(?:\w+\s+)?(?:data|information|info)"
    r"|(?:impossible|not\s+possible)\s+to\s+(?:determine|calculate|compute|tell)"
    r"|(?<!\w)(?:undetermined|undeterminable|indeterminate|not\s+determinable)(?!\w)",
    re.I)

# Отказ отвечать вообще. Стоп-фразы GigaChat и YandexGPT взяты из их типичных ответов на «чувствительные» темы.
REFUSAL_RE = re.compile(
    r"не\s+(?:могу|буду|стану)\s+(?:\w+\s+)?(?:отвечать|ответить|помочь|выполнить|решить|обсуждать)"
    r"|как\s+(?:языковая\s+модель|ии|искусственный\s+интеллект)"
    r"|не\s+люблю\s+менять\s+тему"
    r"|поговорим\s+на\s+другую\s+тему"
    r"|в\s+вашем\s+вопросе\s+меня\s+смущает"
    r"|что\s+нашлось\s+в\s+поиске"
    r"|i\s+(?:cannot|can't|can\s+not|am\s+unable\s+to|'m\s+unable\s+to|won't)\s+(?:help|answer|assist|provide|comply|solve)"
    r"|as\s+an\s+ai(?!\w)",
    re.I)

# Вердикты B. «неверно» проверяется раньше «верно».
FALSE_RE = re.compile(
    r"(?<!\w)(?:неверн\w*|не\s+верн\w*|ложн\w*|ложь|неправильн\w*|не\s+правильн\w*|нет"
    r"|false|incorrect|wrong|not\s+true|not\s+correct|no)(?!\w)", re.I)
TRUE_RE = re.compile(r"(?<!\w)(?:верн\w*|правильн\w*|истинн\w*|истина|правда|да|true|correct|right|yes)(?!\w)", re.I)
PARTIAL_RE = re.compile(r"(?<!\w)(?:частично|отчасти|partially|partly)(?!\w)", re.I)

# Метки типа E. «нет отказа» проверяется раньше «отказ».
NO_FAIL_RE = re.compile(
    r"(?<!\w)(?:нет\s+отказа|без\s+отказа|отказа\s+не\s+будет|не\s+будет\s+отказа|отказ\s+не\s+произойд\w*"
    r"|no\s+failure|will\s+not\s+fail|won't\s+fail|not\s+fail\w*|нет|no|норма|normal)(?!\w)", re.I)
FAIL_RE = re.compile(r"(?<!\w)(?:отказ\w*|failure|fail\w*|да|yes)(?!\w)", re.I)

SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")


# ---------- Вспомогательные функции ----------

def strip_think(text: str) -> tuple[str, bool]:
    """Убирает рассуждения <think>…</think>. Если закрывающий тег есть, берём всё после последнего."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1], True
    if "<think>" in text:
        return text.replace("<think>", ""), True
    return text, False


def extract_code(text: str) -> tuple[str | None, list[str]]:
    """Код для типа D: блок ```python (или блок без языка). Код возвращается без правок.
    Если блоков несколько, берётся самый длинный (обычно это полное решение, а не кусок или вывод)."""
    notes = []
    blocks = CODE_RE.findall(text)
    rest = CODE_RE.sub("", text)
    m = CODE_UNCLOSED.search(rest)
    if m:
        blocks.append(m.groups())
        notes.append("блок кода не закрыт (ответ обрезан?)")
    py = [body for lang, body in blocks if lang.lower() in ("python", "py", "python3")]
    if not py:
        py = [body for lang, body in blocks if lang == ""]
        if py:
            notes.append("у блока кода не указан язык python")
    if not py:
        return None, notes + ["блок кода не найден"]
    if len(py) > 1:
        notes.append(f"блоков кода {len(py)}, взят самый длинный")
    code = max(py, key=len)
    # Блок внутри пункта списка markdown сдвинут целиком; общий отступ — это разметка, а не код.
    dedented = textwrap.dedent(code)
    if dedented != code:
        notes.append("у кода убран общий отступ (блок был вложен в список)")
    return dedented, notes


def remove_code(text: str) -> str:
    """Текст без блоков кода: метки и числа внутри кода не должны попадать в разбор."""
    text = CODE_RE.sub("\n", text)
    return CODE_UNCLOSED.sub("\n", text)


def clean(s: str) -> str:
    """Убирает markdown и LaTeX, приводит минусы, дроби и степени десяти к простому виду."""
    s = s.replace("\u2212", "-").replace("\u00a0", " ").replace("\u202f", " ").replace("\u2009", " ")
    s = re.sub(r"(^|[\s=:(;≈])[–—](?=\d)", r"\1-", s)                     # «класс –1» → -1
    s = re.sub(r"10([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: "10^" + m.group(1).translate(SUPERSCRIPT), s)
    s = re.sub(rf"\\[dt]?frac\s*\{{\s*({NUM})\s*\}}\s*\{{\s*({NUM})\s*\}}", r"\1/\2", s)
    s = re.sub(r"(\d)\s*(?:\\times|\\cdot|×|·|\*)\s*10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?", r"\1e\2", s)
    s = s.replace("{,}", ",").replace("\\%", "%")
    s = re.sub(r"\\(?:approx|sim|simeq)(?![a-zA-Z])|~", "≈", s)
    s = re.sub(r"\\(?:to|rightarrow|Rightarrow)(?![a-zA-Z])", "→", s)
    # Обозначения с цифрами (F1, b0, x_6, b_{1}) — это не числа ответа: оставляем только буквы.
    s = re.sub(r"(?<![\w.,])([A-Za-zА-Яа-яЁё]+)_?\{?\d+\}?(?![\d.,]*\d)", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)                                     # остальные команды LaTeX
    s = re.sub(r"[*_`${}]|\\[()\[\],;:! ]", " ", s)                       # markdown и разделители LaTeX
    s = re.sub(r"(?<![\d.,])(\d{1,3})((?: \d{3})+)(?!\d)",                # «1 551» → 1551
               lambda m: m.group(1) + m.group(2).replace(" ", ""), s)
    return re.sub(r"[ \t]+", " ", s).strip()


def to_float(token: str, lang: str) -> float | None:
    """'0,161' → 0.161; '1,551' → 1551 (en) или 1.551 (ru); '1,234.5' → 1234.5; '1.5e-3' → 0.0015."""
    t = token.replace("+", "")
    mant, exp = re.match(r"^(.*?)([eE][-+]?\d+)?$", t).groups()
    exp = exp or ""
    if "," in mant and "." in mant:          # 1,234.5 — запятая разделяет тысячи
        mant = mant.replace(",", "")
    elif mant.count(",") > 1:                # 1,234,567
        mant = mant.replace(",", "")
    elif "," in mant:
        whole, frac = mant.split(",")
        # Ровно 3 цифры после запятой у ненулевой целой части: в английском это тысячи, в русском — дробь.
        if len(frac) == 3 and whole.lstrip("-") not in ("", "0") and lang == "en":
            mant = whole + frac
        else:
            mant = whole + "." + frac
    if mant.count(".") > 1:
        return None
    try:
        return float(mant + exp)
    except ValueError:
        return None


def numbers_in(s: str, lang: str) -> list[tuple[float, bool, int]]:
    """Все числа строки: (значение, был ли знак процента, позиция). Проценты переводятся в доли."""
    out = []
    for m in TOKEN_RE.finditer(s):
        a = to_float(m.group(1), lang)
        if a is None:
            continue
        if m.group(2):
            b = to_float(m.group(2), lang)
            if not b:
                continue
            a = a / b
        pct = bool(m.group(3))
        out.append((a / 100 if pct else a, pct, m.start()))
    return out


def drop_parens(s: str) -> str:
    """Убирает пояснения в круглых скобках: '0.161 (16.1%)' → '0.161', 'y(6) = 5.8' → 'y = 5.8'."""
    prev = None
    while prev != s:
        prev, s = s, re.sub(r"\([^()]*\)", " ", s)
    return s


def pick_number(s: str, lang: str):
    """Одно число из куска ответа: первое после последнего '=', '≈', '→' или ':', иначе первое в строке."""
    s = CONDITION_RE.sub(" ", BECAUSE_RE.sub("", s))
    for src in (drop_parens(s), s):
        toks = numbers_in(src, lang)
        if toks:
            break
    else:
        return None
    pos = max(src.rfind(a) for a in ARROWS)
    after = [t for t in toks if t[2] > pos]
    return (after or toks)[0]


def pick_numbers(s: str, lang: str) -> tuple[list | None, bool, list[str]]:
    """Список чисел: части через ';' (или ', ' / ' и '), из каждой — одно число."""
    notes = []
    if ";" in s:
        parts = s.split(";")
    else:
        parts = re.split(r",\s+|\s+и\s+|\s+and\s+|\n", s)
        if len(parts) > 1:
            notes.append("числа разделены не «;»")
    parts = [re.sub(r"^\s*(?:[-•]|\d+[.)])\s+", "", p) for p in parts]   # маркеры списка «- », «1. »
    picked = [pick_number(p, lang) for p in parts]
    picked = [p for p in picked if p is not None]
    if len(parts) == 1:
        # Разделителей нет вовсе ('b = 2.2 b = 0.6'): берём все числа.
        picked = numbers_in(drop_parens(s), lang) or numbers_in(s, lang)
    if not picked:
        return None, False, notes
    return [p[0] for p in picked], any(p[1] for p in picked), notes


def parse_bool(s: str) -> tuple[bool | None, list[str]]:
    if PARTIAL_RE.search(s):
        return None, ["частичный вердикт («частично верно»)"]
    f = FALSE_RE.search(s)
    t = TRUE_RE.search(FALSE_RE.sub(lambda m: " " * len(m.group()), s))   # «не верно» не считаем за «верно»
    if f and t:
        # Оба вердикта («верно, а не неверно…»): берём тот, что стоит раньше.
        return t.start() < f.start(), ["в строке ответа оба вердикта, взят первый"]
    if f:
        return False, []
    if t:
        return True, []
    return None, []


def parse_label(s: str) -> int | None:
    digits = set(re.findall(r"(?<![\d.,])[01](?![\d.,]*\d)", s))
    if len(digits) == 1:
        return int(digits.pop())
    if NO_FAIL_RE.search(s):
        return 0
    if FAIL_RE.search(s):
        return 1
    return None


def parse_confidence(body: str) -> tuple[float | None, list[str]]:
    matches = list(CONF_RE.finditer(body))
    if not matches:
        note = "УВЕРЕННОСТЬ без числа" if CONF_LABEL.search(body) else "нет строки УВЕРЕННОСТЬ"
        return None, [note]
    raw = matches[-1].group(1)
    x = float(raw.replace(",", "."))
    notes = []
    if ("." in raw or "," in raw) and x <= 1:
        x *= 100
        notes.append("уверенность дана долей, умножена на 100")
    if x > 100:
        return None, notes + [f"уверенность {raw} > 100"]
    return x, notes


def find_answer(body: str, kind: str) -> tuple[str | None, str | None, list[str]]:
    """Последняя строка с меткой ответа → (содержимое после метки, сама строка, заметки)."""
    notes = []
    matches = list(ANSWER_STRICT.finditer(body))
    if not matches:
        matches = list(ANSWER_LOOSE.finditer(body))
        if matches:
            notes.append("метка ответа не в формате «ОТВЕТ:»")
    if not matches:
        return None, None, notes
    m = matches[-1]
    line_start = body.rfind("\n", 0, m.start()) + 1
    line_end = body.find("\n", m.end())
    line_end = len(body) if line_end == -1 else line_end
    raw_line = body[line_start:line_end].strip()
    content = body[m.end():line_end]
    c = CONF_LABEL.search(content)
    if c:                                        # «ОТВЕТ: 0.16 УВЕРЕННОСТЬ: 90» в одной строке
        content = content[:c.start()]
    if not clean(content):
        # Метка на отдельной строке, значение ниже: берём следующие строки до УВЕРЕННОСТЬ.
        tail = body[line_end:]
        c = CONF_LABEL.search(tail)
        tail = tail[:c.start()] if c else tail
        lines = [ln for ln in tail.split("\n") if clean(ln)]
        if kind == "numbers":
            # Берём подряд идущие строки-значения («- 0,97», «b0 = 2.2»): маркер списка или цифра
            # в первых 15 символах. Фраза вроде «Все значения округлены до 4 знаков.» список обрывает.
            n = 0
            while n < len(lines) and n < 10 and re.match(r"^\s*(?:[-•*]|\d+[.)]\s)|^.{0,15}\d", lines[n]):
                n += 1
            lines = lines[:max(n, 1)]
        if lines:
            notes.append("значение ответа на следующей строке после метки")
            content = "\n".join(lines[:10]) if kind == "numbers" else lines[0]
            raw_line = raw_line + "\n" + content.strip()
    return content, raw_line, notes


def tail_lines(body: str, n: int = 3) -> list[str]:
    """Последние n непустых строк без строки УВЕРЕННОСТЬ."""
    lines = [ln for ln in body.split("\n") if clean(ln) and not CONF_LABEL.search(ln)]
    return lines[-n:]


def boxed(text: str) -> str | None:
    r"""Содержимое последнего \boxed{…} (с учётом вложенных скобок)."""
    i = text.rfind("\\boxed{")
    if i == -1:
        return None
    depth, j = 0, i + len("\\boxed")
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k]
    return None


def parse_value(s: str, kind: str, lang: str):
    """Значение из строки ответа по виду ответа → (value, had_percent, notes)."""
    if kind in ("number", "cannot_determine"):
        p = pick_number(clean(s), lang)
        return (p[0], p[1], []) if p else (None, False, [])
    if kind == "numbers":
        return pick_numbers(clean(s), lang)
    if kind == "bool":
        v, notes = parse_bool(clean(s))
        return v, False, notes
    if kind == "label":
        return parse_label(clean(s)), False, []
    if kind == "text":
        # Название метода: снимаем только markdown, иначе clean() испортит «L1», «k-means++» и т. п.
        v = re.sub(r"[*_`$]", "", s).strip(" \t\n.;:—-")
        return (v or None), False, []
    raise ValueError(f"Неизвестный answer_kind {kind!r}")


def mark_refusal(res: ParsedAnswer, chunk: str, lang: str, where: str) -> bool:
    """Если в куске текста «нельзя определить» или отказ — отмечает это в res и возвращает True."""
    if CANNOT_RE.search(chunk):
        res.value, res.refused = CANNOT, True
        res.notes.append(f"«нельзя определить» ({where})")
        if numbers_in(clean(chunk), lang):
            res.notes.append("рядом с «нельзя определить» есть и числа")
        return True
    if REFUSAL_RE.search(chunk):
        res.refused = True
        res.notes.append(f"отказ отвечать ({where})")
        return True
    return False


# ---------- Главная функция ----------

def parse_answer(text: str | None, answer_kind: str, lang: str = "ru") -> ParsedAnswer:
    res = ParsedAnswer()
    if not text or not text.strip():
        res.notes.append("пустой ответ")
        return res
    text, had_think = strip_think(text)
    if had_think:
        res.notes.append("вырезан блок <think>")
    body = remove_code(text)
    res.confidence, conf_notes = parse_confidence(body)
    res.notes += conf_notes

    if answer_kind == "code":
        code, code_notes = extract_code(text)
        res.notes += code_notes
        content, res.raw_answer_line, ans_notes = find_answer(body, answer_kind)
        if content is None:
            res.notes.append("нет строки ОТВЕТ")
        if code is None and REFUSAL_RE.search(body):
            res.refused = True
            res.notes.append("отказ отвечать")
        res.value = code
        res.parse_ok = code is not None
        return res

    content, res.raw_answer_line, ans_notes = find_answer(body, answer_kind)
    res.notes += ans_notes
    # У числовых задач «нельзя определить» проверяется до разбора чисел. У B, C, E — только если вердикт
    # или метку найти не удалось: «неверно — нельзя сказать, что …» остаётся вердиктом «неверно».
    numeric = answer_kind in ("number", "numbers", "cannot_determine")

    if content is not None:
        if numeric and mark_refusal(res, content, lang, "в строке ОТВЕТ"):
            res.parse_ok = True
            return res
        value, pct, notes = parse_value(content, answer_kind, lang)
        res.notes += notes
        if value is not None:
            res.value, res.had_percent, res.parse_ok = value, pct, True
            if pct:
                res.notes.append("проценты переведены в доли")
            return res
        if not numeric and mark_refusal(res, content, lang, "в строке ОТВЕТ"):
            res.parse_ok = True
            return res
        res.notes.append("строка ОТВЕТ не разобрана, запасная эвристика")
    else:
        res.notes.append("нет строки ОТВЕТ, запасная эвристика")

    # ---------- Запасная эвристика (parse_ok остаётся False) ----------
    tail = "\n".join(tail_lines(body))
    if numeric and mark_refusal(res, tail, lang, "эвристика: в конце ответа"):
        return res
    if len(body) < 500 and REFUSAL_RE.search(body):      # короткий ответ-отказ целиком (GigaChat, YandexGPT)
        res.refused = True
        res.notes.append("эвристика: отказ отвечать")
        return res
    candidates = []
    b = boxed(text)
    if b is not None:
        candidates.append(("\\boxed{}", b))
    if answer_kind == "bool":
        candidates.append(("последних строк", tail))
        first = body.strip().split("\n")[0]
        if re.match(r"^\W*(верно|неверно|true|false)(?!\w)", clean(first), re.I):
            candidates.append(("первой строки", first))
    elif answer_kind != "text":
        for ln in reversed(tail_lines(body)):   # последняя строка, где есть значение
            candidates.append(("последних строк", ln))
    for where, chunk in candidates:
        value, pct, notes = parse_value(chunk, answer_kind, lang)
        if value is not None:
            res.value, res.had_percent = value, pct
            res.raw_answer_line = res.raw_answer_line or chunk.strip()
            res.notes += notes + [f"эвристика: значение взято из {where}"]
            if pct:
                res.notes.append("проценты переведены в доли")
            return res
    if not numeric and mark_refusal(res, tail, lang, "эвристика: в конце ответа"):
        return res
    res.notes.append("значение не найдено")
    return res
