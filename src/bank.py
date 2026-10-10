"""Банк задач: схема задачи, валидатор, статистика «тип × тема», пересчёт эталонов кодом.

Одна задача = одна строка JSONL (data/tasks.jsonl — оценочный банк, data/tasks_dev.jsonl — 5 задач
для отладки промпта, в оценку не входят).

Запуск:
  .venv/bin/python -m src.bank validate           # схема + предупреждения о незаполненном
  .venv/bin/python -m src.bank validate --final   # строгая проверка перед прогонами (гейт G1)
  .venv/bin/python -m src.bank stats              # таблица «тип × тема»
  .venv/bin/python -m src.bank refs [--write]     # прогнать data/refs/<id>.py и data/d_reference/<id>.py,
                                                  # сверить с эталонами; --write записывает reference_metric для D

Поля задачи (все обязательны, null допустим там, где сказано):
  id               уникальный идентификатор (A1, B3, ...; у dev-задач префикс dev-)
  type             A|B|C|D|E|F (E в банк обычно не входит: это отдельный датасет AI4I)
  topic            linreg|logreg|knn|trees|ensembles|svm|bayes|clustering|pca|metrics
  text_ru, text_en условие на русском и английском
  reference        эталон; формат зависит от answer_kind (см. check_reference)
  tolerance        относительный допуск для чисел и метрики D (по умолчанию 0.01), иначе null (в заготовке todo можно null)
  source           учебник|свои
  difficulty       1..3
  answer_kind      number|numbers|bool|label|text|code|cannot_determine
  pair_id          общий id для связанных задач (H1: original ↔ modified и т. п.) или null
  variant          original|modified|missing_data|distractor|paraphrase|verbal|formula|null
  rubric           для B и C: словарь с критериями 0/1/2 (у C можно bonus_arguments — желательные доводы); иначе null
  reference_code   для D: путь к эталонному решению data/d_reference/<id>.py; иначе null
  reference_metric для D: метрика эталонного решения (число); иначе null
  reference_status agent_draft|verified|todo
  notes            комментарий (откуда задача, тонкости проверки)
Необязательные поля (дополнительные прогоны под гипотезы):
  paraphrase_ru, paraphrase_en   эквивалентная перефразировка условия (H8)
  formula_ru, formula_en         формульная версия условия: P(D) = ..., P(+|D) = ... (H2)
  verbal_ru, verbal_en           словесная версия с теми же числами, без обозначений вероятностей (H2)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BANK_PATH = ROOT / "data" / "tasks.jsonl"
DEV_PATH = ROOT / "data" / "tasks_dev.jsonl"
REFS_DIR = ROOT / "data" / "refs"
EXPERIMENT_PATH = ROOT / "config" / "experiment.json"
PYTHON = ROOT / ".venv" / "bin" / "python"

TYPES = ["A", "B", "C", "D", "E", "F"]
TOPICS = ["linreg", "logreg", "knn", "trees", "ensembles", "svm", "bayes", "clustering", "pca", "metrics"]
SOURCES = ["учебник", "свои"]
ANSWER_KINDS = ["number", "numbers", "bool", "label", "text", "code", "cannot_determine"]
VARIANTS = ["original", "modified", "missing_data", "distractor", "paraphrase", "verbal", "formula", None]
STATUSES = ["agent_draft", "verified", "todo"]

REQUIRED_FIELDS = ["id", "type", "topic", "text_ru", "text_en", "reference", "tolerance", "source",
                   "difficulty", "answer_kind", "pair_id", "variant", "rubric", "reference_code",
                   "reference_metric", "reference_status", "notes"]
OPTIONAL_FIELDS = ["paraphrase_ru", "paraphrase_en", "formula_ru", "formula_en", "verbal_ru", "verbal_en"]

# Какие виды ответа допустимы для каждого типа задачи.
KINDS_BY_TYPE = {
    "A": ["number", "numbers"],
    "B": ["bool"],
    "C": ["text"],
    "D": ["code"],
    "E": ["label"],
    "F": ["number", "numbers", "cannot_determine", "bool", "label", "text"],
}
# Ключи рубрики: для C все, для B достаточно последних трёх.
RUBRIC_KEYS_C = ["acceptable_methods", "required_arguments", "typical_errors", "score_2", "score_1", "score_0"]
RUBRIC_KEYS_B = ["key_arguments", "typical_errors", "score_2", "score_1", "score_0"]

# Требования --final (из задания ЛР3 и плана).
FINAL_MIN_TASKS, FINAL_MAX_TASKS = 30, 40
FINAL_MIN_PER_TYPE = 5
FINAL_TYPES = ["A", "B", "C", "D", "F"]


# ---------- Чтение и запись ----------

def load_tasks(path=BANK_PATH) -> list[dict]:
    """Читает JSONL. Пустые строки пропускает, на битом JSON падает с номером строки."""
    tasks = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: битый JSON ({e})") from None
    return tasks


def save_tasks(tasks: list[dict], path=BANK_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


# ---------- Проверка одной задачи ----------

def is_todo(value) -> bool:
    """Незаполненное значение: None, пустая строка или строка с пометкой TODO."""
    return value is None or (isinstance(value, str) and (not value.strip() or "TODO" in value))


def has_todo_anywhere(value) -> bool:
    """Есть ли пометка TODO в любом строковом значении (включая вложенные словари и списки)."""
    if isinstance(value, str):
        return "TODO" in value
    if isinstance(value, dict):
        return any(has_todo_anywhere(v) for v in value.values())
    if isinstance(value, list):
        return any(has_todo_anywhere(v) for v in value)
    return False


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def check_reference(kind: str, reference) -> str | None:
    """Соответствует ли эталон виду ответа. Возвращает текст ошибки или None."""
    if kind == "number" and not _is_number(reference):
        return "reference должен быть числом"
    if kind == "numbers" and not (isinstance(reference, list) and len(reference) >= 2
                                  and all(_is_number(x) for x in reference)):
        return "reference должен быть списком из ≥2 чисел"
    if kind == "bool" and not isinstance(reference, bool):
        return "reference должен быть true/false"
    if kind in ("label", "text") and is_todo(reference):
        return "reference должен быть непустой строкой"
    if kind == "cannot_determine" and reference != "cannot_determine":
        return 'reference должен быть "cannot_determine"'
    return None


def check_task(t: dict) -> tuple[list[str], list[str]]:
    """Ошибки и предупреждения для одной задачи (без связей между задачами)."""
    errors, warnings = [], []
    tid = t.get("id", "?")

    def err(msg):
        errors.append(f"{tid}: {msg}")

    def warn(msg):
        warnings.append(f"{tid}: {msg}")

    if not isinstance(t.get("id"), str) or not t["id"].strip():
        err("id должен быть непустой строкой")
        return errors, warnings
    missing = [k for k in REQUIRED_FIELDS if k not in t]
    if missing:
        err("нет полей " + ", ".join(missing))
        return errors, warnings
    unknown = [k for k in t if k not in REQUIRED_FIELDS + OPTIONAL_FIELDS]
    if unknown:
        warn("неизвестные поля " + ", ".join(unknown))

    # Значения из перечислений.
    for field, allowed in [("type", TYPES), ("topic", TOPICS), ("source", SOURCES),
                           ("answer_kind", ANSWER_KINDS), ("variant", VARIANTS),
                           ("reference_status", STATUSES)]:
        if t[field] not in allowed:
            err(f"{field}={t[field]!r} не из {allowed}")
    if t["difficulty"] not in (1, 2, 3):
        err("difficulty должен быть 1, 2 или 3")
    if errors:
        return errors, warnings

    kind, status = t["answer_kind"], t["reference_status"]
    if kind not in KINDS_BY_TYPE[t["type"]]:
        err(f"answer_kind={kind} не подходит для типа {t['type']} (можно {KINDS_BY_TYPE[t['type']]})")
    if t["type"] == "E":
        warn("тип E обычно живёт отдельно (AI4I 2020), а не в банке")
    if (t["variant"] == "missing_data") != (kind == "cannot_determine"):
        err("variant=missing_data и answer_kind=cannot_determine должны идти вместе")
    if t["pair_id"] is not None and t["variant"] is None:
        err("у задачи с pair_id должен быть variant")
    if t["variant"] == "modified" and t["pair_id"] is None:
        warn("variant=modified без pair_id: не с чем сравнить (H1)")

    # Допуск нужен там, где ответ сравнивается как число (в заготовке todo его можно пока не задавать).
    if kind in ("number", "numbers", "code") and not (status == "todo" and t["tolerance"] is None):
        if not (_is_number(t["tolerance"]) and 0 < t["tolerance"] < 1):
            err("tolerance должен быть числом в (0, 1) — относительный допуск")

    # Заготовка человека: проверяем только каркас.
    if status == "todo":
        return errors, warnings

    if is_todo(t["text_ru"]):
        err("text_ru не заполнен, а статус не todo")
    if is_todo(t["text_en"]):
        warn("text_en не заполнен")
    ref_problem = None if kind == "code" else check_reference(kind, t["reference"])
    if ref_problem:
        err(ref_problem)

    if t["type"] == "C":
        if not isinstance(t["rubric"], dict) or any(k not in t["rubric"] for k in RUBRIC_KEYS_C):
            err("для C нужна rubric с ключами " + ", ".join(RUBRIC_KEYS_C))
    if t["type"] == "B":
        if not isinstance(t["rubric"], dict) or any(k not in t["rubric"] for k in RUBRIC_KEYS_B):
            warn("для B желательна rubric с ключами " + ", ".join(RUBRIC_KEYS_B) + " (проверка обоснования)")
    if t["type"] == "D":
        if not t["reference_code"] or not (ROOT / t["reference_code"]).exists():
            err(f"для D нужен файл эталона reference_code ({t['reference_code']!r} не найден)")
        if not _is_number(t["reference_metric"]):
            err("для D нужна reference_metric (запустите: python -m src.bank refs --write)")

    if status == "verified" and has_todo_anywhere(t):
        err("статус verified, но в задаче остались пометки TODO")
    return errors, warnings


# ---------- Проверка банка целиком ----------

def pair_groups(tasks: list[dict]) -> dict[str, list[dict]]:
    groups = {}
    for t in tasks:
        if t.get("pair_id"):
            groups.setdefault(t["pair_id"], []).append(t)
    return groups


def complete_pairs(tasks: list[dict], other_variant: str) -> list[str]:
    """pair_id групп, где есть оригинал и хотя бы одна задача с вариантом other_variant."""
    return [pid for pid, g in pair_groups(tasks).items()
            if any(t["variant"] == "original" for t in g) and any(t["variant"] == other_variant for t in g)]


def _filled(t: dict, field: str) -> bool:
    return field in t and not is_todo(t[field])


def validate(tasks: list[dict], dev_tasks: list[dict], final: bool = False,
             hypotheses: list[str] | None = None, languages: list[str] | None = None
             ) -> tuple[list[str], list[str]]:
    """Возвращает (ошибки, предупреждения). final=True — строгие требования задания ЛР3."""
    errors, warnings = [], []
    hypotheses = hypotheses or []
    languages = languages or ["ru"]

    for t in tasks:
        e, w = check_task(t)
        errors += e
        warnings += w
    for t in dev_tasks:
        e, w = check_task(t)
        errors += ["dev " + x for x in e]
        warnings += ["dev " + x for x in w]
        # Эталон dev-задачи нужен с самого начала (отладка промпта), поэтому заготовка todo недопустима.
        # Черновик агента (agent_draft) допустим до гейта G1; перед прогонами эталон подтверждает человек (--final).
        status = t.get("reference_status")
        if status == "todo" or (final and status != "verified"):
            errors.append(f"dev {t.get('id')}: статус {status}, нужен verified (эталон подтверждает человек)")

    dev_drafts = [str(t.get("id")) for t in dev_tasks if t.get("reference_status") == "agent_draft"]
    if dev_drafts and not final:
        warnings.append(f"dev-задачи с черновым эталоном агента ({len(dev_drafts)}): {', '.join(dev_drafts)}")

    ids = [str(t.get("id")) for t in tasks]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        errors.append("повторяются id: " + ", ".join(map(str, dupes)))
    overlap = sorted(set(ids) & {str(t.get("id")) for t in dev_tasks})
    if overlap:
        errors.append("id банка совпадают с dev-задачами: " + ", ".join(overlap))

    # Пары: в каждой группе ровно один оригинал и хотя бы одна изменённая версия.
    for pid, group in pair_groups(tasks).items():
        n_orig = sum(t.get("variant") == "original" for t in group)
        if n_orig != 1 or len(group) < 2:
            errors.append(f"пара {pid}: нужен ровно 1 original и ≥1 другая версия "
                          f"(сейчас {[(t.get('id'), t.get('variant')) for t in group]})")
            continue
        # Изменённая версия должна менять ответ, иначе пара ничего не проверяет (H1).
        orig = next(t for t in group if t.get("variant") == "original")
        for t in group:
            if t.get("variant") == "modified" and t.get("reference") is not None \
                    and t.get("reference") == orig.get("reference"):
                errors.append(f"пара {pid}: у {t.get('id')} эталон совпадает с оригиналом — изменение должно менять ответ")
    if errors and not final:
        return errors, warnings

    # Сводные предупреждения о незаполненном (по списку id, чтобы вывод был коротким).
    by_status = {s: [str(t.get("id")) for t in tasks if t.get("reference_status") == s] for s in STATUSES}
    if by_status["todo"]:
        warnings.append(f"заготовки для человека (todo), {len(by_status['todo'])}: " + ", ".join(by_status["todo"]))
    if by_status["agent_draft"]:
        warnings.append(f"черновики агента, эталон не подтверждён человеком ({len(by_status['agent_draft'])}): "
                        + ", ".join(by_status["agent_draft"]))
    no_en = [t["id"] for t in tasks if is_todo(t.get("text_en"))]
    if no_en:
        warnings.append("не заполнен text_en: " + ", ".join(no_en))

    n_false = sum(t.get("reference") is False for t in tasks if t.get("type") == "B")
    n_b = sum(t.get("type") == "B" for t in tasks)

    if not final:
        return errors, warnings

    # ----- Строгие требования (--final) -----
    if not FINAL_MIN_TASKS <= len(tasks) <= FINAL_MAX_TASKS:
        errors.append(f"в банке {len(tasks)} задач, нужно {FINAL_MIN_TASKS}–{FINAL_MAX_TASKS}")
    for tp in FINAL_TYPES:
        n = sum(t.get("type") == tp for t in tasks)
        if n < FINAL_MIN_PER_TYPE:
            errors.append(f"задач типа {tp}: {n}, нужно ≥{FINAL_MIN_PER_TYPE}")
    # Задание: «половина утверждений должна быть ложной» — ровно n/2 (при нечётном n — любая из двух середин).
    if n_b and abs(n_false - n_b / 2) > 0.5:
        errors.append(f"в типе B ложных {n_false} из {n_b}, нужно половину ({n_b // 2}"
                      + (f" или {n_b // 2 + 1})" if n_b % 2 else ")"))
    not_verified = [t["id"] for t in tasks if t.get("reference_status") != "verified"]
    if not_verified:
        errors.append(f"эталон не в статусе verified ({len(not_verified)}): " + ", ".join(not_verified))
    n_own = sum(t.get("source") == "свои" for t in tasks)
    if n_own * 2 < len(tasks):
        errors.append(f"своих задач {n_own} из {len(tasks)}, нужно не меньше половины")
    if no_en:
        errors.append("text_en обязателен: " + ", ".join(no_en))
    with_todo = [t["id"] for t in tasks if has_todo_anywhere(t)]
    if with_todo:
        errors.append("остались пометки TODO: " + ", ".join(with_todo))
    missing_topics = [tp for tp in TOPICS if not any(t.get("topic") == tp for t in tasks)]
    if missing_topics:
        errors.append("не покрыты темы: " + ", ".join(missing_topics))

    # Требования выбранных гипотез к банку (см. docs/hypotheses_guide.md).
    if "H1" in hypotheses and len(complete_pairs(tasks, "modified")) < 4:
        errors.append("H1: нужно ≥4 пары original/modified")
    if "H2" in hypotheses:
        n_h2 = sum(_filled(t, "formula_ru") and _filled(t, "verbal_ru") for t in tasks)
        if n_h2 < 5:
            errors.append(f"H2: задач с formula_ru и verbal_ru {n_h2}, нужно ≥5")
        if "en" in languages:
            no_en = [t["id"] for t in tasks if _filled(t, "verbal_ru")
                     and not (_filled(t, "formula_en") and _filled(t, "verbal_en"))]
            if no_en:
                errors.append("H2 + en: нет formula_en/verbal_en у " + ", ".join(no_en))
    if "H3" in hypotheses and sum(t.get("variant") == "missing_data" for t in tasks) < 3:
        errors.append("H3: нужно ≥3 задачи с variant=missing_data")
    if "H8" in hypotheses:
        n_par = sum(_filled(t, "paraphrase_ru") for t in tasks)
        if n_par < 10:
            errors.append(f"H8: перефразировок {n_par}, нужно ≥10 (поле paraphrase_ru)")
        if "en" in languages:
            no_par_en = [t["id"] for t in tasks if _filled(t, "paraphrase_ru") and not _filled(t, "paraphrase_en")]
            if no_par_en:
                errors.append("H8 + en: нет paraphrase_en у " + ", ".join(no_par_en))
    return errors, warnings


def validate_files(final: bool = False, bank_path=BANK_PATH, dev_path=DEV_PATH,
                   experiment_path=EXPERIMENT_PATH) -> tuple[list[str], list[str]]:
    """validate() для файлов на диске; ошибки чтения тоже возвращаются как ошибки."""
    try:
        tasks = load_tasks(bank_path)
        dev = load_tasks(dev_path) if Path(dev_path).exists() else []
    except (OSError, ValueError) as e:
        return [str(e)], []
    hyps, langs = [], ["ru"]
    if Path(experiment_path).exists():
        with open(experiment_path, encoding="utf-8") as f:
            cfg = json.load(f)
        hyps, langs = cfg.get("hypotheses", []), cfg.get("languages", ["ru"])
    return validate(tasks, dev, final=final, hypotheses=hyps, languages=langs)


# ---------- Статистика ----------

def stats_tables(tasks: list[dict]) -> dict[str, pd.DataFrame]:
    df = pd.DataFrame(tasks)
    return {
        "тип × тема": pd.crosstab(df["type"], df["topic"], margins=True, margins_name="всего"),
        "тип × источник": pd.crosstab(df["type"], df["source"], margins=True, margins_name="всего"),
        "тип × статус эталона": pd.crosstab(df["type"], df["reference_status"], margins=True, margins_name="всего"),
        "тип × вариант": pd.crosstab(df["type"], df["variant"].fillna("—"), margins=True, margins_name="всего"),
    }


# ---------- Пересчёт эталонов кодом ----------

def run_script(path: Path, timeout_s: int = 600) -> tuple[str, str]:
    """Запускает скрипт эталона, возвращает (stdout, stderr)."""
    proc = subprocess.run([str(PYTHON), str(path)], capture_output=True, text=True, timeout=timeout_s, cwd=ROOT)
    return proc.stdout, proc.stderr


def parse_marker(stdout: str, marker: str) -> list[float] | None:
    """Числа из последней строки вида '<marker>: 0.1610; 0.75' (None, если строки нет)."""
    lines = [ln for ln in stdout.splitlines() if ln.startswith(marker + ":")]
    if not lines:
        return None
    return [float(x) for x in lines[-1].split(":", 1)[1].split(";") if x.strip()]


def close(a: float, b: float, rel_tol: float) -> bool:
    """Относительное сравнение; для эталона 0 — с крошечным абсолютным допуском."""
    return abs(a - b) <= max(rel_tol * abs(b), 1e-9)


def check_refs(path: Path, write: bool = False) -> int:
    """Сверяет эталоны задач файла path с выводом скриптов. Возвращает число расхождений."""
    tasks = load_tasks(path)
    problems, changed = 0, False
    for t in tasks:
        tol = t.get("tolerance") or 0.01
        script = REFS_DIR / f"{t['id']}.py"
        if script.exists() and t["answer_kind"] in ("number", "numbers"):
            out, err = run_script(script)
            got = parse_marker(out, "REFERENCE")
            expected = t["reference"] if isinstance(t["reference"], list) else [t["reference"]]
            ok = (got is not None and len(got) == len(expected)
                  and all(_is_number(e) and close(g, e, tol) for g, e in zip(got, expected)))
            problems += not ok
            print(f"  {t['id']:8} refs/{script.name:12} код: {got}  в банке: {expected}  {'ok' if ok else 'РАСХОЖДЕНИЕ ' + err[-300:]}")
        if t["type"] == "D" and t.get("reference_code") and (ROOT / t["reference_code"]).exists():
            out, err = run_script(ROOT / t["reference_code"])
            got = parse_marker(out, "METRIC")
            if got is None:
                problems += 1
                print(f"  {t['id']:8} {t['reference_code']}: нет строки METRIC  {err[-300:]}")
                continue
            metric = got[0]
            old = t.get("reference_metric")
            ok = _is_number(old) and close(metric, old, 1e-9)
            if write and not ok and t["reference_status"] == "verified" and old is not None:
                print(f"  {t['id']:8} {t['reference_code']}: METRIC {metric}, в банке {old} — статус verified, "
                      "не перезаписываю (поправьте руками, если эталон действительно изменился)")
            elif write and not ok:
                t["reference_metric"] = metric
                changed, ok = True, True
                print(f"  {t['id']:8} {t['reference_code']}: METRIC {metric} (было {old}) → записано")
            else:
                print(f"  {t['id']:8} {t['reference_code']}: METRIC {metric}  в банке: {old}  {'ok' if ok else 'РАСХОЖДЕНИЕ'}")
            problems += not ok
    if changed:
        save_tasks(tasks, path)
    return problems


# ---------- CLI ----------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Банк задач ЛР3: проверка и статистика")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_val = sub.add_parser("validate", help="проверить схему банка")
    p_val.add_argument("--final", action="store_true", help="строгая проверка перед прогонами")
    p_val.add_argument("--bank", default=str(BANK_PATH))
    p_val.add_argument("--dev", default=str(DEV_PATH))
    p_stats = sub.add_parser("stats", help="таблица «тип × тема»")
    p_stats.add_argument("--bank", default=str(BANK_PATH))
    p_refs = sub.add_parser("refs", help="сверить эталоны со скриптами data/refs и data/d_reference")
    p_refs.add_argument("--write", action="store_true", help="записать reference_metric для D")
    args = parser.parse_args(argv)

    if args.cmd == "validate":
        errors, warnings = validate_files(final=args.final, bank_path=args.bank, dev_path=args.dev)
        for w in warnings:
            print("  предупреждение:", w)
        for e in errors:
            print("  ОШИБКА:", e)
        mode = "строгая (--final)" if args.final else "схема"
        print(f"\nПроверка {mode}: ошибок {len(errors)}, предупреждений {len(warnings)}.")
        return 1 if errors else 0

    if args.cmd == "stats":
        tasks = load_tasks(args.bank)
        print(f"Задач в банке: {len(tasks)}")
        for title, table in stats_tables(tasks).items():
            print(f"\n{title}:\n{table.to_string()}")
        return 0

    if args.cmd == "refs":
        problems = 0
        for path in (BANK_PATH, DEV_PATH):
            print(path.relative_to(ROOT))
            problems += check_refs(path, write=args.write)
        print(f"\nРасхождений: {problems}")
        return 1 if problems else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
