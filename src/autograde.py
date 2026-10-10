"""Автопроверка всех ответов и лист выборочной проверки парсера (шаг 8 хода работы).

Запуск:
  .venv/bin/python -m src.autograde [--demo] [--bank data/tasks.jsonl] [--force-spotcheck]

1. results/graded_auto.jsonl — одна строка на один ответ модели из таблиц results/*.jsonl
   (main, cot, paraphrase, verbal_formula, selfconsistency, followup_doubt, typeE_llm; без dev_*).
   - A, F, вердикт B, E: ответ разбирается заново (parse.parse_answer) и сравнивается с эталоном
     из текущего банка (grade.grade_auto). Так правка эталона или парсера после прогона учитывается
     без новых вызовов модели. correct_at_run — оценка, поставленная при прогоне (для сверки).
   - D: строка check="d_exec" по results/d_exec.jsonl: pass@1 и корректность метрики
     (|метрика − reference_metric| ≤ tolerance · |reference_metric|). Если метрику не удалось
     извлечь из вывода или нет эталона, correct = None: корректность оценивают люди в слепых листах
     (src/blind.py). Метрика из stdout — подсказка (может взяться не то число), поэтому в листе люди
     ставят корректность для всех D, кроме автоматически верных; итог в анализе — по людям.
   - C и обоснование B автоматически не оцениваются: они в слепых листах.
   Пропускаются (число печатается): ответы с технической ошибкой, задачи без эталона,
   ответы E на старое разбиение (другой split_hash).
2. grading/parser_spotcheck.csv — 20 автопроверенных ответов для ручной проверки парсера
   (столбец human_agrees заполняет человек). Половина выборки — «подозрительные» ответы
   (parse_ok = False или есть пометки парсера, среди них до 5 ответов E с недословной меткой),
   остальные случайные. Столбец selection показывает, откуда строка. Seed — из config/experiment.json.
   Если в листе уже есть отметки человека, он не перезаписывается (--force-spotcheck перезапишет).

--demo: читает и пишет demo/results/ и demo/grading/ (демо-числа не являются результатами).
"""

import argparse
import csv
import io
import random
import sys
from pathlib import Path

from src import bank, grade, parse, run_bank, typeE

ROOT = run_bank.ROOT
GRADING_DIR = ROOT / "grading"
GRADED_NAME = "graded_auto.jsonl"
SPOTCHECK_NAME = "parser_spotcheck.csv"
N_SPOTCHECK = 20
N_FLAGGED = 10          # сколько строк выборки берётся из «подозрительных» ответов
N_FLAGGED_E = 5         # из них ответов E с пометкой «метка не дословная» (HUMAN_TODO, шаг 06)
E_NOTE = "метка не дословная"
# Из каких экспериментов ответы B, C, D идут в слепые листы (src/blind.py; exec_code запускает D тех же).
BLIND_EXPERIMENTS = ["main", "paraphrase"]
# Таблицы results/, которые не являются ответами оценочных прогонов.
SKIP_PREFIXES = ("dev_", "selfconsistency_vote_", "d_exec", "graded_auto")

SPOTCHECK_COLUMNS = ["spot_id", "selection", "type", "task_id", "lang", "answer_kind", "call_id",
                     "response_text", "raw_answer_line", "parsed_value", "refused", "parse_ok", "parse_notes",
                     "reference", "auto_correct", "auto_reason", "human_agrees", "comment"]


def grading_dir(demo: bool) -> Path:
    return run_bank.DEMO_DIR / "grading" if demo else GRADING_DIR


def answer_tables(results_dir: Path) -> list[Path]:
    """Таблицы ответов оценочных прогонов в results_dir."""
    return sorted(p for p in results_dir.glob("*.jsonl") if not p.name.startswith(SKIP_PREFIXES))


def task_for_row(row: dict, tasks: dict) -> dict | None:
    """Задача, по которой проверяется ответ. У типа E задачи в банке нет: эталон — y_true строки."""
    if row["type"] == "E":
        return {"id": row["task_id"], "type": "E", "answer_kind": "label", "reference": row["y_true"]}
    return tasks.get(row["task_id"])


def grade_answer(row: dict, task: dict) -> dict:
    """Строка graded_auto для ответа A/B/E/F: разбор заново и сравнение с текущим эталоном."""
    parsed = parse.parse_answer(row["response_text"], task["answer_kind"], row["lang"])
    graded = grade.grade_auto(task, parsed)
    manual_b = row["type"] == "B" and row["experiment"] in BLIND_EXPERIMENTS
    return {
        "check": "answer",
        "experiment": row["experiment"], "lang": row["lang"], "model_key": row["model_key"],
        "task_id": row["task_id"], "type": row["type"], "topic": row.get("topic"), "source": row.get("source"),
        "answer_kind": task["answer_kind"], "task_variant": row.get("task_variant"), "variant": row["variant"],
        "repeat": row["repeat"], "shots": row.get("shots"), "row_idx": row.get("row_idx"),
        "split_hash": row.get("split_hash"), "call_id": row["call_id"],
        "value": parsed.value, "raw_answer_line": parsed.raw_answer_line, "refused": parsed.refused,
        "parse_ok": parsed.parse_ok, "parse_notes": parsed.notes, "confidence": parsed.confidence,
        "reference": task["reference"],
        "correct": graded["correct"], "score": graded["score"], "reason": graded["reason"],
        "correct_at_run": row.get("correct"),
        # Обоснование B (в экспериментах слепых листов) оценивают люди; сам вердикт проверен здесь.
        "needs_manual": manual_b,
        "manual_part": "обоснование (слепые листы)" if manual_b else None,
    }


def grade_d(rec: dict, row: dict, task: dict | None) -> dict:
    """Строка graded_auto для ответа D по записи запуска кода (d_exec.jsonl)."""
    ref = task.get("reference_metric") if task else None
    tol = task.get("tolerance") if task else None
    tol = grade.DEFAULT_TOLERANCE if tol is None else tol
    if not rec["code_found"]:
        correct, reason = False, "блок кода не найден"
    elif not rec["pass_at_1"]:
        correct, reason = False, "таймаут" if rec["timed_out"] else f"код упал (код возврата {rec['returncode']})"
    elif rec["metric"] is None:
        correct, reason = None, "метрика не извлечена из вывода — корректность оценивают люди"
    elif ref is None:
        correct, reason = None, "у задачи нет reference_metric — корректность оценивают люди"
    elif grade.close(rec["metric"], ref, tol):
        correct, reason = True, f"метрика {rec['metric']:g} в пределах ±{tol:g} (отн.) от {ref:g}"
    else:
        correct, reason = False, f"метрика {rec['metric']:g} вне ±{tol:g} (отн.) от {ref:g}"
    return {
        "check": "d_exec",
        "experiment": rec["experiment"], "lang": rec["lang"], "model_key": rec["model_key"],
        "task_id": rec["task_id"], "type": "D", "topic": row.get("topic"), "source": row.get("source"),
        "answer_kind": "code", "task_variant": row.get("task_variant"), "variant": row["variant"],
        "repeat": row["repeat"], "shots": None, "row_idx": None, "split_hash": None, "call_id": rec["call_id"],
        "code_found": rec["code_found"], "pass_at_1": rec["pass_at_1"], "returncode": rec["returncode"],
        "timed_out": rec["timed_out"], "metric": rec["metric"], "metric_line": rec["metric_line"],
        "reference_metric": ref, "tolerance": tol,
        "correct": correct, "score": None if correct is None else float(correct), "reason": reason,
        "correct_at_run": rec.get("metric_ok"),
        "needs_manual": True,
        "manual_part": "чек-лист (слепые листы)" + ("" if correct is not None else " и корректность метрики"),
    }


def d_exec_records(results_dir: Path) -> dict:
    """call_id -> запись запуска кода D. Каждый ответ запускается один раз (pass@1): если записей
    по одному call_id несколько, берётся первая. Этой же функцией пользуются src/blind.py и src/error_candidates.py."""
    recs = {}
    for rec in run_bank.read_jsonl(results_dir / "d_exec.jsonl"):
        recs.setdefault(rec["call_id"], rec)
    return recs


def build(results_dir: Path, tasks: dict) -> tuple[list[dict], dict]:
    """Все строки graded_auto и счётчики пропусков."""
    skipped = {"техошибка": 0, "нет задачи в банке": 0, "нет эталона": 0, "E на старом разбиении": 0,
               "C (слепые листы)": 0, "D без запуска кода": 0}
    rows = []
    d_rows = {}     # call_id -> строка результатов D (для связи с d_exec)
    s_hash = typeE.current_split_hash()   # ответы E на старое разбиение не проверяем, как в typeE_metrics
    for path in answer_tables(results_dir):
        for row in run_bank.read_jsonl(path):
            if row["error"] is not None:
                skipped["техошибка"] += 1
                continue
            if row["type"] == "E" and row.get("split_hash") != s_hash:
                skipped["E на старом разбиении"] += 1
                continue
            task = task_for_row(row, tasks)
            if task is None:
                skipped["нет задачи в банке"] += 1
            elif task["type"] == "C":
                skipped["C (слепые листы)"] += 1
            elif task["type"] == "D":
                d_rows[row["call_id"]] = row
            elif task["reference"] is None:      # задачу вернули в заготовку: сравнивать не с чем
                skipped["нет эталона"] += 1
            else:
                rows.append(grade_answer(row, task))
    # Берём запуски только тех ответов D, что есть в таблицах сейчас.
    execs = d_exec_records(results_dir)
    for call_id, row in d_rows.items():
        if call_id in execs:
            rows.append(grade_d(execs[call_id], row, tasks.get(row["task_id"])))
        else:
            skipped["D без запуска кода"] += 1
    return rows, skipped


# ---------- Выборочная проверка парсера ----------

def is_flagged(r: dict) -> bool:
    return not r["parse_ok"] or bool(r["parse_notes"])


def sort_key(r: dict) -> tuple:
    """Фиксированный порядок перед случайным выбором: выборка зависит только от seed и данных."""
    return (r["experiment"], r["lang"], r["model_key"], r["task_id"], r["variant"], r["repeat"], r["shots"] or 0)


def spotcheck_sample(rows: list[dict], seed: int, n: int = N_SPOTCHECK) -> list[tuple[str, dict]]:
    """(selection, строка): до N_FLAGGED_E ответов E с недословной меткой, затем другие подозрительные
    до N_FLAGGED, остальное — случайные ответы поровну по типам (иначе почти все были бы E:
    их ~2400 из ~2900). Отказы и неразобранные тоже попадают в выборку."""
    rng = random.Random(seed)
    pool = sorted((r for r in rows if r["check"] == "answer"), key=sort_key)
    taken = set()       # номера строк pool, уже попавших в выборку
    picked = []

    def take(label, candidates, k):
        chosen = rng.sample(candidates, min(k, len(candidates)))
        taken.update(chosen)
        picked.extend((label, pool[i]) for i in chosen)

    e_flag = [i for i, r in enumerate(pool) if r["type"] == "E" and E_NOTE in " ".join(r["parse_notes"])]
    take("подозрительный E", e_flag, N_FLAGGED_E)
    flagged = [i for i, r in enumerate(pool) if is_flagged(r) and i not in taken]
    take("подозрительный", flagged, N_FLAGGED - len(picked))
    # Случайная часть: по кругу по типам, из каждого типа по одной случайной строке.
    by_type = {}
    for i, r in enumerate(pool):
        if i not in taken:
            by_type.setdefault(r["type"], []).append(i)
    for idx in by_type.values():
        rng.shuffle(idx)
    while len(picked) < n and any(by_type.values()):
        for t in sorted(by_type):
            if by_type[t] and len(picked) < n:
                picked.append(("случайный", pool[by_type[t].pop()]))
    return picked


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(fmt(v) for v in value)
    return str(value)


def spotcheck_rows(sample: list[tuple[str, dict]], texts: dict) -> list[dict]:
    out = []
    for n, (selection, r) in enumerate(sample, start=1):
        out.append({
            "spot_id": f"S{n:02d}", "selection": selection, "type": r["type"], "task_id": r["task_id"],
            "lang": r["lang"], "answer_kind": r["answer_kind"], "call_id": r["call_id"],
            "response_text": texts.get(r["call_id"], ""), "raw_answer_line": fmt(r["raw_answer_line"]),
            "parsed_value": fmt(r["value"]), "refused": fmt(r["refused"]), "parse_ok": fmt(r["parse_ok"]),
            "parse_notes": fmt(r["parse_notes"]), "reference": fmt(r["reference"]),
            "auto_correct": fmt(r["correct"]), "auto_reason": r["reason"], "human_agrees": "", "comment": "",
        })
    return out


def read_csv(path: Path) -> list[dict]:
    """CSV листа → список словарей {столбец: строка}. Пишем UTF-8 с BOM и запятой, но лист могли
    пересохранить в Excel: русский Excel ставит «;», а «CSV» без UTF-8 — кодировку cp1251. Учитываем оба."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1251")
    header = text.split("\n", 1)[0]
    delimiter = ";" if header.count(";") > header.count(",") else ","
    rows = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)
    # Короткая строка даёт None в недостающих столбцах — заменяем на пустую строку.
    return [{k: (v or "") for k, v in r.items() if k is not None} for r in rows]


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def human_started(path: Path, columns: list[str], id_col: str) -> bool:
    """Начал ли человек заполнять лист: есть отметка в столбцах columns (кроме «—»).
    Если файл не читается как наш лист (нет столбца id_col), тоже считаем, что трогать нельзя."""
    if not path.exists():
        return False
    rows = read_csv(path)
    if rows and id_col not in rows[0]:
        return True
    return any(r.get(c, "").strip() not in ("", "—") for r in rows for c in columns)


def write_spotcheck(path: Path, rows: list[dict], texts: dict, seed: int, force: bool) -> None:
    if not force and human_started(path, ["human_agrees", "comment"], "spot_id"):
        print(f"  {run_bank.show(path)} уже заполняется человеком — не перезаписываю (--force-spotcheck перезапишет)")
        return
    sample = spotcheck_rows(spotcheck_sample(rows, seed), texts)
    write_csv(path, sample, SPOTCHECK_COLUMNS)
    print(f"  выборка парсера: {run_bank.show(path)} ({len(sample)} строк)")


# ---------- CLI ----------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Автопроверка ответов и лист выборочной проверки парсера")
    ap.add_argument("--bank", default=None, help="файл задач (по умолчанию data/tasks.jsonl)")
    ap.add_argument("--demo", action="store_true", help="читать и писать demo/results/ и demo/grading/")
    ap.add_argument("--force-spotcheck", action="store_true", help="перезаписать уже заполняемый parser_spotcheck.csv")
    args = ap.parse_args(argv)

    results_dir = run_bank.out_dir(args.demo)
    if not answer_tables(results_dir):
        print(f"В {run_bank.show(results_dir)} нет таблиц ответов оценочных прогонов — проверять нечего.")
        return 2
    tasks = {t["id"]: t for t in bank.load_tasks(Path(args.bank) if args.bank else bank.BANK_PATH)}
    rows, skipped = build(results_dir, tasks)

    # Таблица производная: каждый раз пересобирается целиком из таблиц ответов и банка.
    path = results_dir / GRADED_NAME
    run_bank.write_jsonl(path, rows)

    n_auto = sum(r["correct"] is not None for r in rows)
    changed = sum(r["correct"] != r["correct_at_run"] for r in rows if r["check"] == "answer")
    print(f"Автопроверка: {run_bank.show(path)} — строк {len(rows)}, с автооценкой {n_auto}, "
          f"нужна ручная часть {sum(r['needs_manual'] for r in rows)}")
    print("  пропущено: " + ", ".join(f"{k} {v}" for k, v in skipped.items()))
    if changed:
        print(f"  оценка изменилась по сравнению с прогоном: {changed} ответов (правка эталона или парсера)")
    if skipped["D без запуска кода"]:
        print("  ВНИМАНИЕ: есть ответы D без запуска кода — сначала .venv/bin/python -m src.exec_code")

    texts = {}
    for p in answer_tables(results_dir):
        texts.update({r["call_id"]: r["response_text"] or "" for r in run_bank.read_jsonl(p)})
    seed = run_bank.load_experiment().get("seed", 42)
    write_spotcheck(grading_dir(args.demo) / SPOTCHECK_NAME, rows, texts, seed, args.force_spotcheck)
    return 0


if __name__ == "__main__":
    sys.exit(main())
