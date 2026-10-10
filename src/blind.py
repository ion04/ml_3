"""Листы слепой ручной проверки: обоснования B, ответы C, код D (шаг 8 хода работы).

Запуск:
  .venv/bin/python -m src.blind [--experiments main,paraphrase] [--demo] [--force]

Что делает:
- берёт ответы типов B, C, D из results/<experiment>_<lang>.jsonl (по умолчанию main и paraphrase,
  все языки, для которых есть таблица; ответы с технической ошибкой пропускаются);
- убирает из текста ответа рассуждения <think>…</think> и названия моделей и компаний
  («Я — GigaChat», «As DeepSeek», «Qwen», «YandexGPT», «Алиса», ключи из config/models.json и т. п.),
  заменяя их на «[название скрыто]»; ответ, остановленный фильтром сервиса, заменяется пометкой;
- для D добавляет результат запуска кода (results/d_exec.jsonl) и автооценку корректности метрики,
  если её удалось извлечь; корректность ставят люди во всех D, кроме автоматически верных;
- пишет grading/sheet_r1.csv и grading/sheet_r2.csv — одинаковые файлы: строки сгруппированы по типу
  (B, C, D), внутри типа перемешаны с фиксированным seed из config/experiment.json; item_id анонимный;
- пишет ключ grading/key_DO_NOT_OPEN.csv: item_id → модель, задача, эксперимент, call_id.

Столбцы оценок пустые; клетки, которые к строке не относятся, заполнены «—». Проверяющий заполняет
все пустые клетки (правила — grading/INSTRUCTIONS.md). Полноту проверяет `src.checks g2`.

Если в листах уже есть оценки, файлы не перезаписываются: при тех же ответах команда ничего не делает,
при изменившемся наборе ответов — останавливается с кодом 1 (--force перезапишет, оценки пропадут).
--demo: читает demo/results/, пишет demo/grading/.
"""

import argparse
import hashlib
import random
import re
import sys
from pathlib import Path

from src import autograde, bank, grade, llm, prompts, run_bank

NA = "—"   # клетка не относится к строке, заполнять не нужно
SHEETS = ["sheet_r1.csv", "sheet_r2.csv"]
KEY_NAME = "key_DO_NOT_OPEN.csv"
TYPES = ["B", "C", "D"]
MAX_EXEC_CHARS = 1500   # сколько последних символов stdout/stderr показывать в листе

# Пункты чек-листа data/checklist_D.md: столбец листа → короткое название.
CHECKLIST = {
    "d1_preprocessing_before_split": "масштабирование/препроцессинг до разбиения",
    "d2_tuning_on_test": "подбор гиперпараметров по тесту",
    "d3_eval_on_train": "оценка на обучающей выборке",
    "d4_accuracy_imbalance": "accuracy при сильном дисбалансе",
    "d5_no_stratify": "нет стратификации",
    "d6_no_random_state": "не зафиксирован random_state",
}
SCORE_COL = "score_0_2"
CORRECT_COL = "d_result_correct"
GRADE_COLUMNS = [SCORE_COL, *CHECKLIST, CORRECT_COL]
SHEET_COLUMNS = ["item_id", "type", "task_text", "reference", "rubric", "answer_text", "exec_result",
                 "auto_metric", *GRADE_COLUMNS, "comment"]
# required — какие столбцы оценок проверяющий обязан заполнить (через «;»); answer_sha — отпечаток текста
# ответа в листе: по нему src.checks g2 замечает, что лист не соответствует ключу (например, старая копия).
KEY_COLUMNS = ["item_id", "type", "task_id", "model_key", "model_reported", "experiment", "lang", "variant",
               "call_id", "required", "answer_sha"]
# Ответ, остановленный фильтром сервиса, выдаёт модель шаблонной фразой — в лист идёт только пометка.
BLOCKED_FINISH = {"blacklist", "content_filter"}
BLOCKED_TEXT = "[ответ остановлен фильтром сервиса]"

# Названия моделей и компаний, которые выдают автора ответа (только участники эксперимента и то, чем модели
# часто себя называют). К ним добавляются ключи и идентификаторы моделей из config/models.json и model_reported
# из таблиц. Хвост вроде «-V3», «3:8b», «-2-Max» тоже убирается. Формы слов перечислены явно, чтобы не задеть
# обычные слова («сбережения»).
BRANDS = [r"Giga\s?Chat", r"Гига\s?Чат(?:а|у|ом|е)?", r"Сбер(?:а|у|ом|е|банк\w*)?", r"Sber(?:bank)?", r"SaluteDevices",
          r"Yandex\s?GPT", r"Яндекс\s?GPT", r"YaGPT", r"Yandex", r"Яндекс(?:а|у|ом|е)?", r"ya\.ru",
          r"Алис(?:а|ы|е|у|ой)", r"Alice\s?AI", r"Deep[\s-]?Seek", r"Дип\s?[Сс]ик(?:а|у|ом|е)?", r"深度求索",
          r"Qwen", r"QwQ", r"Tongyi(?:\s?Qianwen)?", r"Qianwen", r"Тонги", r"Цяньвэнь",
          r"Alibaba", r"Алибаба", r"ChatGPT", r"OpenAI", r"GPT-?\d\w*"]
# Китайские названия пишутся без пробелов, поэтому для них не проверяем границу слова.
BRANDS_CJK = ["深度求索", "通义千问", "通义", "千问"]
MASK = "[название скрыто]"


def name_regex(extra_names: list[str]) -> re.Pattern:
    names = sorted({n for n in extra_names if n and len(n) >= 3}, key=len, reverse=True)
    alts = [re.escape(n) for n in names] + BRANDS
    # (?<![^\W_]) и (?![^\W_]) — до и после названия нет буквы или цифры («_GigaChat_» из курсива ловится,
    # «сбережения» — нет). Хвост версии: «Qwen3:8b», «DeepSeek-V3.1», «GigaChat-2-Max» (дефис бывает U+2011).
    latin = r"(?<![^\W_])(?:" + "|".join(alts) + r")(?:[-‑.:][^\W_]+|\d[^\W_]*)*(?![^\W_])"
    return re.compile(latin + "|" + "|".join(BRANDS_CJK), re.I)


def model_names(rows: list[dict]) -> list[str]:
    names = []
    for cfg in llm.load_models().values():
        names += [cfg["key"], cfg["model"].split("/")[-1]]   # gpt://<folder>/yandexgpt-5.1 → yandexgpt-5.1
    names += [r.get("model_reported") or "" for r in rows]
    return names


def anonymize(text: str, pattern: re.Pattern) -> str:
    """Текст ответа без рассуждений <think>…</think> (незакрытый тег — до конца) и без названий моделей."""
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text or "", flags=re.S)
    return pattern.sub(MASK, text).strip()


# ---------- Содержимое клеток ----------

def task_text(task: dict, row: dict) -> str | None:
    """Условие, которое видела модель (для paraphrase — перефразировка); None, если поле в банке теперь пустое."""
    try:
        return prompts.task_text(task, row["variant"], row["lang"])
    except ValueError:
        return None


def bullets(items) -> str:
    return "\n".join(f"- {x}" for x in items or [])


def rubric_text(task: dict) -> str:
    r = task.get("rubric") or {}
    if task["type"] == "D":
        return ("Чек-лист (data/checklist_D.md): 1 — ошибка есть, 0 — нет, н/п — неприменимо.\n"
                + "\n".join(f"{n}. {v}" for n, v in enumerate(CHECKLIST.values(), start=1)))
    if not r:
        return "РУБРИКА В БАНКЕ НЕ ЗАПОЛНЕНА — оценивайте по общим правилам grading/INSTRUCTIONS.md"
    parts = []
    if task["type"] == "B":
        parts += ["Ключевые аргументы:\n" + bullets(r.get("key_arguments"))]
    else:
        parts += ["Приемлемые методы:\n" + bullets(r.get("acceptable_methods")),
                  "Обязательные аргументы (все нужны для 2):\n" + bullets(r.get("required_arguments"))]
        if r.get("bonus_arguments"):
            parts.append("Дополнительно (не обязательно):\n" + bullets(r["bonus_arguments"]))
    parts += ["Типичные ошибки:\n" + bullets(r.get("typical_errors")),
              f"2 — {r.get('score_2', '')}\n1 — {r.get('score_1', '')}\n0 — {r.get('score_0', '')}"]
    return "\n\n".join(parts)


def reference_text(task: dict) -> str:
    if task["type"] == "B":
        return "утверждение верно" if task["reference"] else "утверждение неверно"
    if task["type"] == "C":
        return str(task["reference"])
    if task.get("reference_metric") is None:
        return "эталонная метрика не задана — сравните с эталонным решением (data/d_reference/)"
    tol = task.get("tolerance")
    return f"эталонная метрика {task['reference_metric']} ± {tol if tol is not None else grade.DEFAULT_TOLERANCE} (отн.)"


def tail(text: str | None) -> str:
    text = (text or "").strip()
    return text if len(text) <= MAX_EXEC_CHARS else "…" + text[-MAX_EXEC_CHARS:]


def exec_text(rec: dict | None) -> str:
    if rec is None:
        return "код не запускался (нужно .venv/bin/python -m src.exec_code)"
    if not rec["code_found"]:
        return "блок кода python не найден — запускать нечего"
    status = "таймаут" if rec["timed_out"] else f"код возврата {rec['returncode']}"
    lines = [f"{status}, время {rec['duration_s']} с, pass@1 = {int(rec['pass_at_1'])}"]
    if rec.get("extract_notes"):
        lines.append("извлечение кода: " + "; ".join(rec["extract_notes"]))
    lines.append("stdout (конец):\n" + tail(rec["stdout"]))
    if (rec.get("stderr") or "").strip():
        lines.append("stderr (конец):\n" + tail(rec["stderr"]))
    return "\n".join(lines)


def d_auto(graded: dict | None) -> tuple[str, str]:
    """(текст для столбца auto_metric, значение столбца d_result_correct: NA или пусто = ставят люди).
    Люди не ставят корректность, только если автопроверка сказала «верно». «Неверно» может быть ошибкой
    извлечения (метрика берётся из последней строки stdout с числом), поэтому такие строки проверяют люди."""
    if graded is None:
        return "код не запускался — корректность оценивают люди", ""
    if graded["correct"] is True:
        return f"верно (авто): {graded['reason']}", NA
    if graded["correct"] is False:
        return f"подсказка автопроверки — неверно: {graded['reason']}", ""
    return graded["reason"], ""


# ---------- Сборка листов ----------

def collect(results_dir: Path, experiments: list[str], tasks: dict) -> list[dict]:
    """Ответы B, C, D из таблиц выбранных экспериментов (все языки)."""
    rows = []
    for exp in experiments:
        for lang in ["ru", "en"]:
            for row in run_bank.read_jsonl(results_dir / f"{exp}_{lang}.jsonl"):
                if row["type"] in TYPES and row["error"] is None and row["task_id"] in tasks:
                    rows.append(row)
    return rows


def build(results_dir: Path, experiments: list[str], tasks: dict, seed: int) -> tuple[list[dict], list[dict]]:
    """(строки листа, строки ключа). Одинаковые входные данные → одинаковые файлы."""
    rows = []
    for row in collect(results_dir, experiments, tasks):
        if task_text(tasks[row["task_id"]], row) is None:
            print(f"  пропуск {row['task_id']} ({row['experiment']}, {row['lang']}): условие в банке теперь пустое")
        else:
            rows.append(row)
    pattern = name_regex(model_names(rows))
    execs = autograde.d_exec_records(results_dir)
    rng = random.Random(seed)
    sheet, key = [], []
    for t in TYPES:
        group = sorted((r for r in rows if r["type"] == t),
                       key=lambda r: (r["experiment"], r["lang"], r["task_id"], r["model_key"], r["variant"]))
        rng.shuffle(group)
        for n, row in enumerate(group, start=1):
            task = tasks[row["task_id"]]
            item_id = f"{t}{n:03d}"
            if row.get("finish_reason") in BLOCKED_FINISH:
                answer = BLOCKED_TEXT
            else:
                answer = anonymize(row["response_text"], pattern)
            item = {c: NA for c in GRADE_COLUMNS}      # «—» = клетку заполнять не нужно
            item.update({"item_id": item_id, "type": t, "task_text": task_text(task, row),
                         "reference": reference_text(task), "rubric": rubric_text(task), "answer_text": answer,
                         "exec_result": NA, "auto_metric": NA, "comment": ""})
            if t == "D":
                rec = execs.get(row["call_id"])
                graded = autograde.grade_d(rec, row, task) if rec else None
                item["exec_result"] = exec_text(rec)
                item["auto_metric"], item[CORRECT_COL] = d_auto(graded)
                for c in CHECKLIST:
                    item[c] = ""
            else:
                item[SCORE_COL] = ""
            required = [c for c in GRADE_COLUMNS if item[c] == ""]
            sheet.append({c: item[c] for c in SHEET_COLUMNS})   # порядок столбцов как в листе
            key.append({"item_id": item_id, "type": t, "task_id": row["task_id"], "model_key": row["model_key"],
                        "model_reported": row.get("model_reported") or "", "experiment": row["experiment"],
                        "lang": row["lang"], "variant": row["variant"], "call_id": row["call_id"],
                        "required": ";".join(required), "answer_sha": answer_sha(answer)})
    return sheet, key


def answer_sha(text: str) -> str:
    """Отпечаток текста ответа. Пробелы и переводы строк схлопываются, берутся первые 1000 символов:
    при пересохранении в Excel меняются переводы строк, а слишком длинные клетки обрезаются."""
    return hashlib.sha256(" ".join(text.split())[:1000].encode("utf-8")).hexdigest()[:12]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Листы слепой проверки B/C/D и ключ деанонимизации")
    ap.add_argument("--experiments", default=",".join(autograde.BLIND_EXPERIMENTS), help="из каких таблиц брать ответы")
    ap.add_argument("--bank", default=None, help="файл задач (по умолчанию data/tasks.jsonl)")
    ap.add_argument("--demo", action="store_true", help="читать demo/results/, писать demo/grading/")
    ap.add_argument("--force", action="store_true", help="перезаписать листы, даже если в них есть оценки")
    args = ap.parse_args(argv)

    results_dir = run_bank.out_dir(args.demo)
    out = autograde.grading_dir(args.demo)
    tasks = {t["id"]: t for t in bank.load_tasks(Path(args.bank) if args.bank else bank.BANK_PATH)}
    seed = run_bank.load_experiment().get("seed", 42)
    sheet, key = build(results_dir, run_bank.parse_list(args.experiments), tasks, seed)
    if not sheet:
        print(f"В {run_bank.show(results_dir)} нет ответов B/C/D ({args.experiments}) — листы не созданы.")
        return 2

    key_path = out / KEY_NAME
    if not args.force and any(autograde.human_started(out / s, [*GRADE_COLUMNS, "comment"], "item_id") for s in SHEETS):
        old = {r["call_id"] for r in autograde.read_csv(key_path)} if key_path.exists() else set()
        if old == {k["call_id"] for k in key}:
            print("Листы уже заполняются, набор ответов тот же — ничего не меняю.")
            return 0
        print("ОСТАНОВКА: в листах уже есть оценки, а набор ответов изменился. Решите вручную: "
              "--force перезапишет листы (оценки пропадут), сначала сохраните копию.")
        return 1

    for s in SHEETS:
        autograde.write_csv(out / s, sheet, SHEET_COLUMNS)
    autograde.write_csv(key_path, key, KEY_COLUMNS)
    counts = ", ".join(f"{t}: {sum(r['type'] == t for r in sheet)}" for t in TYPES)
    manual_d = sum(r["type"] == "D" and r[CORRECT_COL] == "" for r in sheet)
    print(f"Листы: {', '.join(run_bank.show(out / s) for s in SHEETS)} — строк {len(sheet)} ({counts}); "
          f"D с ручной оценкой корректности: {manual_d}")
    print(f"Ключ: {run_bank.show(key_path)} (проверяющим не открывать)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
