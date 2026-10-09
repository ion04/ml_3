"""Отладка промпта на dev-задачах и заморозка (шаг 07, шаг 5 хода работы).

Запуск:
  .venv/bin/python -m src.prompt_debug            # таблицы по журналу (experiment=dev) на экран
  .venv/bin/python -m src.prompt_debug --write    # то же + обновить автоматический раздел docs/prompt_debug.md
  .venv/bin/python -m src.prompt_debug freeze     # записать prompt_version и prompt_hash в config/experiment.json

Откуда берутся числа: только из logs/calls.jsonl (записи с meta.experiment == "dev").
Каждый ответ заново разбирается текущим парсером (parse.parse_answer), поэтому таблица не зависит
от производных results/dev_*.jsonl (они перезаписываются при новой версии промпта).

Что считается:
  parse_ok   — парсер нашёл строку ОТВЕТ и разобрал значение (parse_ok=True);
  формат     — parse_ok, есть УВЕРЕННОСТЬ, ответ не обрезан (finish_reason != length),
               у задач с кодом ровно один блок ```python;
  хвост      — после строки УВЕРЕННОСТЬ ничего не написано (так требует системный промпт).

freeze отказывается замораживать, если хотя бы у одной включённой модели нет ответов на все dev-задачи
с base-промптом текущей версии (без --force). Промпт отлаживается на всех моделях, а не на одной
(вопрос самоконтроля № 10).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from src import bank, llm, parse, prompts, run_bank

DOC_PATH = llm.ROOT / "docs" / "prompt_debug.md"
BEGIN = "<!-- AUTO:BEGIN (src.prompt_debug --write, руками не править) -->"
END = "<!-- AUTO:END -->"


# ---------- Записи журнала ----------

def dev_calls(log_path: Path = None) -> list[dict]:
    """Записи журнала с experiment=dev. Если вызов повторяли из-за технической ошибки,
    берётся последняя успешная попытка (или последняя ошибка, если успешных нет)."""
    path = Path(log_path or llm.REAL_LOG)
    if not path.exists():
        return []
    latest = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            meta = rec.get("meta") or {}
            if meta.get("experiment") != "dev":
                continue
            key = (rec["model_key"], meta.get("prompt_version"), meta.get("variant"), meta.get("lang"),
                   meta.get("task_id"), meta.get("shots"), meta.get("repeat"), rec.get("cache_key"))
            old = latest.get(key)
            if old is None or rec.get("error") is None or old.get("error") is not None:
                latest[key] = rec
    return list(latest.values())


def answer_kind(meta: dict, dev_tasks: dict) -> str | None:
    if meta.get("variant") == "classify_row":
        return "label"
    task = dev_tasks.get(meta.get("task_id"))
    return task["answer_kind"] if task else None


def nothing_after_confidence(text: str) -> bool:
    """После строки УВЕРЕННОСТЬ (последней) нет другого текста."""
    text, _ = parse.strip_think(text)
    matches = list(parse.CONF_LABEL.finditer(text))
    if not matches:
        return False
    rest = text[matches[-1].end():].split("\n", 1)
    return len(rest) == 1 or not rest[1].strip()


def check_call(rec: dict, dev_tasks: dict) -> dict:
    """Одна запись журнала → признаки соблюдения формата."""
    meta = rec["meta"]
    kind = answer_kind(meta, dev_tasks)
    out = {"model": rec["model_key"], "version": meta.get("prompt_version"),
           "mode": meta.get("variant"), "shots": meta.get("shots"), "task_id": meta.get("task_id"),
           "error": rec.get("error") is not None, "tokens_in": rec.get("tokens_in"),
           "tokens_out": rec.get("tokens_out")}
    if out["error"] or kind is None:
        return out
    text = rec.get("response_text") or ""
    p = parse.parse_answer(text, kind, meta.get("lang") or "ru")
    # Для кода промпт требует ровно один блок ```python (extract_code помечает «блоков кода N»).
    code, code_notes = parse.extract_code(text) if kind == "code" else ("", [])
    code_ok = code is not None and not any(n.startswith("блоков кода") for n in code_notes)
    out["parse_ok"] = p.parse_ok
    out["format_ok"] = (p.parse_ok and p.confidence is not None and rec.get("finish_reason") != "length"
                        and code_ok)
    out["tail_ok"] = nothing_after_confidence(text)
    out["truncated"] = rec.get("finish_reason") == "length"
    return out


# ---------- Таблицы ----------

def frac(rows: list[dict], field: str) -> str:
    ok = [r for r in rows if not r["error"] and field in r]
    return f"{sum(r[field] for r in ok)}/{len(ok)}" if ok else "—"


def mean(rows: list[dict], field: str) -> str:
    vals = [r[field] for r in rows if r.get(field) is not None]
    return f"{sum(vals) / len(vals):.0f}" if vals else "—"


def tables(checked: list[dict]) -> str:
    """Markdown: (1) модель × версия для base, (2) все режимы подробно, (3) токены типа E по числу примеров."""
    out = []
    versions = sorted({r["version"] for r in checked if r["version"]})
    models = sorted({r["model"] for r in checked})

    # 1. Главная таблица итераций: base-промпт на 5 dev-задачах.
    out.append("### Итерации: base-промпт на dev-задачах (parse_ok · формат)\n")
    out.append("| модель | " + " | ".join(versions) + " |")
    out.append("|---|" + "---|" * len(versions))
    for m in models:
        cells = []
        for v in versions:
            rows = [r for r in checked if r["model"] == m and r["version"] == v and r["mode"] == "base"]
            cells.append(f"{frac(rows, 'parse_ok')} · {frac(rows, 'format_ok')}" if rows else "—")
        out.append(f"| {m} | " + " | ".join(cells) + " |")

    # 2. Все режимы.
    out.append("\n### Все режимы\n")
    out.append("| модель | версия | режим | вызовов | тех. ошибок | parse_ok | формат | ничего после УВЕРЕННОСТЬ "
               "| обрезано | токены вход (ср.) | токены выход (ср.) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    groups = defaultdict(list)
    for r in checked:
        groups[(r["model"], r["version"], r["mode"])].append(r)
    for (m, v, mode), rows in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        out.append(f"| {m} | {v} | {mode} | {len(rows)} | {sum(r['error'] for r in rows)} "
                   f"| {frac(rows, 'parse_ok')} | {frac(rows, 'format_ok')} | {frac(rows, 'tail_ok')} "
                   f"| {frac(rows, 'truncated')} | {mean(rows, 'tokens_in')} | {mean(rows, 'tokens_out')} |")

    # 3. Тип E: токены по числу примеров (для оценки бюджета шага 09).
    e_rows = [r for r in checked if r["mode"] == "classify_row"]
    if e_rows:
        out.append("\n### Тип E: токены на один вызов по числу примеров\n")
        out.append("| модель | версия | примеров | вызовов | вход (ср.) | выход (ср.) | выход (макс.) |")
        out.append("|---|---|---|---|---|---|---|")
        eg = defaultdict(list)
        for r in e_rows:
            eg[(r["model"], r["version"], r["shots"])].append(r)
        for (m, v, k), rows in sorted(eg.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2] or 0)):
            mx = max((r["tokens_out"] for r in rows if r.get("tokens_out") is not None), default=None)
            out.append(f"| {m} | {v} | {k} | {len(rows)} | {mean(rows, 'tokens_in')} "
                       f"| {mean(rows, 'tokens_out')} | {mx if mx is not None else '—'} |")
    return "\n".join(out)


def build(log_path: Path = None) -> tuple[list[dict], str]:
    dev_tasks = {t["id"]: t for t in bank.load_tasks(bank.DEV_PATH)}
    checked = [check_call(r, dev_tasks) for r in dev_calls(log_path)]
    return checked, tables(checked)


def write_doc(text: str, doc_path: Path = DOC_PATH) -> None:
    """Заменяет раздел между маркерами BEGIN/END; ручной текст вокруг не трогает."""
    doc = Path(doc_path).read_text(encoding="utf-8")
    head, rest = doc.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    Path(doc_path).write_text(f"{head}{BEGIN}\n{text}\n{END}{tail}", encoding="utf-8")


# ---------- Заморозка ----------

def freeze_problems(checked: list[dict]) -> list[str]:
    """Каких моделей не хватает для заморозки текущей версии промпта."""
    problems = []
    dev_ids = {t["id"] for t in bank.load_tasks(bank.DEV_PATH)}
    models = [k for k, m in llm.load_models().items() if m["enabled"] and m["provider"] != "mock"]
    for m in models:
        done = {r["task_id"] for r in checked if r["model"] == m and r["version"] == prompts.PROMPT_VERSION
                and r["mode"] == "base" and not r["error"]}
        if dev_ids - done:
            problems.append(f"{m}: нет ответов base-промпта {prompts.PROMPT_VERSION} на "
                            f"{', '.join(sorted(dev_ids - done))}")
    return problems


def freeze(checked: list[dict], force: bool = False, cfg_path: Path = None) -> int:
    cfg_path = Path(cfg_path or run_bank.EXPERIMENT_PATH)
    problems = freeze_problems(checked)
    if problems:
        print("Промпт отлажен не на всех включённых моделях:\n" + "\n".join(f"  ✗ {p}" for p in problems))
        if not force:
            print("Заморозка не выполнена. Прогон: .venv/bin/python -m src.run_bank --experiment dev "
                  "(или --force, если модель осознанно исключена).")
            return 1
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["prompt_version"] = prompts.PROMPT_VERSION
    cfg["prompt_hash"] = prompts.prompt_hash()
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Заморожено: prompt_version={cfg['prompt_version']}, prompt_hash={cfg['prompt_hash']} "
          f"→ {run_bank.show(cfg_path)}")
    print(f'Коммит: git add -A && git commit -m "step 07: freeze prompt {prompts.PROMPT_VERSION}"')
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Отладка промпта на dev-задачах (по журналу) и заморозка")
    ap.add_argument("command", nargs="?", default="table", choices=["table", "freeze"])
    ap.add_argument("--write", action="store_true", help="обновить автоматический раздел docs/prompt_debug.md")
    ap.add_argument("--force", action="store_true", help="freeze даже без ответов всех моделей")
    args = ap.parse_args(argv)

    checked, text = build()
    print(f"PROMPT_VERSION={prompts.PROMPT_VERSION}  prompt_hash={prompts.prompt_hash()}\n")
    print(text)
    if args.write or args.command == "freeze":
        write_doc(text)
        print(f"\nОбновлено: {run_bank.show(DOC_PATH)}")
    if args.command == "freeze":
        return freeze(checked, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
