"""Проверки гейтов человека.

Запуск:
  .venv/bin/python -m src.checks g1         # готово ли всё к оценочным прогонам (код 0 = да)
  .venv/bin/python -m src.checks g2         # ручная проверка завершена (код 0 = да)
  .venv/bin/python -m src.checks hypcommit  # хэш и дата коммита с гипотезами (для отчёта)

G1 проверяет:
  1. в .env есть ключи для всех включённых моделей (кроме mock);
  2. smoke-тест проходит (запускается, только если ключи есть: он делает реальные вызовы);
  3. банк задач проходит `src.bank validate --final`;
  4. в config/experiment.json 3 гипотезы из разных групп;
  5. в hypotheses.md заполнены прогнозы, файл закоммичен и после коммита не менялся;
  6. в logs/calls.jsonl нет оценочных вызовов раньше коммита гипотез.
     Оценочными считаются все записи, кроме experiment = smoke и dev (отладка промпта).

G2 проверяет:
  1–2. grading/sheet_r1.csv и sheet_r2.csv заполнены полностью: строки совпадают с ключом
       grading/key_DO_NOT_OPEN.csv, в B и C стоит балл 0/1/2, в D — пункты чек-листа 1/0/н/п
       и корректность 1/0 там, где её не поставила автопроверка (клетки «—» заполнять не нужно);
  3. grading/parser_spotcheck.csv: не меньше 20 строк, в каждой human_agrees = да/нет;
  4. config/pricing.json: verified_by_human = true, у каждой включённой модели есть цена.
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from src import bank, llm

ROOT = Path(__file__).resolve().parent.parent
HYPOTHESES_MD = ROOT / "hypotheses.md"
EXPERIMENT_PATH = ROOT / "config" / "experiment.json"
LOG_PATH = llm.REAL_LOG
GRADING_DIR = ROOT / "grading"
PRICING_PATH = ROOT / "config" / "pricing.json"

# Группы гипотез из задания ЛР3.
HYPOTHESIS_GROUPS = {"H1": 1, "H2": 1, "H3": 1,
                     "H4": 2, "H5": 2, "H6": 2, "H7": 2,
                     "H8": 3, "H9": 3, "H10": 3, "H11": 3,
                     "H12": 4, "H13": 4}
# Поля каждого раздела hypotheses.md (значение пишется на той же строке после двоеточия).
MD_FIELDS = ["ID", "Группа", "Формулировка", "Прогноз", "Обоснование прогноза", "Метрика",
             "Критерий решения (тест, порог)"]
# Эти вызовы можно делать до коммита гипотез: проверка связи и отладка промпта на dev-задачах.
NON_EVAL_EXPERIMENTS = {"smoke", "dev"}


# ---------- 1–2. Ключи и smoke-тест ----------

def check_keys(models: dict) -> list[str]:
    """Каких переменных не хватает включённым моделям (mock не нужны ключи)."""
    problems = []
    for cfg in models.values():
        if cfg["enabled"] and cfg["provider"] != "mock":
            missing = llm.missing_env(cfg)
            if missing:
                problems.append(f"{cfg['key']}: в .env нет {', '.join(missing)}")
    return problems


def check_smoke() -> list[str]:
    proc = subprocess.run([sys.executable, "-m", "src.smoke_test"], cwd=ROOT, capture_output=True, text=True)
    if proc.returncode == 0:
        return []
    tail = [ln for ln in proc.stdout.splitlines() if "ОШИБКА" in ln or "нет переменных" in ln]
    return ["smoke-тест не прошёл (.venv/bin/python -m src.smoke_test): " + ("; ".join(tail) or proc.stderr[-300:])]


# ---------- 4. Гипотезы в config/experiment.json ----------

def check_hypotheses_config(cfg: dict) -> list[str]:
    hyps = cfg.get("hypotheses", [])
    groups = {**HYPOTHESIS_GROUPS, **cfg.get("custom_hypothesis_groups", {})}
    problems = []
    if len(hyps) != 3 or len(set(hyps)) != 3:
        problems.append(f"в config/experiment.json → hypotheses нужно 3 разные гипотезы, сейчас {hyps}")
    unknown = [h for h in hyps if h not in groups]
    if unknown:
        problems.append(f"неизвестные гипотезы {unknown}: для своей укажите группу в custom_hypothesis_groups")
    known = [groups[h] for h in hyps if h in groups]
    if len(known) != len(set(known)):
        problems.append(f"гипотезы {hyps} должны быть из разных групп (группы: {known})")
    return problems


# ---------- 5. hypotheses.md ----------

def parse_hypotheses_md(text: str) -> list[dict]:
    """Разделы «## Гипотеза N» → словарь {поле: значение} по строкам вида «- Поле: значение»."""
    sections = []
    for block in re.split(r"^## Гипотеза \d+\s*$", text, flags=re.M)[1:]:
        block = block.split("\n---")[0]   # раздел заканчивается горизонтальной чертой
        fields = dict(re.findall(r"^- ([^:\n]+):[ \t]*(.*)$", block, flags=re.M))
        sections.append({k.strip(): v.strip() for k, v in fields.items()})
    return sections


def check_hypotheses_md(text: str, selected: list[str]) -> list[str]:
    sections = parse_hypotheses_md(text)
    problems = []
    if len(sections) != 3:
        return [f"в hypotheses.md нужно 3 раздела «## Гипотеза N», найдено {len(sections)}"]
    for n, s in enumerate(sections, start=1):
        absent = [k for k in MD_FIELDS if k not in s]
        if absent:
            problems.append(f"hypotheses.md, гипотеза {n}: нет строк {', '.join(absent)}")
        empty = [k for k, v in s.items() if not v or "TODO" in v]
        if empty:
            problems.append(f"hypotheses.md, гипотеза {n}: не заполнено {', '.join(empty)} "
                            "(значение пишется на той же строке после двоеточия)")
        prediction = s.get("Прогноз", "").replace("*", "").replace("_", "").strip().lower()
        if not prediction.startswith(("подтвердится", "не подтвердится")):
            problems.append(f"hypotheses.md, гипотеза {n}: «Прогноз» должен начинаться с «подтвердится» или «не подтвердится»")
    ids = [s.get("ID", "").replace("*", "").strip() for s in sections]
    if sorted(ids) != sorted(selected):
        problems.append(f"ID в hypotheses.md {ids} не совпадают с config/experiment.json {selected}")
    return problems


def _git(args: list[str], cwd=ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def last_commit(path, cwd=ROOT) -> tuple[str, datetime] | None:
    """(хэш, время коммита) последнего коммита, изменившего файл; None, если файл не закоммичен."""
    out = _git(["log", "-1", "--format=%H %cI", "--", str(path)], cwd)
    if not out:
        return None
    sha, when = out.split()
    return sha, datetime.fromisoformat(when)


def is_modified(path, cwd=ROOT) -> bool:
    """Есть ли незакоммиченные изменения файла (включая «файл не отслеживается»)."""
    return bool(_git(["status", "--porcelain", "--", str(path)], cwd))


# ---------- 6. Нет оценочных вызовов до коммита гипотез ----------

def early_eval_calls(log_path, commit_time: datetime | None) -> list[str]:
    """Оценочные записи журнала раньше commit_time (если коммита нет — все оценочные записи)."""
    early = []
    if not Path(log_path).exists():
        return early
    with open(log_path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                early.append(f"{log_path}:{n}: битая строка журнала — разберитесь вручную (файл не переписывать)")
                continue
            meta = rec.get("meta") or {}
            exp = meta.get("experiment")
            if exp in NON_EVAL_EXPERIMENTS:
                continue
            when = datetime.fromisoformat(rec["ts_utc"])
            if commit_time is None or when < commit_time:
                early.append(f"{rec['ts_utc']} {rec.get('model_key')} experiment={exp} task={meta.get('task_id')}")
    return early


def check_hypotheses_committed(cfg: dict) -> tuple[list[str], tuple | None]:
    """Прогнозы в hypotheses.md заполнены, файл закоммичен и не менялся, список гипотез в конфиге совпадает
    с коммитом. Возвращает (проблемы, (хэш, время) коммита или None). Её же вызывают прогонщики."""
    md_problems = check_hypotheses_md(HYPOTHESES_MD.read_text(encoding="utf-8"), cfg.get("hypotheses", []))
    commit = last_commit(HYPOTHESES_MD)
    if commit is None:
        md_problems.append("hypotheses.md не закоммичен")
    else:
        if is_modified(HYPOTHESES_MD):
            md_problems.append("hypotheses.md изменён после коммита — закоммитьте (до прогонов) или верните изменения")
        # Список гипотез в config/experiment.json должен совпадать с закоммиченным вместе с прогнозом
        # (сам файл потом меняют другие шаги, например prompt_version, поэтому сравниваем только hypotheses).
        committed = _git(["show", f"{commit[0]}:config/experiment.json"])
        committed_hyps = json.loads(committed).get("hypotheses") if committed else None
        if committed_hyps != cfg.get("hypotheses", []):
            md_problems.append(f"hypotheses в config/experiment.json ({cfg.get('hypotheses', [])}) не совпадают "
                               f"с коммитом гипотез ({committed_hyps}) — закоммитьте их вместе с hypotheses.md")
    return md_problems, commit


# ---------- Гейты ----------

def g1() -> int:
    results = []   # (название проверки, список проблем)

    models = llm.load_models()
    key_problems = check_keys(models)
    results.append(("ключи в .env для включённых моделей", key_problems))
    if key_problems:
        results.append(("smoke-тест", ["пропущен: сначала нужны все ключи"]))
    else:
        results.append(("smoke-тест", check_smoke()))

    bank_errors, _ = bank.validate_files(final=True)
    bank_problems = []
    if bank_errors:
        bank_problems = [f"ошибок {len(bank_errors)}, подробности: .venv/bin/python -m src.bank validate --final"]
        bank_problems += bank_errors[:5]
    results.append(("банк задач (src.bank validate --final)", bank_problems))

    with open(EXPERIMENT_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    results.append(("3 гипотезы из разных групп в config/experiment.json", check_hypotheses_config(cfg)))

    md_problems, commit = check_hypotheses_committed(cfg)
    results.append(("прогнозы в hypotheses.md заполнены и закоммичены", md_problems))

    commit_time = commit[1] if commit else None
    early = early_eval_calls(LOG_PATH, commit_time)
    results.append(("нет оценочных вызовов в logs/calls.jsonl раньше коммита гипотез",
                    [f"найдено {len(early)} записей, первые: " + "; ".join(early[:3])] if early else []))

    print("Гейт G1 — готовность к оценочным прогонам\n")
    for title, problems in results:
        print(("  ✓ " if not problems else "  ✗ ") + title)
        for p in problems:
            print("      - " + p)
    failed = [t for t, p in results if p]
    if failed:
        print(f"\nНе выполнено условий: {len(failed)} из {len(results)}. Что делать — HUMAN_TODO.md, раздел G1.")
        return 1
    print("\nВсе условия G1 выполнены.")
    return 0


# ---------- G2: листы ручной проверки, выборка парсера, тарифы ----------

# src.autograde и src.blind импортируются внутри функций: они импортируют run_bank, а run_bank — этот модуль.

AGREE_VALUES = {"да", "нет"}


def _show(path: Path) -> str:
    """Копия run_bank.show: импортировать run_bank здесь нельзя, он сам импортирует этот модуль."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def norm(value: str) -> str:
    """Значение клетки к единому виду: «2.0» и «2,0» → «2», «Н/П», «n/a» → «н/п»."""
    v = (value or "").strip().lower().replace(",", ".")
    if v in ("н/п", "нп", "n/a", "na"):
        return "н/п"
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else v
    except ValueError:
        return v


def check_sheet(path: Path, key: list[dict]) -> list[str]:
    """Полнота одного листа. Обязательные клетки каждой строки записаны в ключе (столбец required),
    поэтому «—», поставленное проверяющим в обязательную клетку, считается незаполненным."""
    from src import autograde, blind
    if not path.exists():
        return [f"нет файла {_show(path)}"]
    rows = {r.get("item_id", ""): r for r in autograde.read_csv(path)}
    problems = []
    if sorted(rows) != sorted(k["item_id"] for k in key):
        problems.append("строки листа не совпадают с ключом (удалены, добавлены или изменены item_id) — "
                        "восстановите лист из копии или пересоздайте: .venv/bin/python -m src.blind --force")
    stale = [k["item_id"] for k in key
             if k["item_id"] in rows and blind.answer_sha(rows[k["item_id"]].get("answer_text", "")) != k["answer_sha"]]
    if stale:
        problems.append(f"текст ответа не совпадает с ключом в {len(stale)} строках (первые: {', '.join(stale[:5])}): "
                        "лист от другого набора ответов или текст правили")
    allowed = {blind.SCORE_COL: {"0", "1", "2"}, blind.CORRECT_COL: {"0", "1"},
               **{c: {"0", "1", "н/п"} for c in blind.CHECKLIST}}
    empty, bad = [], []
    for k in key:
        row = rows.get(k["item_id"], {})
        for col in filter(None, k["required"].split(";")):
            raw = row.get(col, "").strip()
            if raw in ("", blind.NA):
                empty.append(f"{k['item_id']}:{col}")
            elif norm(raw) not in allowed[col]:
                bad.append(f"{k['item_id']}:{col}={raw!r}")
    if empty:
        problems.append(f"не заполнено клеток: {len(empty)} (первые: {', '.join(empty[:5])})")
    if bad:
        problems.append(f"недопустимые значения: {len(bad)} (первые: {', '.join(bad[:5])}); "
                        "балл B/C — 0/1/2, чек-лист D — 1/0/н/п, корректность D — 1/0")
    return problems


def check_spotcheck(path: Path, min_rows: int = 20) -> list[str]:
    from src import autograde
    if not path.exists():
        return [f"нет файла {_show(path)} (создаёт .venv/bin/python -m src.autograde)"]
    rows = autograde.read_csv(path)
    problems = [] if len(rows) >= min_rows else [f"строк {len(rows)}, нужно не меньше {min_rows}"]
    empty = [r.get("spot_id") for r in rows if not (r.get("human_agrees") or "").strip()]
    bad = [r.get("spot_id") for r in rows
           if (r.get("human_agrees") or "").strip() and norm(r["human_agrees"]) not in AGREE_VALUES]
    if empty:
        problems.append(f"human_agrees не заполнен: {', '.join(empty[:10])}" + (" …" if len(empty) > 10 else ""))
    if bad:
        problems.append(f"human_agrees должен быть «да» или «нет»: {', '.join(bad[:10])}")
    return problems


def check_pricing(path: Path, models: dict) -> list[str]:
    if not path.exists():
        return [f"нет файла {_show(path)}"]
    pricing = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if pricing.get("verified_by_human") is not True:
        problems.append("verified_by_human не true: сверьте цены с официальными страницами (source_url) и поставьте true")
    for cfg in models.values():
        if cfg["enabled"] and cfg["provider"] != "mock":
            price = pricing.get("models", {}).get(cfg["key"]) or {}
            if price.get("rub_per_1m_in") is None or price.get("rub_per_1m_out") is None:
                problems.append(f"{cfg['key']}: нет цены rub_per_1m_in / rub_per_1m_out")
    return problems


def g2(grading_dir: Path = None, pricing_path: Path = None) -> int:
    from src import autograde, blind
    grading_dir = grading_dir or GRADING_DIR
    pricing_path = pricing_path or PRICING_PATH
    key_path = grading_dir / blind.KEY_NAME
    results = []
    if key_path.exists():
        key = autograde.read_csv(key_path)
        for name in blind.SHEETS:
            results.append((f"лист {name} заполнен полностью", check_sheet(grading_dir / name, key)))
    else:
        results.append(("листы слепой проверки созданы", [f"нет {_show(key_path)}: листы ещё не созданы "
                                                          "(.venv/bin/python -m src.blind после прогонов)"]))
    results.append(("выборочная проверка парсера заполнена",
                     check_spotcheck(grading_dir / autograde.SPOTCHECK_NAME)))
    results.append(("тарифы проверены человеком (config/pricing.json)", check_pricing(pricing_path, llm.load_models())))

    print("Гейт G2 — ручная проверка\n")
    for title, problems in results:
        print(("  ✓ " if not problems else "  ✗ ") + title)
        for p in problems:
            print("      - " + p)
    # Совпадение двух листов во всех оценках подозрительно: проверка должна быть независимой.
    sheets = [grading_dir / name for name in blind.SHEETS]
    if all(p.exists() for p in sheets):
        r1, r2 = (sorted(autograde.read_csv(p), key=lambda r: r["item_id"]) for p in sheets)
        grades1 = [[r.get(c, "") for c in blind.GRADE_COLUMNS] for r in r1]
        grades2 = [[r.get(c, "") for c in blind.GRADE_COLUMNS] for r in r2]
        if grades1 == grades2 and any(c not in ("", blind.NA) for row in grades1 for c in row):
            print("\n  ВНИМАНИЕ: оценки в sheet_r1 и sheet_r2 совпадают полностью. Проверка точно была независимой?")
    failed = [t for t, p in results if p]
    if failed:
        print(f"\nНе выполнено условий: {len(failed)} из {len(results)}. Что делать — HUMAN_TODO.md, раздел G2.")
        return 1
    print("\nВсе условия G2 выполнены.")
    return 0


def hypcommit() -> int:
    commit = last_commit(HYPOTHESES_MD)
    if commit is None:
        print("hypotheses.md ещё не закоммичен.")
        return 1
    print(f"{commit[0]}  {commit[1].isoformat()}" + ("  (есть незакоммиченные изменения!)" if is_modified(HYPOTHESES_MD) else ""))
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    commands = {"g1": g1, "g2": g2, "hypcommit": hypcommit}
    if len(argv) != 1 or argv[0] not in commands:
        print("Использование: .venv/bin/python -m src.checks {g1|g2|hypcommit}")
        return 2
    return commands[argv[0]]()


if __name__ == "__main__":
    sys.exit(main())
