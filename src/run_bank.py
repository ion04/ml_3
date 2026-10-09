"""Основной прогон банка задач (шаг 6 хода работы): задачи × модели → журнал + таблица результатов.

Запуск:
  .venv/bin/python -m src.run_bank --experiment main --models all --lang ru [--types A,B] [--limit N] [--demo]
  .venv/bin/python -m src.run_bank --experiment dev                      # отладка промпта (шаг 07), data/tasks_dev.jsonl

Что делает:
- для каждой пары «задача × модель» строит промпт (prompts.build_messages), вызывает llm.ask
  с meta {task_id, type, variant, experiment, lang, prompt_version};
- разбирает ответ (parse.parse_answer), ставит автооценку (grade.grade_auto);
- пишет производную таблицу results/<experiment>_<lang>.jsonl (одна строка = один ответ).
  Сырые ответы остаются в logs/calls.jsonl, в таблице есть call_id для связи.

Повторный запуск не делает новых вызовов: ask берёт ответы из журнала (кэш). Вызовы с технической
ошибкой в кэш не попадают, поэтому повторный запуск дозапрашивает только их, и это видно в журнале.

--demo: реальные модели заменяются mock-моделями той же роли (mock-ru1, ...), журнал и таблицы пишутся
в demo/ (demo/calls.jsonl, demo/results/). Демо-числа не являются результатами эксперимента.

Оценочный прогон (не demo и не dev) запускается, только если банк проходит validate --final,
гипотезы закоммичены (как в гейте G1) и промпт заморожен (prompt_version и prompt_hash
в config/experiment.json совпадают с src/prompts.py).
Полная проверка готовности — .venv/bin/python -m src.checks g1.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from tqdm import tqdm

from src import bank, checks, grade, llm, parse, prompts

ROOT = llm.ROOT
RESULTS_DIR = ROOT / "results"
DEMO_DIR = ROOT / "demo"
EXPERIMENT_PATH = ROOT / "config" / "experiment.json"

# Сколько запросов одновременно отправлять одному сервису (провайдер + base_url).
# GigaChat для физлиц обрабатывает запросы в один поток; локальная модель одна на GPU, и параллельные
# запросы исказили бы замер задержки и памяти. Можно переопределить полем max_workers в config/models.json.
WORKERS = {"gigachat": 1, "openai_compat": 3, "ollama": 1, "mock": 4}

# Поля задачи, которые копируются в каждую строку результатов (для анализа без повторного чтения банка).
TASK_FIELDS = ["type", "topic", "source", "difficulty", "answer_kind", "pair_id"]


# ---------- Настройки, пути ----------

def load_experiment() -> dict:
    with open(EXPERIMENT_PATH, encoding="utf-8") as f:
        return json.load(f)


def out_dir(demo: bool) -> Path:
    """Куда писать таблицы результатов: results/ или demo/results/."""
    return DEMO_DIR / "results" if demo else RESULTS_DIR


def log_path(demo: bool) -> Path:
    return DEMO_DIR / "calls.jsonl" if demo else llm.REAL_LOG


def results_path(experiment: str, lang: str, demo: bool) -> Path:
    return out_dir(demo) / f"{experiment}_{lang}.jsonl"


def show(path: Path) -> str:
    """Путь для печати: относительно корня проекта, если возможно."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def parse_list(text: str | None) -> list[str] | None:
    """'A,B' → ['A', 'B']; None → None."""
    if not text:
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


# ---------- Выбор моделей и задач ----------

def ollama_alive(base_url: str) -> bool:
    try:
        return httpx.get(base_url.rstrip("/") + "/api/version", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


def select_models(spec: str, demo: bool, allow_disabled: bool = False) -> list[str]:
    """Ключи моделей для прогона. spec: 'all' (все включённые реальные) или 'key1,key2'.

    Модель пропускается с сообщением, если она выключена (enabled=false), если для неё нет ключа в .env
    или если не отвечает Ollama. В demo-режиме каждая модель заменяется mock-моделью той же роли.
    """
    models = llm.load_models()
    if spec == "all":
        keys = [k for k, m in models.items() if m["enabled"] and m["provider"] != "mock"]
    else:
        keys = parse_list(spec) or []
    chosen = []
    for key in keys:
        cfg = models.get(key)
        if cfg is None:
            print(f"  пропуск {key}: нет в config/models.json")
            continue
        if demo:
            mock = key if cfg["provider"] == "mock" else f"mock-{cfg['role']}"
            if mock not in models:
                print(f"  пропуск {key}: нет mock-модели для роли {cfg['role']}")
            elif mock not in chosen:
                chosen.append(mock)
            continue
        if key in chosen:
            continue        # ключ указан дважды: иначе один и тот же запрос ушёл бы в API два раза
        if cfg["provider"] == "mock":
            print(f"  пропуск {key}: mock-модели только с --demo")
        elif not cfg["enabled"] and not allow_disabled:
            print(f"  пропуск {key}: выключена в config/models.json (enabled=false)")
        elif llm.missing_env(cfg):
            print(f"  пропуск {key}: в .env нет {', '.join(llm.missing_env(cfg))}")
        elif cfg["provider"] == "ollama" and not ollama_alive(cfg["base_url"]):
            print(f"  пропуск {key}: Ollama не отвечает на {cfg['base_url']} (brew services start ollama)")
        else:
            chosen.append(key)
    return chosen


def has_text(task: dict, variant: str, lang: str) -> bool:
    """Есть ли у задачи условие для этого варианта промпта (например, paraphrase_en)."""
    try:
        prompts.build_messages(task, variant, lang)
        return True
    except ValueError:
        return False


def select_tasks(tasks: list[dict], types: list[str] | None = None, limit: int | None = None) -> list[dict]:
    """Задачи без заготовок todo, нужных типов, не больше limit."""
    todo = [t["id"] for t in tasks if t.get("reference_status") == "todo"]
    if todo:
        print(f"  пропущено заготовок todo (их заполняет человек): {len(todo)}")
    chosen = [t for t in tasks if t.get("reference_status") != "todo" and (not types or t["type"] in types)]
    return chosen[:limit] if limit else chosen


# ---------- Проверка перед оценочным прогоном ----------

def eval_problems(experiment: str, bank_path: Path) -> list[str]:
    """Почему оценочный прогон запускать нельзя (пустой список = можно).

    Задание: гипотезы и прогноз коммитятся ДО оценочных прогонов, промпт отлаживается только на dev.
    """
    problems = []
    is_dev_bank = Path(bank_path).resolve() == bank.DEV_PATH.resolve()
    if experiment == "dev":
        if not is_dev_bank:
            problems.append("experiment=dev запускается только на data/tasks_dev.jsonl")
        return problems
    if is_dev_bank:
        problems.append("dev-задачи прогоняются только с --experiment dev")
        return problems
    errors, _ = bank.validate_files(final=True, bank_path=bank_path)
    if errors:
        problems.append(f"банк не проходит validate --final ({len(errors)} ошибок): "
                        ".venv/bin/python -m src.bank validate --final")
    cfg = load_experiment()
    # Те же проверки гипотез, что в гейте G1: 3 гипотезы из разных групп, прогнозы заполнены и закоммичены.
    problems += checks.check_hypotheses_config(cfg)
    problems += checks.check_hypotheses_committed(cfg)[0]
    # Промпт заморожен на шаге 07: версия и отпечаток текстов совпадают с записанными при заморозке.
    if cfg.get("prompt_version") != prompts.PROMPT_VERSION or cfg.get("prompt_hash") != prompts.prompt_hash():
        problems.append(f"промпт не заморожен или изменён после заморозки: в config/experiment.json "
                        f"prompt_version={cfg.get('prompt_version')!r}, prompt_hash={cfg.get('prompt_hash')!r}; "
                        f"в src/prompts.py {prompts.PROMPT_VERSION!r}, {prompts.prompt_hash()!r} (заморозка — шаг 07)")
    return problems


# ---------- Один вызов → одна строка результатов ----------

def make_job(task, model_key, experiment, lang, variant="base", temperature=0.0, repeat=0,
             history=None, extra=None) -> dict:
    """Описание одного вызова. extra — дополнительные поля для строки результатов."""
    return {"task": task, "model_key": model_key, "experiment": experiment, "lang": lang,
            "variant": variant, "temperature": temperature, "repeat": repeat,
            "history": history, "extra": extra or {}}


def make_row(job: dict, res: llm.LLMResult) -> dict:
    """Строка таблицы результатов: кто, что спросили, что разобрал парсер, автооценка."""
    task = job["task"]
    parsed = parse.parse_answer(res.text, task["answer_kind"], job["lang"])
    if res.error is not None:
        graded = {"correct": None, "score": None, "reason": "техническая ошибка вызова, ответа нет"}
    else:
        graded = grade.grade_auto(task, parsed)
    row = {
        "experiment": job["experiment"], "lang": job["lang"], "task_id": task["id"],
        **{f: task.get(f) for f in TASK_FIELDS},
        "task_variant": task.get("variant"),        # original / modified / missing_data ... (из банка)
        "variant": job["variant"],                  # вариант промпта: base / cot / paraphrase ...
        "model_key": job["model_key"], "model_reported": res.model_reported,
        "repeat": job["repeat"], "temperature": res.effective_temperature,
        "prompt_version": prompts.PROMPT_VERSION,
        "call_id": res.call_id, "error": res.error, "finish_reason": res.finish_reason,
        "tokens_in": res.tokens_in, "tokens_out": res.tokens_out, "latency_s": res.latency_s,
        "response_text": res.text,
        "value": parsed.value, "raw_answer_line": parsed.raw_answer_line, "confidence": parsed.confidence,
        "refused": parsed.refused, "parse_ok": parsed.parse_ok, "parse_notes": parsed.notes,
        "correct": graded["correct"], "score": graded["score"], "reason": graded["reason"],
    }
    row.update(job["extra"])
    return row


def run_job(job: dict, log: Path) -> tuple[llm.LLMResult, dict]:
    task = job["task"]
    system, prompt = prompts.build_messages(task, job["variant"], job["lang"])
    meta = {"task_id": task["id"], "type": task["type"], "variant": job["variant"],
            "experiment": job["experiment"], "lang": job["lang"], "prompt_version": prompts.PROMPT_VERSION}
    res = llm.ask(job["model_key"], prompt, job["temperature"], system=system, history=job["history"],
                  meta=meta, repeat=job["repeat"], log_path=log)
    return res, make_row(job, res)


def run_jobs(jobs: list[dict], demo: bool, desc: str) -> list[dict]:
    """Выполняет вызовы параллельно: у каждого сервиса свой пул из WORKERS потоков.
    Возвращает строки результатов в порядке jobs и печатает сводку по моделям."""
    models = llm.load_models()
    log = log_path(demo)
    pools = {}      # сервис (провайдер + base_url) -> пул потоков
    futures = {}    # future -> номер вызова в jobs
    for i, job in enumerate(jobs):
        cfg = models[job["model_key"]]
        service = (cfg["provider"], cfg.get("base_url"))
        if service not in pools:
            pools[service] = ThreadPoolExecutor(max_workers=cfg.get("max_workers") or WORKERS[cfg["provider"]])
        futures[pools[service].submit(run_job, job, log)] = i

    rows = [None] * len(jobs)
    # model_key -> счётчики для сводки (порядок моделей — как в jobs)
    stats = {job["model_key"]: {"всего": 0, "новых": 0, "из кэша": 0, "ошибок": 0, "parse_ok": 0} for job in jobs}
    for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="вызов"):
        i = futures[fut]
        res, rows[i] = fut.result()
        s = stats[jobs[i]["model_key"]]
        s["всего"] += 1
        s["из кэша" if res.cached else "новых"] += 1
        s["ошибок"] += res.error is not None
        s["parse_ok"] += rows[i]["parse_ok"]
    for pool in pools.values():
        pool.shutdown()

    for key, s in stats.items():
        print(f"  {key:22} " + "  ".join(f"{k}: {v}" for k, v in s.items()))
    return rows


# ---------- Таблица результатов ----------

def row_key(row: dict) -> tuple:
    return (row["experiment"], row["lang"], row["model_key"], row["task_id"], row["variant"], row["repeat"])


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_results(path: Path, rows: list[dict]) -> None:
    """Обновляет таблицу: строки с тем же ключом заменяются, новые добавляются в конец.
    Таблица производная (её всегда можно пересобрать из журнала), поэтому файл переписывается целиком.
    Так частичные прогоны (--types, --limit) не стирают строки, посчитанные раньше."""
    table = {row_key(r): r for r in read_jsonl(path)}
    for r in rows:
        table[row_key(r)] = r
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in table.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def count_errors(rows: list[dict]) -> int:
    n = sum(r["error"] is not None for r in rows)
    if n:
        print(f"  ВНИМАНИЕ: {n} вызовов с технической ошибкой. Запустите команду ещё раз: "
              "дозапрашиваются только они (повтор виден в журнале).")
    return n


# ---------- CLI ----------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Прогон банка задач через LLM (base-промпт)")
    ap.add_argument("--experiment", default="main", choices=["main", "dev"])
    ap.add_argument("--models", default="all", help="all или ключи через запятую")
    # Английский основной прогон (H4) делает run_extra lang_en: там задаются типы задач (extra_runs.lang_en).
    ap.add_argument("--lang", default="ru", help="ru, en или ru,en")
    ap.add_argument("--types", default=None, help="например A,B")
    ap.add_argument("--limit", type=int, default=None, help="только первые N задач")
    ap.add_argument("--bank", default=None, help="файл задач (по умолчанию data/tasks.jsonl, для dev — data/tasks_dev.jsonl)")
    ap.add_argument("--demo", action="store_true", help="mock-модели, всё пишется в demo/")
    args = ap.parse_args(argv)

    cfg = load_experiment()
    bank_path = Path(args.bank) if args.bank else (bank.DEV_PATH if args.experiment == "dev" else bank.BANK_PATH)
    langs = parse_list(args.lang)
    if args.demo:
        print("DEMO: mock-модели, результаты в demo/ (это не результаты эксперимента)")
    else:
        problems = eval_problems(args.experiment, bank_path)
        if problems:
            print("Прогон не запущен:\n" + "\n".join(f"  ✗ {p}" for p in problems))
            return 1

    models = select_models(args.models, args.demo)
    if not models:
        print("Нет доступных моделей: прогонять нечего.")
        return 2
    tasks = select_tasks(bank.load_tasks(bank_path), parse_list(args.types), args.limit)
    temperature = cfg.get("temperature_main", 0)

    n_errors = 0
    for lang in langs:
        jobs = [make_job(t, m, args.experiment, lang, "base", temperature)
                for m in models for t in tasks if has_text(t, "base", lang)]
        print(f"{args.experiment}, {lang}: задач {len(tasks)}, моделей {len(models)}, вызовов {len(jobs)}")
        rows = run_jobs(jobs, args.demo, f"{args.experiment}_{lang}")
        path = results_path(args.experiment, lang, args.demo)
        write_results(path, rows)
        print(f"  таблица: {show(path)}")
        n_errors += count_errors(rows)
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
