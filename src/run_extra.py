"""Дополнительные прогоны под гипотезы (шаг 7 хода работы).

Запуск:
  .venv/bin/python -m src.run_extra                    # режимы для гипотез из config/experiment.json
  .venv/bin/python -m src.run_extra cot followup_doubt # конкретные режимы
  .venv/bin/python -m src.run_extra all --demo         # все режимы на mock (проверка кода)
  .venv/bin/python -m src.run_extra cot followup_doubt --dev   # проверка режимов на dev-задачах (шаг 07)
  общие опции: --models all|key1,key2  --limit N  --bank PATH  --demo

Режимы (гипотеза → что прогоняется → таблица):
  cot             H6  задачи A, B с просьбой рассуждать по шагам        results/cot_<lang>.jsonl
                      (+ модели из extra_runs.cot.extra_models с обычным промптом: они рассуждают сами)
  selfconsistency H7  5 повторов при T=0.7 (sc_repeats, sc_temperature) results/selfconsistency_<lang>.jsonl
                      + ответ большинства                              results/selfconsistency_vote_<lang>.jsonl
  followup_doubt  H9  второй ход «Мне кажется, ты ошибся.» только для задач A, B, F,
                      решённых верно в main (нужна results/main_<lang>.jsonl) results/followup_doubt_<lang>.jsonl
  paraphrase      H8  задачи с заполненным paraphrase_<lang>             results/paraphrase_<lang>.jsonl
  lang_en         H4  основной прогон на английском                      results/main_en.jsonl
  verbal_formula  H2  задачи с formula_<lang> и verbal_<lang>, обе версии  results/verbal_formula_<lang>.jsonl
H1, H3, H5, H10, H11, H12, H13 отдельных прогонов не требуют: они считаются из main и типа E.

Языки берутся из config/experiment.json → languages (lang_en всегда прогоняет en).
Какие типы задач брать, задаёт config/experiment.json → extra_runs.<режим>.types.
--dev: задачи data/tasks_dev.jsonl, в журнал пишется experiment="dev" (гейт G1 не считает это оценочным
прогоном), таблицы results/dev_<режим>_<lang>.jsonl, followup_doubt берёт ответы из results/dev_<lang>.jsonl.
Как и run_bank, повторный запуск не делает новых вызовов (кэш журнала), --demo пишет только в demo/.
"""

import argparse
import sys
from pathlib import Path

from src import bank, grade, prompts, run_bank
from src.parse import CANNOT

# Гипотеза → режим дополнительного прогона.
MODE_OF = {"H2": "verbal_formula", "H4": "lang_en", "H6": "cot", "H7": "selfconsistency",
           "H8": "paraphrase", "H9": "followup_doubt"}
# Гипотезы без отдельных прогонов и откуда берутся их данные.
NO_EXTRA = {"H1": "main: пары original ↔ modified",
            "H3": "main: F с недостающими данными и контроль A",
            "H5": "main: типы A и B",
            "H10": "main: уверенность в строке ответа",
            "H11": "main + src.exec_code: тип D",
            "H12": "тип E: src.run_classifier",
            "H13": "main + тарифы config/pricing.json"}
# Порядок важен: followup_doubt берёт ответы main (при "en" в languages — и main_en после lang_en).
MODES = ["lang_en", "cot", "paraphrase", "verbal_formula", "selfconsistency", "followup_doubt"]


def modes_for(hypotheses: list[str]) -> list[str]:
    """Какие режимы нужны выбранным гипотезам. Печатает, какие гипотезы считаются без доп. прогонов."""
    for h in hypotheses:
        if h in NO_EXTRA:
            print(f"{h}: отдельный прогон не нужен ({NO_EXTRA[h]})")
        elif h not in MODE_OF:
            print(f"{h}: своя гипотеза — режим прогона не определён, допишите его в src/run_extra.py")
    needed = {MODE_OF[h] for h in hypotheses if h in MODE_OF}
    return [m for m in MODES if m in needed]


# ---------- Построение вызовов для каждого режима ----------

def jobs_simple(tasks, models, experiment, lang, variants, temperature, types):
    """Один вызов на задачу × модель × вариант промпта (cot, paraphrase, formula/verbal, base)."""
    jobs = []
    for m in models:
        for t in tasks:
            if types and t["type"] not in types:
                continue
            # Для пары formula/verbal нужны обе версии, иначе задачу не берём.
            if all(run_bank.has_text(t, v, lang) for v in variants):
                jobs += [run_bank.make_job(t, m, experiment, lang, v, temperature) for v in variants]
    return jobs


def jobs_followup(tasks, models, experiment, lang, temperature, types, demo, main_table="main"):
    """H9: второй ход только там, где ответ в main верный. История = вопрос main и ответ модели на него."""
    main_path = run_bank.results_path(main_table, lang, demo)
    main_rows = run_bank.read_jsonl(main_path)
    if not main_rows:
        print(f"  нет {run_bank.show(main_path)}: сначала основной прогон "
              f"(src.run_bank --experiment {main_table} --lang {lang})")
        return []
    by_id = {t["id"]: t for t in tasks}
    jobs = []
    for r in main_rows:
        task = by_id.get(r["task_id"])
        if (task is None or r["model_key"] not in models or r["variant"] != "base" or r["correct"] is not True
                or (types and task["type"] not in types)):
            continue
        history = prompts.followup_history(task, lang, r["response_text"])
        extra = {"first_call_id": r["call_id"], "first_value": r["value"], "first_correct": r["correct"]}
        jobs.append(run_bank.make_job(task, r["model_key"], experiment, lang, "followup_doubt",
                                      temperature, history=history, extra=extra))
    return jobs


# ---------- H7: ответ большинства ----------

def answer_of(row: dict):
    """Ответ для голосования: 'отказ' для отказа и «нельзя определить», None — ответ не разобран."""
    if row["error"] is not None:
        return None
    if row["refused"] or row["value"] == CANNOT:
        return "отказ"
    return row["value"]


def same_answer(a, b, tol: float) -> bool:
    """Числа совпадают, если все в пределах tolerance задачи; остальное — точное совпадение."""
    nums = (int, float)
    if isinstance(a, nums) and isinstance(b, nums) and not isinstance(a, bool) and not isinstance(b, bool):
        return grade.close(a, b, tol)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_answer(x, y, tol) for x, y in zip(a, b))
    return a == b


def vote(task: dict, rows: list[dict]) -> dict:
    """Ответ большинства по повторам одной задачи одной модели.

    Повторы перебираются по порядку repeat; ответ попадает в первую группу, с которой совпал.
    Побеждает самая большая группа; при ничьей — группа, которая появилась раньше.
    Её первый ответ и его автооценка считаются ответом большинства.
    Если ни один ответ не разобран, большинство неверно (как «ответ не найден» в main);
    correct=None, только если все вызовы закончились технической ошибкой.
    """
    tol = task.get("tolerance")
    tol = grade.DEFAULT_TOLERANCE if tol is None else tol
    rows = sorted(rows, key=lambda r: r["repeat"])
    groups = []     # [[row, row, ...], ...]
    for r in rows:
        a = answer_of(r)
        if a is None:
            continue
        for g in groups:
            if same_answer(answer_of(g[0]), a, tol):
                g.append(r)
                break
        else:
            groups.append([r])
    sizes = [len(g) for g in groups]
    top = groups[sizes.index(max(sizes))][0] if groups else {}     # первый ответ самой большой группы
    all_errors = all(r["error"] is not None for r in rows)
    return {
        "experiment": rows[0]["experiment"], "lang": rows[0]["lang"], "task_id": task["id"],
        **{f: task.get(f) for f in run_bank.TASK_FIELDS}, "task_variant": task.get("variant"),
        "model_key": rows[0]["model_key"], "temperature": rows[0]["temperature"],
        "n_repeats": len(rows), "n_voted": sum(sizes), "n_groups": len(groups),
        "majority_size": max(sizes, default=0), "tie": sizes.count(max(sizes, default=0)) > 1,
        "majority_repeat": top.get("repeat"), "majority_call_id": top.get("call_id"),
        "value": top.get("value"), "refused": top.get("refused"),
        "correct": top.get("correct", None if all_errors else False),
        "score": top.get("score", None if all_errors else 0.0),
        "repeats_correct": [r["correct"] for r in rows],
    }


def write_votes(rows: list[dict], tasks: list[dict], path: Path) -> None:
    """Пересчитывает голоса для пар (модель, задача) из rows; голоса других пар в файле сохраняются."""
    by_id = {t["id"]: t for t in tasks}
    groups = {}
    for r in rows:
        if r["task_id"] in by_id:
            groups.setdefault((r["model_key"], r["task_id"]), []).append(r)
    votes = [vote(by_id[tid], g) for (_, tid), g in groups.items()]
    run_bank.write_results(path, votes, key=lambda v: (v["model_key"], v["task_id"]))
    print(f"  ответы большинства: {run_bank.show(path)}")


# ---------- Запуск одного режима ----------

def run_mode(mode: str, tasks, models, cfg, demo: bool, dev: bool = False) -> int:
    """Выполняет режим для всех нужных языков. Возвращает число технических ошибок."""
    extra_cfg = (cfg.get("extra_runs") or {}).get(mode) or {}
    types = extra_cfg.get("types")
    t_main = cfg.get("temperature_main", 0)
    langs = ["en"] if mode == "lang_en" else cfg["languages"]
    # experiment пишется в meta журнала, table — имя таблицы results/<table>_<lang>.jsonl.
    # На dev-задачах (шаг 07) experiment всегда "dev": гейт G1 не считает такие вызовы оценочными.
    experiment = table = "main" if mode == "lang_en" else mode
    main_table = "main"
    if dev:
        experiment, main_table = "dev", "dev"
        table = "dev" if mode == "lang_en" else f"dev_{mode}"
    n_errors = 0
    for lang in langs:
        if mode == "lang_en":
            jobs = jobs_simple(tasks, models, experiment, lang, ["base"], t_main, types)
        elif mode == "cot":
            jobs = jobs_simple(tasks, models, experiment, lang, ["cot"], t_main, types)
            # Модели с собственным режимом рассуждений (qwen3-8b-think) — с обычным промптом.
            # Они берутся из config (extra_runs.cot.extra_models), а не из --models.
            if extra_cfg.get("extra_models"):
                think = run_bank.select_models(",".join(extra_cfg["extra_models"]), demo, allow_disabled=True)
                jobs += jobs_simple(tasks, think, experiment, lang, ["base"], t_main, types)
        elif mode == "paraphrase":
            jobs = jobs_simple(tasks, models, experiment, lang, ["paraphrase"], t_main, types)
        elif mode == "verbal_formula":
            jobs = jobs_simple(tasks, models, experiment, lang, ["formula", "verbal"], t_main, types)
        elif mode == "selfconsistency":
            jobs = []
            for rep in range(cfg.get("sc_repeats", 5)):
                for job in jobs_simple(tasks, models, experiment, lang, ["base"], cfg.get("sc_temperature", 0.7), types):
                    job["repeat"] = rep
                    jobs.append(job)
        elif mode == "followup_doubt":
            jobs = jobs_followup(tasks, models, experiment, lang, t_main, types, demo, main_table)
        else:
            raise ValueError(f"неизвестный режим {mode!r}")

        path = run_bank.results_path(table, lang, demo)
        print(f"{mode} → {run_bank.show(path)}: вызовов {len(jobs)}")
        if not jobs:
            print("  нет задач для этого режима (не заполнены нужные поля банка или нет верных ответов в main)")
            continue
        rows = run_bank.run_jobs(jobs, demo, f"{table}_{lang}")
        run_bank.write_results(path, rows)
        if mode == "selfconsistency":
            # Голосуют все повторы из таблицы (в том числе прошлых запусков) для задач и моделей этого запуска.
            done = {(j["model_key"], j["task"]["id"]) for j in jobs}
            sc_rows = [r for r in run_bank.read_jsonl(path) if (r["model_key"], r["task_id"]) in done
                       and r["repeat"] < cfg.get("sc_repeats", 5)]
            write_votes(sc_rows, tasks, run_bank.results_path(table + "_vote", lang, demo))
        n_errors += run_bank.count_errors(rows)
    return n_errors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Дополнительные прогоны под гипотезы")
    ap.add_argument("modes", nargs="*", help=f"selected (по умолчанию) | all | {' | '.join(MODES)}")
    ap.add_argument("--models", default="all")
    ap.add_argument("--limit", type=int, default=None, help="только первые N задач банка")
    ap.add_argument("--bank", default=None, help="файл задач (по умолчанию data/tasks.jsonl)")
    ap.add_argument("--dev", action="store_true",
                    help="отладка промпта (шаг 07): data/tasks_dev.jsonl, experiment=dev, таблицы results/dev_<режим>_<lang>.jsonl")
    ap.add_argument("--demo", action="store_true", help="mock-модели, всё пишется в demo/")
    args = ap.parse_args(argv)

    cfg = run_bank.load_experiment()
    requested = args.modes or ["selected"]
    if requested == ["selected"]:
        hyps = cfg.get("hypotheses") or []
        print(f"Гипотезы из config/experiment.json: {hyps or 'не выбраны'}")
        modes = modes_for(hyps)
    elif requested == ["all"]:
        modes = MODES
    else:
        unknown = [m for m in requested if m not in MODES]
        if unknown:
            print(f"Неизвестные режимы {unknown}, допустимо: selected, all, {', '.join(MODES)}")
            return 2
        modes = [m for m in MODES if m in requested]
    if not modes:
        print("Дополнительных прогонов не нужно.")
        return 0

    bank_path = Path(args.bank) if args.bank else (bank.DEV_PATH if args.dev else bank.BANK_PATH)
    if args.demo:
        print("DEMO: mock-модели, результаты в demo/ (это не результаты эксперимента)")
    else:
        problems = run_bank.eval_problems("dev" if args.dev else "extra", bank_path)
        if problems:
            print("Прогон не запущен:\n" + "\n".join(f"  ✗ {p}" for p in problems))
            return 1

    models = run_bank.select_models(args.models, args.demo)
    if not models:
        print("Нет доступных моделей: прогонять нечего.")
        return 2
    tasks = run_bank.select_tasks(bank.load_tasks(bank_path), None, args.limit)
    n_errors = sum(run_mode(m, tasks, models, cfg, args.demo, args.dev) for m in modes)
    return run_bank.EXIT_TECH_ERRORS if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
