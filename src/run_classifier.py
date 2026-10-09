"""Тип E: LLM как классификатор на AI4I 2020 (0, 5 и 20 примеров в промпте).

Запуск:
  .venv/bin/python -m src.run_classifier --models all --shots 0,5,20 --limit 20   # пробный прогон, 20 строк теста
  .venv/bin/python -m src.run_classifier --models all --shots 0,5,20              # полный прогон, 200 строк
  .venv/bin/python -m src.run_classifier --demo [--limit 20]                     # mock-модели, всё в demo/
  .venv/bin/python -m src.run_classifier --dev                                   # отладка промпта (шаг 07): 6 строк пула
  .venv/bin/python -m src.run_classifier --estimate                              # сколько вызовов и символов, без вызовов

Что делает: для каждой модели, каждого числа примеров и каждой строки теста строит промпт
(prompts.build_messages(..., "classify_row")), вызывает llm.ask с meta {experiment: "typeE", shots, row_idx},
разбирает ответ (parse_answer, answer_kind="label"), сравнивает с Machine failure и пишет
results/typeE_llm.jsonl (одна строка = один ответ; сырые ответы — в logs/calls.jsonl, связь по call_id).
Метрики (F1 «отказа», balanced accuracy) считает ноутбук анализа, здесь только таблица ответов.

Строки и примеры берутся из data/e_split.json (src/typeE.py split): тест перемешан с seed, поэтому
--limit 20 берёт начало теста, где есть оба класса. Примеры одни и те же для всех строк и моделей.

Оценочный прогон (не --demo и не --dev) запускается, только если гипотезы закоммичены (как в гейте G1)
и промпт заморожен (prompt_version и prompt_hash в config/experiment.json). --dev пишет experiment="dev"
и берёт строки из пула, а не из теста: это не оценочный прогон.
"""

import argparse
import sys

from src import checks, prompts, run_bank, typeE

VARIANT = "classify_row"


def results_path(experiment: str, demo: bool):
    name = "typeE_llm.jsonl" if experiment == "typeE" else "dev_typeE_llm.jsonl"
    return run_bank.out_dir(demo) / name


def row_key(row: dict) -> tuple:
    """Ключ строки таблицы: одна модель, один язык, одно число примеров, одна строка датасета, одно разбиение."""
    return (row["experiment"], row["lang"], row["model_key"], row["shots"], row["row_idx"], row["repeat"],
            row.get("split_hash"))


def make_task(df, row_idx: int, shot_idx: list[int], lang: str) -> dict:
    """«Задача» для run_bank: строка датасета текстом + примеры. Эталон — Machine failure этой строки."""
    row = df.iloc[row_idx]
    return {"id": f"E{row_idx}", "type": "E", "topic": "ai4i2020", "source": "AI4I 2020",
            "difficulty": None, "answer_kind": "label", "pair_id": None,
            "reference": int(row[typeE.TARGET]), "tolerance": None,
            "row_text": typeE.row_text(row, lang), "shots": typeE.shots_list(df, shot_idx, lang)}


def build_jobs(df, split, models, shots, rows, lang, experiment, temperature) -> list[dict]:
    jobs = []
    s_hash = typeE.split_hash(split)
    for model in models:
        for k in shots:
            shot_idx = split["shots"][str(k)] if k else []
            for i in rows:
                task = make_task(df, i, shot_idx, lang)
                extra = {"shots": k, "row_idx": i, "udi": int(df.iloc[i]["UDI"]), "y_true": task["reference"],
                         "split_hash": s_hash}
                jobs.append(run_bank.make_job(task, model, experiment, lang, VARIANT, temperature,
                                              extra=extra, meta={"shots": k, "row_idx": i}))
    return jobs


def split_problems() -> list[str]:
    """Разбиение и датасет закоммичены и не менялись: иначе по коммиту нельзя восстановить, на чём считали."""
    problems = []
    for path in [typeE.SPLIT_PATH, typeE.DATA_PATH]:
        name = path.relative_to(typeE.ROOT)
        if checks.last_commit(path) is None:
            problems.append(f"{name} не закоммичен")
        elif checks.is_modified(path):
            problems.append(f"{name} изменён после коммита")
    return problems


def eval_problems(cfg: dict) -> list[str]:
    """Почему оценочный прогон типа E запускать нельзя (пустой список = можно)."""
    return (checks.check_hypotheses_config(cfg) + checks.check_hypotheses_committed(cfg)[0]
            + run_bank.prompt_problems(cfg) + split_problems())


def estimate(jobs: list[dict]) -> None:
    """Объём прогона без вызовов: число вызовов и длина промптов. Токены — грубо по числу символов
    (у русского текста 1 токен ≈ 2,5–4 символа в зависимости от токенизатора модели)."""
    by_shots = {}
    for job in jobs:
        system, prompt = prompts.build_messages(job["task"], VARIANT, job["lang"])
        k = job["extra"]["shots"]
        n, chars = by_shots.get(k, (0, 0))
        by_shots[k] = (n + 1, chars + len(system) + len(prompt))
    total_calls = sum(n for n, _ in by_shots.values())
    total_chars = sum(c for _, c in by_shots.values())
    print(f"Вызовов: {total_calls}")
    for k, (n, chars) in sorted(by_shots.items()):
        print(f"  {k:>2} примеров: вызовов {n}, символов на промпт ≈ {chars / n:.0f}, "
              f"токенов на промпт ≈ {chars / n / 4:.0f}–{chars / n / 2.5:.0f}")
    print(f"Входные токены всего ≈ {total_chars / 4 / 1e3:.0f}–{total_chars / 2.5 / 1e3:.0f} тыс. "
          f"(символов {total_chars / 1e3:.0f} тыс.)")
    print(f"Выходные токены: ≈ {total_calls * 20 / 1e3:.0f} тыс., если модель отвечает двумя строками (~20 токенов); "
          f"≈ {total_calls * 300 / 1e3:.0f} тыс., если перед ответом рассуждает (~300 токенов)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Тип E: LLM-классификатор AI4I 2020")
    ap.add_argument("--models", default="all", help="all или ключи через запятую")
    ap.add_argument("--shots", default=None, help="например 0,5,20 (по умолчанию typeE.shots из конфига)")
    ap.add_argument("--limit", type=int, default=None, help="только первые N строк теста (пробный прогон)")
    ap.add_argument("--lang", default="ru", choices=prompts.LANGS)
    ap.add_argument("--demo", action="store_true", help="mock-модели, всё пишется в demo/")
    ap.add_argument("--dev", action="store_true", help="отладка промпта: строки пула, experiment=dev")
    ap.add_argument("--estimate", action="store_true", help="только посчитать объём, без вызовов")
    args = ap.parse_args(argv)

    cfg = run_bank.load_experiment()
    tcfg = typeE.load_config()
    shots = [int(k) for k in run_bank.parse_list(args.shots)] if args.shots else tcfg["shots"]
    experiment = "dev" if args.dev else "typeE"
    df = typeE.load_data()
    split = typeE.load_split()
    typeE.check_split(split, typeE.features(df)[1])
    bad = [k for k in shots if k and str(k) not in split["shots"]]
    if bad:
        print(f"Нет набора примеров для shots={bad}; есть {sorted(split['shots'], key=int)}")
        return 2
    rows = split["dev_rows"] if args.dev else split["test"]
    rows = rows[:args.limit] if args.limit else rows
    temperature = cfg.get("temperature_main", 0)

    if args.estimate:
        # Для оценки берём все включённые реальные модели, даже без ключей (вызовов нет).
        from src import llm
        models = [k for k, m in llm.load_models().items() if m["enabled"] and m["provider"] != "mock"]
        if args.models != "all":
            models = run_bank.parse_list(args.models)
        print(f"Модели: {', '.join(models)}; примеров: {shots}; строк: {len(rows)}; язык: {args.lang}")
        estimate(build_jobs(df, split, models, shots, rows, args.lang, experiment, temperature))
        return 0

    if args.demo:
        print("DEMO: mock-модели, результаты в demo/ (это не результаты эксперимента)")
    elif not args.dev:
        problems = eval_problems(cfg)
        if problems:
            print("Прогон типа E не запущен:\n" + "\n".join(f"  ✗ {p}" for p in problems))
            return 1

    models = run_bank.select_models(args.models, args.demo)
    if not models:
        print("Нет доступных моделей: прогонять нечего.")
        return 2
    jobs = build_jobs(df, split, models, shots, rows, args.lang, experiment, temperature)
    print(f"{experiment}, {args.lang}: строк {len(rows)}, примеров {shots}, моделей {len(models)}, вызовов {len(jobs)}")
    result_rows = run_bank.run_jobs(jobs, args.demo, f"{experiment}_E")
    path = results_path(experiment, args.demo)
    run_bank.write_results(path, result_rows, key=row_key)
    print(f"  таблица: {run_bank.show(path)}")
    return run_bank.EXIT_TECH_ERRORS if run_bank.count_errors(result_rows) else 0


if __name__ == "__main__":
    sys.exit(main())
