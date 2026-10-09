"""Тип E: сводная таблица метрик LLM и классических моделей (шаг 09) и проверка формата пробного прогона.

Запуск:
  .venv/bin/python -m src.typeE_metrics                       # results/typeE_llm.jsonl + typeE_classical.csv → results/typeE_metrics.csv
  .venv/bin/python -m src.typeE_metrics --demo                # demo/results/typeE_llm.jsonl → demo/results/typeE_metrics.csv
  .venv/bin/python -m src.typeE_metrics check --limit 20      # формат ответов пробного прогона (код 4, если ломается)

Только числа, без выводов. Первоисточник — журнал: typeE_llm.jsonl собирается из него src.run_classifier,
эта таблица — из typeE_llm.jsonl. Ноутбук анализа может вызвать build() и получить те же числа.

Строки LLM: группа = модель × число примеров × язык; берутся только experiment="typeE", строки теста из
data/e_split.json и текущий split_hash (ответы на старое разбиение не смешиваются с новыми).
- Вызовы с технической ошибкой (сеть, 429, 5xx) — не ответы модели: в метрики не входят, считаются в n_tech_error,
  а complete=False. Такие вызовы дозапрашивает повторный запуск run_classifier.
- «Неответ» (value не 0/1: «нельзя определить», отказ отвечать, не разобрано) в основных метриках считается
  ошибкой: на строке с отказом (y=1) как пропуск отказа (fn), на строке без отказа (y=0) как ложная тревога (fp).
  Так n_eval у LLM и у классических моделей одинаковый (200 строк). Метрики только по ответам 0/1 —
  в колонках *_answered, доля таких ответов — coverage.

Строки классических моделей (из results/typeE_classical.csv, считаются без API):
- train_set=fixed, eval_set=test — LogReg, GaussianNB и Dummy(most_frequent) (константа «нет отказа»)
  на тех же 5 и 20 примерах, что видит LLM, и на тех же 200 строках теста;
- train_set=resample — среднее по 20 случайным стратифицированным наборам примеров, разброс в *_sd;
- constant(нет отказа), eval_set=full — константа на всём датасете (естественная доля отказов).
"""

import argparse
import sys

import numpy as np
import pandas as pd

from src import run_bank, typeE

METRICS_NAME = "typeE_metrics.csv"
LLM_NAME = "typeE_llm.jsonl"
MIN_PARSE_OK = 0.9        # порог проверки формата пробного прогона: доля parse_ok в каждой группе модель × примеры
EXIT_FORMAT = 4           # код выхода check: формат ломается → вернуться к шагу 07 (промпт здесь не правим)
INT_COLS = ["shots", "n_eval", "n_eval_failures", "n_answered", "n_cannot_determine", "n_refused_or_unparsed",
            "n_tech_error", "tp", "fp", "fn", "tn", "n_resamples", "tokens_in_sum", "tokens_out_sum"]
METRIC_COLS = ["f1_failure", "balanced_accuracy", "accuracy", "precision_failure", "recall_failure"]

COLUMNS = (["source", "model", "model_reported", "shots", "lang", "train_set", "eval_set",
            "n_eval", "n_eval_failures", "n_answered", "coverage", "n_cannot_determine", "n_refused_or_unparsed",
            "n_tech_error", "complete"]
           + METRIC_COLS + ["tp", "fp", "fn", "tn"]
           + ["f1_failure_answered", "balanced_accuracy_answered", "accuracy_answered"]
           + ["f1_failure_sd", "balanced_accuracy_sd", "accuracy_sd", "n_resamples"]
           + ["tokens_in_mean", "tokens_out_mean", "tokens_in_sum", "tokens_out_sum",
              "latency_s_mean", "latency_s_median", "prompt_version", "split_hash"])


def llm_rows(rows: list[dict], test_rows: list[int], split_hash: str) -> list[dict]:
    """Строки оценочного прогона E на текущем тесте (порядок не важен)."""
    test = set(test_rows)
    return [r for r in rows if r["experiment"] == "typeE" and r.get("split_hash") == split_hash
            and r["row_idx"] in test]


def is_label(value) -> bool:
    """Ответ — класс 0/1 (а не «нельзя определить» и не None). bool не считаем: парсер его не выдаёт."""
    return value in (0, 1) and not isinstance(value, bool)


def llm_group(group: list[dict]) -> dict:
    """Метрики одной группы модель × примеры × язык."""
    ok = [r for r in group if r["error"] is None]                 # ответы модели (без техошибок)
    y = np.array([r["y_true"] for r in ok], dtype=int)
    answered = np.array([is_label(r["value"]) for r in ok], dtype=bool)
    # Неответ = ошибка: предсказание противоположно истинному классу.
    pred = np.array([r["value"] if is_label(r["value"]) else 1 - r["y_true"] for r in ok], dtype=int)

    out = {"n_answered": int(answered.sum()),
           "coverage": answered.mean() if len(ok) else np.nan,
           "n_cannot_determine": sum(r["value"] == "cannot_determine" for r in ok),
           "n_refused_or_unparsed": sum(r["value"] is None for r in ok),
           "n_tech_error": len(group) - len(ok)}
    if len(ok):
        out.update(typeE.metrics(y, pred))
    else:
        out.update({"n_eval": 0, "n_eval_failures": 0})
    if answered.any():
        m = typeE.metrics(y[answered], pred[answered])
        out.update({f"{k}_answered": m[k] for k in ["f1_failure", "balanced_accuracy", "accuracy"]})

    tin = [r["tokens_in"] for r in ok if r["tokens_in"] is not None]
    tout = [r["tokens_out"] for r in ok if r["tokens_out"] is not None]
    lat = [r["latency_s"] for r in ok if r["latency_s"] is not None]
    out.update({"tokens_in_mean": np.mean(tin) if tin else np.nan, "tokens_out_mean": np.mean(tout) if tout else np.nan,
                "tokens_in_sum": sum(tin) if tin else np.nan, "tokens_out_sum": sum(tout) if tout else np.nan,
                "latency_s_mean": np.mean(lat) if lat else np.nan,
                "latency_s_median": np.median(lat) if lat else np.nan})
    out["model_reported"] = ";".join(sorted({str(r["model_reported"]) for r in ok if r.get("model_reported")}))
    out["prompt_version"] = ";".join(sorted({str(r["prompt_version"]) for r in group}))
    return out


def llm_table(rows: list[dict], split: dict) -> pd.DataFrame:
    """Таблица LLM: одна строка на модель × примеры × язык."""
    s_hash = typeE.split_hash(split)
    rows = llm_rows(rows, split["test"], s_hash)
    groups = {}
    for r in rows:
        groups.setdefault((r["model_key"], r["shots"], r["lang"]), []).append(r)
    out = []
    for (model, shots, lang), group in sorted(groups.items()):
        rec = {"source": "llm", "model": model, "shots": shots, "lang": lang,
               "train_set": "prompt" if shots else "none", "eval_set": "test", "split_hash": s_hash,
               **llm_group(group)}
        # Полная группа: все строки теста получены и нет техошибок.
        rec["complete"] = rec["n_tech_error"] == 0 and len({r["row_idx"] for r in group}) == len(split["test"])
        out.append(rec)
    return pd.DataFrame(out)


def classical_table(classical: pd.DataFrame) -> pd.DataFrame:
    """Классические модели из typeE_classical.csv: fixed и константа как есть, resample — среднее и sd."""
    c = classical
    fixed = c[(c["train_set"] == "fixed") & (c["eval_set"] == "test")].copy()
    full = c[c["eval_set"] == "full"].copy()

    res = c[(c["train_set"] == "resample") & (c["eval_set"] == "test")]
    agg = []
    for (model, shots), g in res.groupby(["model", "shots"]):
        rec = {"model": model, "shots": shots, "train_set": "resample", "eval_set": "test",
               "n_eval": int(g["n_eval"].iloc[0]), "n_eval_failures": int(g["n_eval_failures"].iloc[0]),
               "n_resamples": len(g)}
        for k in METRIC_COLS:
            rec[k] = g[k].mean()
        for k in ["f1_failure", "balanced_accuracy", "accuracy"]:
            rec[f"{k}_sd"] = g[k].std(ddof=1)
        agg.append(rec)

    out = pd.concat([fixed.sort_values(["model", "shots"]), pd.DataFrame(agg), full], ignore_index=True)
    out["source"] = "classical"
    out["lang"] = None
    # Классические модели отвечают на каждую строку.
    out["n_answered"] = out["n_eval"]
    out["coverage"] = 1.0
    out["n_cannot_determine"] = out["n_refused_or_unparsed"] = out["n_tech_error"] = 0
    out["complete"] = True
    return out


def build(llm_path, classical_path=typeE.CLASSICAL_PATH, split=None) -> pd.DataFrame:
    """Сводная таблица: сначала LLM, затем классические модели. Колонки — COLUMNS."""
    split = split or typeE.load_split()
    parts = [llm_table(run_bank.read_jsonl(llm_path), split)]
    if classical_path.exists():
        parts.append(classical_table(pd.read_csv(classical_path)))
    table = pd.concat([p for p in parts if len(p)], ignore_index=True).reindex(columns=COLUMNS)
    # Целые колонки с пропусками (у константы нет shots) — без «.0» в CSV.
    for col in INT_COLS:
        table[col] = table[col].astype("Int64")
    return table


def format_table(rows: list[dict], split: dict, limit: int | None = None) -> pd.DataFrame:
    """Формат ответов по группам модель × примеры (для пробного прогона): сколько строк, parse_ok, ответов 0/1,
    «нельзя определить», не разобрано, техошибок, обрезанных по длине (finish_reason=length)."""
    test = split["test"][:limit] if limit else split["test"]
    rows = llm_rows(rows, test, typeE.split_hash(split))
    out = []
    groups = {}
    for r in rows:
        groups.setdefault((r["model_key"], r["shots"], r["lang"]), []).append(r)
    for (model, shots, lang), g in sorted(groups.items()):
        ok = [r for r in g if r["error"] is None]
        out.append({"model": model, "shots": shots, "lang": lang, "rows": len(g), "tech_error": len(g) - len(ok),
                    "parse_ok": sum(bool(r["parse_ok"]) for r in ok),
                    "parse_ok_share": np.mean([bool(r["parse_ok"]) for r in ok]) if ok else np.nan,
                    "label_0_1": sum(is_label(r["value"]) for r in ok),
                    "cannot_determine": sum(r["value"] == "cannot_determine" for r in ok),
                    "none": sum(r["value"] is None for r in ok),
                    "finish_length": sum(r.get("finish_reason") == "length" for r in ok)})
    return pd.DataFrame(out)


def check(rows: list[dict], split: dict, limit: int | None, min_share: float) -> int:
    """Печатает таблицу формата; 0 — во всех группах доля parse_ok ≥ min_share, иначе EXIT_FORMAT.
    Нет строк — код 2 (прогон не делали). Порог проверяет только формат, а не правильность ответов."""
    table = format_table(rows, split, limit)
    if table.empty:
        print("Нет строк typeE на этом тесте: сначала пробный прогон run_classifier --limit N.")
        return 2
    print(table.to_string(index=False))
    bad = table[~(table["parse_ok_share"] >= min_share)]
    if len(bad):
        print(f"\nФормат ломается (parse_ok < {min_share:.0%}) в группах: "
              + ", ".join(f"{m}/{s}" for m, s in zip(bad["model"], bad["shots"]))
              + ".\nПромпт здесь не правим: вернуться к шагу 07 (отладка на dev, новая версия промпта).")
        return EXIT_FORMAT
    print(f"\nФормат в норме: parse_ok ≥ {min_share:.0%} во всех группах.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Тип E: таблица метрик и проверка формата пробного прогона")
    ap.add_argument("command", nargs="?", default="table", choices=["table", "check"])
    ap.add_argument("--demo", action="store_true", help="читать и писать demo/results/")
    ap.add_argument("--limit", type=int, default=None, help="check: только первые N строк теста")
    ap.add_argument("--min-parse-ok", type=float, default=MIN_PARSE_OK)
    args = ap.parse_args(argv)

    out_dir = run_bank.out_dir(args.demo)
    llm_path = out_dir / LLM_NAME
    split = typeE.load_split()
    if args.command == "check":
        return check(run_bank.read_jsonl(llm_path), split, args.limit, args.min_parse_ok)

    table = build(llm_path, split=split)
    path = out_dir / METRICS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.6f")
    n_llm = int((table["source"] == "llm").sum())
    n_incomplete = int(((table["source"] == "llm") & ~table["complete"].astype(bool)).sum())
    print(f"{path.relative_to(typeE.ROOT) if path.is_relative_to(typeE.ROOT) else path}: "
          f"строк LLM {n_llm} (неполных {n_incomplete}), классических {len(table) - n_llm}")
    if n_llm == 0:
        print("  Ответов LLM типа E нет: в таблице только классические модели.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
