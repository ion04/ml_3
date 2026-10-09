"""Тесты src/typeE_metrics.py: метрики LLM по группам, неответы и техошибки, классические строки,
проверка формата пробного прогона, сквозной прогон на mock (всё во временной папке)."""

import pandas as pd
import pytest

from src import run_bank, run_classifier, typeE, typeE_metrics
from tests.test_runners import env  # noqa: F401  (общая фикстура: всё пишется во временную папку)


@pytest.fixture(scope="module")
def split():
    return typeE.load_split()


def row(split, i, y, value, *, error=None, model="m", shots=0, experiment="typeE", parse_ok=True, s_hash=None):
    """Строка typeE_llm.jsonl с нужными для метрик полями. i — номер строки теста."""
    return {"experiment": experiment, "lang": "ru", "model_key": model, "model_reported": "m-1", "shots": shots,
            "row_idx": split["test"][i], "repeat": 0, "split_hash": s_hash or typeE.split_hash(split),
            "y_true": y, "value": value, "error": error, "parse_ok": parse_ok, "finish_reason": "stop",
            "tokens_in": 100, "tokens_out": 10, "latency_s": 1.0, "prompt_version": "v1"}


def test_unanswered_counted_as_error_and_tech_errors_excluded(split):
    rows = [row(split, 0, 1, 1), row(split, 1, 1, "cannot_determine"), row(split, 2, 0, 0),
            row(split, 3, 0, None, parse_ok=False), row(split, 4, 1, None, error="timeout")]
    rec = typeE_metrics.llm_group(rows)
    assert rec["n_tech_error"] == 1 and rec["n_eval"] == 4 and rec["n_answered"] == 2
    assert rec["coverage"] == 0.5 and rec["n_cannot_determine"] == 1 and rec["n_refused_or_unparsed"] == 1
    # «нельзя определить» на отказе → fn, неразобранный ответ на «нет отказа» → fp.
    assert (rec["tp"], rec["fp"], rec["fn"], rec["tn"]) == (1, 1, 1, 1)
    assert rec["f1_failure"] == 0.5 and rec["accuracy"] == 0.5
    assert rec["f1_failure_answered"] == 1.0 and rec["accuracy_answered"] == 1.0
    assert rec["tokens_in_sum"] == 400 and rec["latency_s_mean"] == 1.0   # без строки с техошибкой


def test_llm_table_filters_and_complete_flag(split):
    full = [row(split, i, int(i < 40), int(i < 40)) for i in range(len(split["test"]))]
    other = [row(split, 0, 1, 1, experiment="dev"), row(split, 1, 1, 1, s_hash="старое"),
             {**row(split, 0, 1, 1), "row_idx": -1}]                      # не строка теста
    partial = [row(split, i, 0, 0, model="p") for i in range(5)]
    t = typeE_metrics.llm_table(full + other + partial, split)
    m = t.set_index("model")
    assert m.loc["m", "n_eval"] == len(split["test"]) and bool(m.loc["m", "complete"])
    assert m.loc["m", "f1_failure"] == 1.0
    assert m.loc["p", "n_eval"] == 5 and not bool(m.loc["p", "complete"])
    assert m.loc["m", "train_set"] == "none"


def test_classical_rows_match_csv():
    csv = pd.read_csv(typeE.CLASSICAL_PATH)
    t = typeE_metrics.classical_table(csv)
    res = t[(t["train_set"] == "resample") & (t["model"] == "LogisticRegression") & (t["shots"] == 20)].iloc[0]
    src = csv[(csv["train_set"] == "resample") & (csv["model"] == "LogisticRegression")
              & (csv["shots"] == 20) & (csv["eval_set"] == "test")]
    assert res["n_resamples"] == len(src) == 20
    assert res["f1_failure"] == pytest.approx(src["f1_failure"].mean())
    assert res["f1_failure_sd"] == pytest.approx(src["f1_failure"].std(ddof=1))
    assert set(t[t["train_set"] == "fixed"]["model"]) == {"Dummy(most_frequent)", "LogisticRegression", "GaussianNB"}
    assert (t["eval_set"] == "full").sum() == 1 and (t["eval_set"] != "natural").all()


def test_check_format_exit_codes(split, capsys):
    good = [row(split, i, 0, 0) for i in range(20)]
    assert typeE_metrics.check(good, split, 20, 0.9) == 0
    bad = good[:17] + [row(split, i, 0, None, parse_ok=False) for i in range(17, 20)]   # 85% < 90%
    assert typeE_metrics.check(bad, split, 20, 0.9) == typeE_metrics.EXIT_FORMAT
    assert "шагу 07" in capsys.readouterr().out
    assert typeE_metrics.check([], split, 20, 0.9) == 2
    # Строки за пределами --limit не входят в проверку.
    assert typeE_metrics.check(good + [row(split, 30, 0, None, parse_ok=False)], split, 20, 0.9) == 0


def test_demo_end_to_end(env):  # noqa: F811
    assert run_classifier.main(["--demo", "--limit", "5"]) == 0
    assert typeE_metrics.main(["check", "--demo", "--limit", "5"]) == 0
    assert typeE_metrics.main(["--demo"]) == 0
    t = pd.read_csv(env / "demo" / "results" / "typeE_metrics.csv")
    assert list(t.columns) == typeE_metrics.COLUMNS
    llm_part = t[t["source"] == "llm"]
    assert len(llm_part) == 4 * 3 and (llm_part["n_eval"] == 5).all() and not llm_part["complete"].any()
    assert (t["source"] == "classical").sum() > 0
    assert not (env / "real_results").exists()
    assert len(run_bank.read_jsonl(env / "demo" / "results" / "typeE_llm.jsonl")) == 60
