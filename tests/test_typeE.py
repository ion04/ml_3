"""Тесты типа E: данные AI4I 2020, разбиение, текст строки, классические модели, run_classifier на mock.

Сеть не нужна: data/ai4i2020.csv лежит в репозитории. LLM-часть проверяется только через --demo
(фикстура env из test_runners подменяет demo/, logs/ и results/ временными папками).
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import llm, parse, prompts, run_bank, run_classifier, typeE
from tests.test_runners import env  # noqa: F401  (общая фикстура: всё пишется во временную папку)


@pytest.fixture(scope="module")
def df():
    return typeE.load_data()


@pytest.fixture(scope="module")
def cfg():
    return typeE.load_config()


# ---------- Данные ----------

def test_normalize_columns_zip_style():
    raw = pd.DataFrame(columns=["UDI", "Product ID", "Type", "Air temperature [K]", "Process temperature [K]",
                                "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure"])
    assert list(typeE.normalize_columns(raw).columns) == ["UDI", "Product ID", "Type", *typeE.NUMERIC, typeE.TARGET]
    assert "UDI" in typeE.normalize_columns(pd.DataFrame(columns=["UID"])).columns     # вариант ucimlrepo


def test_features_have_no_leaks(df):
    X, y = typeE.features(df)
    assert list(X.columns) == ["Type", "Air temperature", "Process temperature", "Rotational speed",
                               "Torque", "Tool wear"]
    assert not set(X.columns) & set(typeE.LEAK_COLS + typeE.ID_COLS)
    assert set(np.unique(y)) == {0, 1} and len(df) == 10000
    assert (df["UDI"] == np.arange(1, len(df) + 1)).all()          # row_idx = UDI - 1


# ---------- Разбиение ----------

def test_committed_split_is_reproducible(df, cfg):
    """data/e_split.json совпадает с тем, что даёт код при текущем seed и настройках."""
    assert typeE.make_split(df, cfg) == typeE.load_split()


@pytest.mark.parametrize("mode,n_fail", [("enriched", 40), ("natural", 7)])
def test_split_invariants(df, cfg, mode, n_fail):
    split = typeE.make_split(df, {**cfg, "test_mode": mode})
    y = typeE.features(df)[1]
    typeE.check_split(split, y)                     # тест ∩ примеры = ∅, 5 ⊂ 20, состав классов
    assert split["n_test"] == 200 and split["n_test_failures"] == n_fail
    assert [int(y[split["shots"][k]].sum()) for k in ("5", "20")] == [2, 5]
    assert split["shots"]["5"] == [i for i in split["shots"]["20"] if i in set(split["shots"]["5"])]
    assert y[split["test"][:20]].min() == 0 or mode == "natural"


def test_split_depends_on_seed(df, cfg):
    assert typeE.make_split(df, {**cfg, "seed": 7})["test"] != typeE.make_split(df, cfg)["test"]


def test_check_split_catches_overlap(df, cfg):
    split = typeE.make_split(df, cfg)
    y = typeE.features(df)[1]
    bad = json.loads(json.dumps(split))
    bad["shots"]["20"][0] = split["test"][0]
    with pytest.raises(AssertionError):
        typeE.check_split(bad, y)


# ---------- Строка → текст ----------

PDF_ROW = {"Type": "L", "Air temperature": 300.1, "Process temperature": 310.5, "Rotational speed": 1551,
           "Torque": 42.8, "Tool wear": 108}


def test_row_text_matches_pdf_example():
    assert typeE.row_text(PDF_ROW, "ru") == (
        "Тип изделия: L. Температура воздуха 300,1 K, температура процесса 310,5 K, скорость вращения 1 551 об/мин, "
        "крутящий момент 42,8 Н·м, износ инструмента 108 мин.")
    assert typeE.row_text(PDF_ROW, "en") == (
        "Product type: L. Air temperature 300.1 K, process temperature 310.5 K, rotational speed 1,551 rpm, "
        "torque 42.8 N·m, tool wear 108 min.")


def test_row_text_small_numbers():
    row = {**PDF_ROW, "Rotational speed": 998, "Tool wear": 0, "Torque": 4.0}
    assert "скорость вращения 998 об/мин" in typeE.row_text(row, "ru")
    assert "крутящий момент 4,0 Н·м, износ инструмента 0 мин." in typeE.row_text(row, "ru")


def test_prompt_shots_and_labels(df):
    split = typeE.load_split()
    task = run_classifier.make_task(df, split["test"][0], split["shots"]["5"], "ru")
    _, prompt = prompts.build_messages(task, "classify_row", "ru")
    assert prompt.count("Исход: отказ.") == 2 and prompt.count("Исход: нет отказа.") == 3
    assert "AI4I" not in prompt and task["row_text"] in prompt
    _, zero = prompts.build_messages({"row_text": task["row_text"]}, "classify_row", "en")
    assert "Examples" not in zero and zero.endswith(prompts.FORMAT["en"]["label"])


# ---------- Классические модели ----------

def test_metrics_failure_is_positive_class():
    m = typeE.metrics(np.array([1, 1, 0, 0, 0]), np.array([1, 0, 0, 0, 1]))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["f1_failure"] == pytest.approx(0.5) and m["balanced_accuracy"] == pytest.approx((0.5 + 2 / 3) / 2)


def test_classical_table(df, cfg):
    split = typeE.load_split()
    t = typeE.classical(df, split, cfg["seed"])
    fixed = t[t["train_set"] == "fixed"]
    assert set(fixed["model"]) == {"Dummy(most_frequent)", "LogisticRegression", "GaussianNB"}
    assert len(fixed) == 3 * 2 * 2                                     # модели × (5, 20) × (test, natural)
    res = t[t["train_set"] == "resample"]
    assert len(res) == cfg["n_resamples"] * 2 * 2 * 2 and set(res["model"]) == {"LogisticRegression", "GaussianNB"}
    for col in ["f1_failure", "balanced_accuracy", "accuracy"]:
        assert t[col].between(0, 1).all()
    test_rows = t[t["eval_set"] == "test"]
    assert (test_rows["n_eval"] == 200).all() and (test_rows["n_eval_failures"] == 40).all()
    dummy = fixed[(fixed["model"] == "Dummy(most_frequent)") & (fixed["eval_set"] == "test")]
    assert (dummy["f1_failure"] == 0).all() and (dummy["balanced_accuracy"] == 0.5).all()
    full = t[t["eval_set"] == "full"].iloc[0]
    y = typeE.features(df)[1]
    assert full["accuracy"] == pytest.approx(1 - y.mean()) and full["n_eval"] == len(y)


# ---------- LLM-часть на mock ----------

def test_mock_answers_label():
    res = llm._call_mock({}, "mock-big", [{"role": "user", "content": prompts.FORMAT["ru"]["label"]}], 0, None, 0)
    assert parse.parse_answer(res["text"], "label", "ru").parse_ok


def test_run_classifier_demo(env):  # noqa: F811
    assert run_classifier.main(["--demo", "--limit", "5"]) == 0
    rows = run_bank.read_jsonl(env / "demo" / "results" / "typeE_llm.jsonl")
    assert len(rows) == 4 * 3 * 5
    assert len({run_classifier.row_key(r) for r in rows}) == len(rows)
    split = typeE.load_split()
    assert {r["row_idx"] for r in rows} == set(split["test"][:5]) and {r["shots"] for r in rows} == {0, 5, 20}
    assert all(r["parse_ok"] and r["value"] in (0, 1) and r["correct"] == (r["value"] == r["y_true"]) for r in rows)
    log = [json.loads(line) for line in (env / "demo" / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(log) == len(rows)
    meta = log[0]["meta"]
    assert meta["experiment"] == "typeE" and {"shots", "row_idx", "task_id", "repeat"} <= set(meta)
    by_call = {rec["call_id"]: rec for rec in log}
    r20 = next(r for r in rows if r["shots"] == 20)
    assert by_call[r20["call_id"]]["prompt"].count("Исход:") == 20

    # Повтор: всё из кэша, таблица не растёт.
    assert run_classifier.main(["--demo", "--limit", "5"]) == 0
    assert len(run_bank.read_jsonl(env / "demo" / "results" / "typeE_llm.jsonl")) == len(rows)
    assert len((env / "demo" / "calls.jsonl").read_text(encoding="utf-8").splitlines()) == len(log)
    assert {r["split_hash"] for r in rows} == {typeE.split_hash(split)}


def test_eval_problems_require_committed_split(monkeypatch):
    monkeypatch.setattr(run_classifier.checks, "last_commit", lambda path: None)
    assert any("e_split.json" in p for p in run_classifier.split_problems())


def test_run_classifier_dev_uses_pool_rows(env):  # noqa: F811
    assert run_classifier.main(["--demo", "--dev", "--models", "mock-big", "--shots", "0,5"]) == 0
    rows = run_bank.read_jsonl(env / "demo" / "results" / "dev_typeE_llm.jsonl")
    split = typeE.load_split()
    assert {r["row_idx"] for r in rows} == set(split["dev_rows"]) and {r["experiment"] for r in rows} == {"dev"}
    assert not set(split["dev_rows"]) & set(split["test"])


def test_real_run_blocked_without_hypotheses(env, capsys, monkeypatch):  # noqa: F811
    # Даже если гейт когда-нибудь откроется (гипотезы закоммичены, промпт заморожен), тест не должен звать API.
    monkeypatch.setattr(run_classifier, "eval_problems", lambda cfg: ["гипотезы не закоммичены (подмена в тесте)"])
    monkeypatch.setattr(llm, "ask", lambda *a, **k: pytest.fail("реальный вызов LLM из теста"))
    assert run_classifier.main(["--models", "all", "--limit", "2"]) == 1
    assert "не запущен" in capsys.readouterr().out
    assert not (env / "real_logs").exists()


def test_estimate_makes_no_calls(env, capsys):  # noqa: F811
    assert run_classifier.main(["--estimate", "--limit", "10"]) == 0
    out = capsys.readouterr().out
    assert "Вызовов:" in out and not (env / "demo").exists()
