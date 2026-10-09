"""Тесты src/analysis.py (функции analysis.ipynb) и src/demo_grades.py на синтетических данных."""

import json

import nbformat
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score

from src import analysis as an, autograde, blind, demo_grades, run_bank

MODELS = {
    "m-ru1": {"role": "ru1", "provider": "gigachat", "enabled": True, "model": "x1"},
    "m-ru2": {"role": "ru2", "provider": "openai_compat", "enabled": True, "model": "x2"},
    "m-big": {"role": "big", "provider": "openai_compat", "enabled": True, "model": "x3"},
    "m-loc": {"role": "local", "provider": "ollama", "enabled": True, "model": "q:8b"},
    "m-loc-think": {"role": "local", "provider": "ollama", "enabled": False, "model": "q:8b"},
    "mock-ru1": {"role": "ru1", "provider": "mock", "enabled": False, "model": "mock-ru1"},
}
PRICING = {"models": {"m-ru1": {"rub_per_1m_in": 1000.0, "rub_per_1m_out": 2000.0},
                      "m-loc": {"rub_per_1m_in": 0.0, "rub_per_1m_out": 0.0}}}


def row(model, task, typ, call, correct=None, value=0.5, kind="number", refused=False, conf=80.0, error=None,
        pair=None, variant_t=None, exp="main", **extra):
    return {"experiment": exp, "lang": "ru", "task_id": task, "type": typ, "answer_kind": kind, "pair_id": pair,
            "task_variant": variant_t, "variant": "base", "model_key": model, "model_reported": model, "repeat": 0,
            "call_id": call, "error": error, "tokens_in": 100, "tokens_out": 10, "latency_s": 1.0, "value": value,
            "confidence": conf, "refused": refused, "parse_ok": True, "correct": correct, **extra}


def graded_row(r):
    return {"check": "answer", "call_id": r["call_id"], "correct": r["correct"], "refused": r["refused"],
            "value": r["value"], "confidence": r["confidence"], "parse_ok": True}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


# ---------- Статистика ----------

def test_fast_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 50)
        p = rng.integers(0, 2, 50)
        assert an.balanced_acc(y, p) == pytest.approx(balanced_accuracy_score(y, p))
        assert an.f1_failure(y, p) == pytest.approx(f1_score(y, p, pos_label=1, zero_division=0))
    assert np.isnan(an.balanced_acc(np.array([0, 0]), np.array([0, 1])))
    assert an.f1_failure(np.array([0, 0]), np.array([0, 0])) == 0.0


def test_ece_known_example():
    # Корзина 80–100: уверенность 0,9, точность 0,5 → |0,4|; корзина 40–60: 0,5 и 0,5 → 0. ECE = 0,5·0,4 = 0,2.
    value, table = an.ece([90, 90, 50, 50], [1, 0, 1, 0])
    assert value == pytest.approx(0.2)
    assert table["n"].tolist() == [0, 0, 2, 0, 2]
    assert an.ece([100], [1], table=False) == pytest.approx(0.0)    # 100 попадает в последнюю корзину


def test_mcnemar_counts_and_exclusions():
    t = an.mcnemar_test([1, 1, 0, 0, 1, 0.5, np.nan], [1, 0, 1, 0, 0, 1, 1])
    assert (t["n"], t["both_correct"], t["only_first"], t["only_second"], t["both_wrong"]) == (5, 1, 2, 1, 1)
    # 6 расхождений в одну сторону → p = 2·0,5^6 = 0,03125 (справка по гипотезам)
    assert an.mcnemar_test([1] * 6, [0] * 6)["p_value"] == pytest.approx(0.03125)


def test_mcnemar_pairs_holm():
    df = pd.DataFrame([{"model_key": m, "task_id": f"T{i}", "correct": float(c)}
                       for m, cs in {"a": [1] * 8, "b": [0] * 8, "c": [1] * 4 + [0] * 4}.items()
                       for i, c in enumerate(cs)])
    t = an.mcnemar_pairs(df, ["a", "b", "c"])
    assert len(t) == 3
    assert (t["p_holm"] >= t["p_value"]).all()
    assert t.loc[0, "only_first"] == 8 and t.loc[0, "p_value"] == pytest.approx(2 * 0.5 ** 8)


def test_cluster_bootstrap_bounds_and_seed():
    df = pd.DataFrame({"task_id": list("abcdefghij"), "v": [1.0] * 7 + [0.0] * 3})
    lo, hi = an.cluster_bootstrap(df, "v", n_boot=1000, seed=42)
    assert 0 <= lo <= 0.7 <= hi <= 1
    assert (lo, hi) == an.cluster_bootstrap(df, "v", n_boot=1000, seed=42)
    assert an.cluster_bootstrap(df.assign(v=1.0), "v") == (1.0, 1.0)
    assert np.isnan(an.cluster_bootstrap(df.assign(v=np.nan), "v")[0])


def test_clopper_pearson_reference_values():
    # Справка по гипотезам: 6 из 12 → [0,21; 0,79], 10 из 48 → [0,10; 0,35].
    assert [round(x, 2) for x in an.clopper_pearson(6, 12)] == [0.21, 0.79]
    assert [round(x, 2) for x in an.clopper_pearson(10, 48)] == [0.10, 0.35]


# ---------- Ручные оценки ----------

def make_grading(tmp_path, items):
    """items: (item_id, type, call_id, required, r1: dict, r2: dict)."""
    g = tmp_path / "grading"
    key, s1, s2 = [], [], []
    for item_id, t, call, required, r1, r2 in items:
        key.append({"item_id": item_id, "type": t, "task_id": f"{t}1", "model_key": "m-ru1", "model_reported": "",
                    "experiment": "main", "lang": "ru", "variant": "base", "call_id": call,
                    "required": ";".join(required), "answer_sha": "x"})
        for sheet, grades in [(s1, r1), (s2, r2)]:
            base = {c: "" for c in blind.SHEET_COLUMNS}
            base.update({c: blind.NA for c in blind.GRADE_COLUMNS})
            base.update({c: "" for c in required})
            base.update({"item_id": item_id, "type": t, **grades})
            sheet.append(base)
    autograde.write_csv(g / blind.KEY_NAME, key, blind.KEY_COLUMNS)
    autograde.write_csv(g / blind.SHEETS[0], s1, blind.SHEET_COLUMNS)
    autograde.write_csv(g / blind.SHEETS[1], s2, blind.SHEET_COLUMNS)
    return g


D_ALL = [*blind.CHECKLIST, blind.CORRECT_COL]
CHECK0 = {c: "0" for c in blind.CHECKLIST}


def test_load_manual_scores_and_rules(tmp_path):
    g = make_grading(tmp_path, [
        ("B001", "B", "cb", ["score_0_2"], {"score_0_2": "2"}, {"score_0_2": "1,0"}),
        ("C001", "C", "cc", ["score_0_2"], {"score_0_2": "1"}, {"score_0_2": "0"}),
        # D с ручной корректностью: r1 «верно», r2 «неверно»; у r2 ошибка d6.
        ("D001", "D", "cd1", D_ALL, {**CHECK0, blind.CORRECT_COL: "1"},
         {**CHECK0, "d6_no_random_state": "1", blind.CORRECT_COL: "0"}),
        # D, где автопроверка сказала «верно» (корректность людьми не ставится) → 1; кода нет: всё «н/п».
        ("D002", "D", "cd2", list(blind.CHECKLIST), {c: "Н/П" for c in blind.CHECKLIST},
         {c: "н/п" for c in blind.CHECKLIST}),
    ])
    m = an.load_manual(g).set_index("item_id")
    assert m.loc["B001", "score_mean"] == 1.5
    assert m.loc["C001", "correct_manual"] == 0.5          # балл ≥ 1 у одного из двух
    assert m.loc["D001", "correct_manual"] == 0.5
    assert m.loc["D001", "any_error_mean"] == 0.5
    assert m.loc["D001", "d6_no_random_state_mean"] == 0.5
    assert m.loc["D002", "correct_manual"] == 1.0
    assert np.isnan(m.loc["D002", "any_error_mean"])        # кода нет — в долю ошибок не входит
    assert m["complete"].all()

    dis = an.disagreements(m.reset_index())
    assert set(zip(dis["item_id"], dis["column"])) == {("B001", "score_0_2"), ("C001", "score_0_2"),
                                                       ("D001", "d6_no_random_state"), ("D001", blind.CORRECT_COL)}
    k = an.kappa_table(m.reset_index()).set_index(["type", "what"])
    assert k.loc[("B", "балл 0–2"), "n_pairs"] == 1
    assert k.loc[("D", "корректность метрики (ручная)"), "n_pairs"] == 1


def test_kappa_flags_low_agreement(tmp_path):
    items = [(f"C{i:03d}", "C", f"c{i}", ["score_0_2"], {"score_0_2": a}, {"score_0_2": b})
             for i, (a, b) in enumerate([("0", "2"), ("2", "0"), ("1", "1"), ("2", "1"), ("0", "1"), ("1", "2")])]
    k = an.kappa_table(an.load_manual(make_grading(tmp_path, items)))
    c = k[k["type"] == "C"].iloc[0]
    assert c["kappa"] < an.KAPPA_MIN and c["below_min"]


def test_load_manual_missing_files(tmp_path):
    assert an.load_manual(tmp_path).empty


def test_incomplete_sheet_leaves_answer_ungraded(tmp_path):
    g = make_grading(tmp_path, [("C001", "C", "cc", ["score_0_2"], {"score_0_2": "2"}, {"score_0_2": ""})])
    m = an.load_manual(g)
    assert not m["complete"].iloc[0]
    res = tmp_path / "results"
    write_jsonl(res / "main_ru.jsonl", [row("m-ru1", "C1", "C", "cc", kind="text", value="x")])
    write_jsonl(res / "graded_auto.jsonl", [])
    df, _ = an.answers(res, "main", "ru", m)
    assert np.isnan(df["correct"].iloc[0])


# ---------- Ответы, точность, отказы ----------

@pytest.fixture
def bank40(tmp_path):
    """Вопрос самоконтроля 9: 40 задач, на 8 отказ, на 32 верно. Плюс техошибка и задача «нельзя определить»."""
    rows = [row("m-ru1", f"A{i}", "A", f"c{i}", correct=True) for i in range(32)]
    rows += [row("m-ru1", f"A{i}", "A", f"c{i}", correct=False, value=None, refused=True) for i in range(32, 40)]
    rows += [row("m-ru1", "F6", "F", "cf", correct=True, value="cannot_determine", refused=True,
                 kind="cannot_determine")]
    rows += [row("m-ru1", "A99", "A", "cerr", error="429")]
    res = tmp_path / "results"
    write_jsonl(res / "main_ru.jsonl", rows)
    write_jsonl(res / "graded_auto.jsonl", [graded_row(r) for r in rows if r["error"] is None])
    return res


def test_answers_and_two_accuracies(bank40):
    df, n_err = an.answers(bank40, "main", "ru", pd.DataFrame())
    assert n_err == 1 and len(df) == 41
    f6 = df[df["task_id"] == "F6"].iloc[0]
    assert f6["refusal"] and not f6["abstain"] and f6["correct"] == 1     # отказ здесь — правильный ответ
    a = an.acc_table(df[df["type"] == "A"], ["model_key"], 200, 42).iloc[0]
    assert a["accuracy"] == pytest.approx(0.8)              # отказ = ошибка: 32 / 40
    assert a["accuracy_answered"] == pytest.approx(1.0)     # только ответившие: 32 / 32
    assert a["refusal_rate"] == pytest.approx(0.2) and a["n_abstain"] == 8


def test_answers_regraded_value_wins(tmp_path):
    """correct берётся из graded_auto (повторный разбор с текущим эталоном), а не из таблицы прогона."""
    r = row("m-ru1", "A1", "A", "c1", correct=False)
    res = tmp_path / "results"
    write_jsonl(res / "main_ru.jsonl", [r])
    write_jsonl(res / "graded_auto.jsonl", [{**graded_row(r), "correct": True}])
    df, _ = an.answers(res, "main", "ru", pd.DataFrame())
    assert df["correct"].iloc[0] == 1.0


def test_answers_d_uses_manual_and_exec(tmp_path):
    g = make_grading(tmp_path, [("D001", "D", "cd", D_ALL, {**CHECK0, blind.CORRECT_COL: "1"},
                                 {**CHECK0, blind.CORRECT_COL: "1"})])
    res = tmp_path / "results"
    r = row("m-ru1", "D1", "D", "cd", kind="code", value="print(1)")
    write_jsonl(res / "main_ru.jsonl", [r])
    write_jsonl(res / "graded_auto.jsonl", [{"check": "d_exec", "call_id": "cd", "pass_at_1": False,
                                             "correct": False}])
    df, _ = an.answers(res, "main", "ru", an.load_manual(g))
    t = an.typeD_table(df)
    assert df["correct"].iloc[0] == 1.0       # итог по людям, не по подсказке автопроверки
    assert t.iloc[0]["pass_at_1"] == 0.0 and t.iloc[0]["method_error_any"] == 0.0


def test_answers_missing_table(tmp_path):
    df, n = an.answers(tmp_path, "main", "ru", pd.DataFrame())
    assert df.empty and n == 0


# ---------- Гипотезы ----------

def test_h1_pairs_and_comparison():
    df = pd.DataFrame([
        {"model_key": m, "task_id": t, "pair_id": p, "task_variant": v, "correct": c}
        for m in ["a", "b"]
        for t, p, v, c in [("A1", "P1", "original", 1.0), ("F1", "P1", "modified", 0.0),
                           ("A2", "P2", "original", 1.0), ("F2", "P2", "modified", 1.0)]])
    pairs = an.h1_pairs(df)
    assert len(pairs) == 4
    t = an.compare_conditions(pairs, "model_key", 200, 42, cluster="pair_id").set_index("model_key")
    assert t.loc["все", "acc_a"] == 1.0 and t.loc["все", "acc_b"] == 0.5
    assert t.loc["все", "only_a_correct"] == 2 and t.loc["все", "only_b_correct"] == 0


def test_h3_invented_vs_refused():
    df = pd.DataFrame([
        {"model_key": "a", "task_id": "F6", "answer_kind": "cannot_determine", "pair_id": "P6", "value": 0.3,
         "refusal": False},
        {"model_key": "a", "task_id": "F7", "answer_kind": "cannot_determine", "pair_id": "P7",
         "value": "cannot_determine", "refusal": True},
        {"model_key": "a", "task_id": "F8", "answer_kind": "cannot_determine", "pair_id": "P8", "value": np.nan,
         "refusal": False},
        {"model_key": "a", "task_id": "A6", "answer_kind": "number", "pair_id": "P6", "value": 0.1, "refusal": False},
    ])
    t = an.h3_table(df).set_index("model_key")
    assert t.loc["a", "n_invented_number"] == 1                 # NaN (не разобрано) — не «додумала»
    assert t.loc["a", "n_cannot_determine_or_refused"] == 1 and t.loc["a", "n_other_unparsed"] == 1
    assert t.loc["a", "control_n"] == 1 and t.loc["a", "control_refusal_rate"] == 0


def test_h9_flip_rate():
    main = pd.DataFrame([{"call_id": f"m{i}", "correct": c} for i, c in enumerate([1.0, 1.0, 1.0, 0.0])])
    follow = pd.DataFrame([{"model_key": "a", "first_call_id": f"m{i}", "correct": c, "refusal": False,
                            "parse_ok_auto": True} for i, c in enumerate([1.0, 0.0, 0.0, 1.0])])
    t = an.h9_table(main, follow).set_index("model_key")
    assert t.loc["a", "n_first_correct"] == 3 and t.loc["a", "n_flipped"] == 2


def test_mismatch_table():
    pairs = pd.DataFrame({"model_key": ["a"] * 4, "correct_a": [1.0, 1.0, 0.0, 0.5], "correct_b": [1.0, 0.0, 0.0, 1.0]})
    t = an.mismatch_table(pairs).set_index("model_key")
    assert t.loc["a", "n_pairs"] == 3 and t.loc["a", "n_mismatch"] == 1


def test_h6_think_model_compared_with_base():
    main = pd.DataFrame([{"model_key": "m-loc", "task_id": "A1", "type": "A", "correct": 0.0, "tokens_in": 100,
                          "tokens_out": 10, "latency_s": 1.0, "call_id": "x"}])
    cot = pd.DataFrame([{"model_key": "m-loc-think", "task_id": "A1", "type": "A", "correct": 1.0, "tokens_in": 100,
                         "tokens_out": 50, "latency_s": 4.0, "call_id": "y"}])
    acc, tok = an.h6_tables(main, cot, MODELS, PRICING, 100, 42)
    assert acc.set_index("group").loc["A: режим рассуждений (m-loc-think)", "only_cot_correct"] == 1
    assert tok.iloc[0]["main_model"] == "m-loc" and tok.iloc[0]["tokens_out_ratio"] == 5.0


def test_h12_against_logreg():
    preds = pd.DataFrame({"model_key": "a", "shots": 5, "row_idx": [0, 1, 2, 3], "y_true": [1, 1, 0, 0],
                          "pred": [1, 1, 0, 0]})
    logreg = {5: {0: 0, 1: 1, 2: 0, 3: 1}}
    t = an.h12_table(preds, logreg, 200, 42).iloc[0]
    assert t["ba_llm"] == 1.0 and t["ba_logreg"] == 0.5 and t["ba_diff"] == 0.5
    assert t["only_llm_correct"] == 2 and t["only_logreg_correct"] == 0


# ---------- Цена ----------

def test_price_and_cost():
    assert an.price_for("mock-ru1", PRICING, MODELS) == (1000.0, 2000.0)    # mock берёт цену модели своей роли
    assert an.price_for("m-big", PRICING, MODELS) is None
    df = pd.DataFrame([{"model_key": "m-ru1", "task_id": f"T{i}", "correct": c, "tokens_in": 1000,
                        "tokens_out": 500, "latency_s": 2.0} for i, c in enumerate([1.0, 0.0])])
    t = an.cost_table(df, PRICING, MODELS, 100, 42).iloc[0]
    # одна задача: 1000·1000/1e6 + 500·2000/1e6 = 2 ₽; две задачи — 4 ₽, верный ответ один → 4 ₽ за верный
    assert t["cost_rub_total"] == pytest.approx(4.0) and t["rub_per_correct"] == pytest.approx(4.0)


# ---------- Экспорт, демо-оценки, ноутбук ----------

def test_write_summary_cleans_nan(tmp_path):
    path = tmp_path / "s.json"
    an.write_summary({"a": np.float64(np.nan), "b": pd.DataFrame({"x": [1.5, np.nan]}), "c": np.int64(3),
                      "d": (np.bool_(True),)}, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {"a": None, "b": [{"x": 1.5}, {"x": None}], "c": 3, "d": [True]}


def test_demo_grades_only_in_demo(tmp_path):
    with pytest.raises(ValueError):
        demo_grades.fill(tmp_path)


def test_demo_grades_fill(tmp_path, monkeypatch):
    monkeypatch.setattr(run_bank, "DEMO_DIR", tmp_path)
    g = make_grading(tmp_path, [("C001", "C", "cc", ["score_0_2"], {}, {}),
                                ("D001", "D", "cd", D_ALL, {}, {})])
    assert demo_grades.fill(g, seed=1) == 0
    m = an.load_manual(g)
    assert m["complete"].all()


def test_notebook_committed_clean():
    """В репозитории ноутбук лежит с DEMO = False; выходы — только у выполненного на реальных данных."""
    nb = nbformat.read(an.ROOT / "analysis.ipynb", as_version=4)
    first = next(c for c in nb.cells if c.cell_type == "code")
    assert first.source.startswith("DEMO = False")
    # Места под выводы человека (H1–H13; неиспользованные гипотезы человек может удалить, выбранных — 3).
    conclusions = [c for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("**Вывод (заполняет человек):**")]
    assert len(conclusions) >= 3


# ---------- Случаи из ревью ----------

def test_partially_filled_sheet_pandas3(tmp_path):
    """Столбец с заполненными и пустыми клетками: pandas 3 не должен превращать пустую клетку в «заполненную» NaN."""
    g = make_grading(tmp_path, [("C001", "C", "c1", ["score_0_2"], {"score_0_2": "2"}, {"score_0_2": "2"}),
                                ("C002", "C", "c2", ["score_0_2"], {"score_0_2": "1"}, {"score_0_2": ""})])
    m = an.load_manual(g).set_index("item_id")
    assert m.loc["C001", "complete"] and not m.loc["C002", "complete"]
    assert an.disagreements(m.reset_index()).empty
    assert an.kappa_table(m.reset_index()).set_index("type").loc["C", "n_pairs"] == 1


def test_h6_think_is_separate_group():
    main = pd.DataFrame([{"model_key": "m-loc", "task_id": f"A{i}", "type": "A", "correct": 0.0, "tokens_in": 10,
                          "tokens_out": 1, "latency_s": 1.0, "call_id": f"x{i}"} for i in range(3)])
    cot = pd.DataFrame([{"model_key": k, "task_id": f"A{i}", "type": "A", "correct": 1.0, "tokens_in": 10,
                         "tokens_out": 5, "latency_s": 1.0, "call_id": f"{k}{i}"}
                        for k in ["m-loc", "m-loc-think"] for i in range(3)])
    acc, _ = an.h6_tables(main, cot, MODELS, PRICING, 100, 42)
    assert sorted(acc["group"]) == ["A: CoT-промпт", "A: режим рассуждений (m-loc-think)"]
    assert (acc["n_pairs"] == 3).all()          # ответы main не удвоены


def test_h7_uses_regraded_and_ignores_tech_errors():
    main = pd.DataFrame([{"model_key": "a", "task_id": "A1", "correct": 1.0, "call_id": "m1"}])
    votes = pd.DataFrame([{"model_key": "a", "task_id": "A1", "correct": False, "majority_call_id": "v1",
                           "n_groups": 1, "n_voted": 4, "n_repeats": 5, "tie": False}])
    repeats = pd.DataFrame([{"model_key": "a", "task_id": "A1", "error": None if i < 4 else "429"} for i in range(5)])
    graded = pd.DataFrame([{"check": "answer", "call_id": "v1", "correct": True}])
    acc, spread = an.h7_tables(main, votes, repeats, graded, 100, 42)
    assert acc.set_index("model_key").loc["a", "acc_vote"] == 1.0
    assert spread.set_index("model_key").loc["a", "n_not_unanimous"] == 0


def test_cost_ci_keeps_infinite_upper_bound():
    df = pd.DataFrame([{"model_key": "m-ru1", "task_id": f"T{i}", "correct": 1.0 if i < 2 else 0.0, "tokens_in": 1000,
                        "tokens_out": 0, "latency_s": 1.0} for i in range(20)])
    t = an.cost_table(df, PRICING, MODELS, 1000, 42).iloc[0]
    assert t["rub_per_correct_ci_high"] == np.inf and np.isfinite(t["rub_per_correct_ci_low"])


def test_answers_value_fallback_without_autograde(tmp_path):
    """Задача без эталона (нет строки в graded_auto): значение и отказ берутся из разбора при прогоне."""
    res = tmp_path / "results"
    write_jsonl(res / "main_ru.jsonl", [row("m-ru1", "F6", "F", "c1", value=0.3, kind="cannot_determine")])
    write_jsonl(res / "graded_auto.jsonl", [])
    df, _ = an.answers(res, "main", "ru", pd.DataFrame())
    assert df["value"].iloc[0] == 0.3 and not df["has_auto"].iloc[0] and np.isnan(df["correct"].iloc[0])
