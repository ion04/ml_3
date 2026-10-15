"""Тесты src/build_report.py и src/error_candidates.py (шаг 12) на синтетических данных, без реальных results/."""

import json

import pandas as pd
import pytest

from src import build_report as br
from src import error_candidates as ec


@pytest.fixture
def paths(tmp_path, monkeypatch):
    """report_paths → tmp: отчёт собирается без summary/manifest, если тест их не создал."""
    p = {"report": tmp_path / "report" / "report.md", "figures": tmp_path / "report" / "figures",
         "summary": tmp_path / "results" / "summary_metrics.json",
         "manifest": tmp_path / "results" / "run_manifest.json",
         "errors": tmp_path / "report" / "error_candidates.md"}
    monkeypatch.setattr(br, "report_paths", lambda demo: p)
    cfg = br.run_bank.load_experiment() | {"hypotheses": []}   # не зависеть от выбора гипотез человеком (G1)
    monkeypatch.setattr(br.run_bank, "load_experiment", lambda: cfg)
    return p


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


# ---------- build_report ----------

def test_fmt_and_table():
    assert br.fmt(None) == "—" and br.fmt(float("nan")) == "—" and br.fmt(float("inf")) == "∞"
    assert br.fmt(True) == "да" and br.fmt(3) == "3" and br.fmt(0.12345) == "0.123" and br.fmt(0.00012) == "1.2e-04"
    assert br.fmt([0.5, 1]) == "0.500; 1" and br.fmt("a|b") == "a\\|b"
    assert br.md_table([]) == br.NO_DATA
    t = br.md_table([{"a": 1, "b": 0.5}], [("a", "A"), "b"])
    assert t.splitlines() == ["| A | b |", "|---|---|", "| 1 | 0.500 |"]
    assert br.ci({"accuracy": 0.75, "ci_low": 0.5, "ci_high": 0.9}, "accuracy") == "0.750 [0.500; 0.900]"


def test_fence_longer_than_backticks_inside():
    text = "```python\nprint(1)\n```"
    out = br.fence(text)
    assert out.startswith("````text\n") and out.endswith("\n````") and text in out


def test_no_data_report(paths):
    text, blocks = br.build(demo=False)
    assert "Нет данных: `results/summary_metrics.json` ещё не создан" in text
    assert "нет данных" in text and "Гипотезы не выбраны" in text
    ids = [b for b, _ in blocks]
    for need in ["intro", "limitations", "recommendations", "conclusion", "error.1", "error.5"]:
        assert need in ids
    assert not any(filled for _, filled in blocks)
    # Полный текст промпта и таблица «тип × тема» попадают в методику.
    from src import prompts
    assert prompts.SYSTEM["ru"] in text and "**Тип × тема**" in text


def test_human_text_survives_rebuild_and_orphans_kept(paths):
    text, _ = br.build(demo=False)
    filled = text.replace("\n<!-- /ЧЕЛОВЕК:intro -->", "\nМой текст введения.\nВторая строка.\n<!-- /ЧЕЛОВЕК:intro -->", 1)
    filled += "\n<!-- ЧЕЛОВЕК:verdict.H99 — старый блок -->\nВердикт по старой гипотезе.\n<!-- /ЧЕЛОВЕК:verdict.H99 -->\n"
    text2, blocks = br.build(demo=False, old_text=filled)
    assert "Мой текст введения.\nВторая строка." in text2
    assert dict(blocks)["intro"] is True and dict(blocks)["conclusion"] is False
    assert "Сохранённые блоки человека" in text2 and "Вердикт по старой гипотезе." in text2
    # Пересборка уже пересобранного текста ничего не меняет (детерминированность).
    assert br.build(demo=False, old_text=text2)[0] == text2


def test_recommendations_default_table_is_unfilled(paths):
    text, blocks = br.build(demo=False)
    assert "| A — расчётные |  |  |" in text
    assert dict(blocks)["recommendations"] is False


def test_numbers_come_from_summary(paths):
    summary = {"demo": False, "seed": 42, "bootstrap_n": 1000,
               "accuracy_by_model": [{"model_key": "m1", "n": 38, "n_graded": 38, "accuracy": 0.5,
                                      "ci_low": 0.25, "ci_high": 0.75, "refusal_rate": 0.0,
                                      "accuracy_answered": 0.5}],
               "hypotheses": {"H11": {"n_programs": 24, "mcnemar_p": 0.0039}}}
    write_json(paths["summary"], summary)
    text, _ = br.build(demo=False)
    assert "| m1 | 38 | 38 | 0.500 [0.250; 0.750] | 0.000 | 0.500 |" in text
    assert "#### H11." in text and "- `mcnemar_p`: 0.0039" in text and "разведочный анализ" in text
    assert "Нет данных: `results/summary_metrics.json`" not in text


def test_demo_summary_rejected_in_real_mode(paths):
    write_json(paths["summary"], {"demo": True})
    with pytest.raises(ValueError):
        br.build(demo=False)


def test_figure_link_only_if_file_exists(tmp_path):
    assert "нет данных" in br.figure("x", tmp_path, "подпись")
    (tmp_path / "x.png").write_bytes(b"png")
    assert "![подпись](figures/x.png)" in br.figure("x", tmp_path, "подпись")


def test_check_command(paths):
    assert br.main(["check"]) == 1                    # отчёта ещё нет
    assert br.main([]) == 0 and paths["report"].exists()
    assert br.main(["check"]) == 1                    # блоки пустые


def test_block_with_spaces_and_same_line_text(paths):
    text, _ = br.build(demo=False)
    edited = text.replace("\n\n<!-- /ЧЕЛОВЕК:conclusion -->", "  \nИтог.   \n  <!-- /ЧЕЛОВЕК:conclusion  -->", 1)
    assert "Итог." in br.build(demo=False, old_text=edited)[0]


def test_broken_block_not_overwritten(paths):
    assert br.main([]) == 0
    text = paths["report"].read_text(encoding="utf-8")
    broken = text.replace("<!-- /ЧЕЛОВЕК:intro -->", "<!-- /ЧЕЛОВЕК:intr -->", 1).replace(
        "\n<!-- /ЧЕЛОВЕК:intr", "\nЦенный текст\n<!-- /ЧЕЛОВЕК:intr", 1)
    paths["report"].write_text(broken, encoding="utf-8")
    assert br.main([]) == 1
    assert paths["report"].read_text(encoding="utf-8") == broken          # не перезаписан


def test_backup_and_p_values(paths):
    assert br.main([]) == 0
    write_json(paths["summary"], {"demo": False, "mcnemar_models": [
        {"model_a": "a", "model_b": "b", "n": 38.0, "only_first": 3.0, "only_second": 0.0, "p_value": 0.0496,
         "p_holm": 0.0504}]})
    assert br.main([]) == 0
    text = paths["report"].read_text(encoding="utf-8")
    assert "| a | b | 38 | 3 | 0 | — | — | 0.0496 | 0.0504 |" in text
    assert paths["report"].with_name("report.md.bak").exists()


# ---------- error_candidates ----------

def c(cat, model, task, prio=0.0, call="x"):
    return {"category": cat, "model_key": model, "task_id": task, "priority": prio, "call_id": call}


def test_select_quotas_round_robin_and_no_duplicates():
    pools = {"high_conf_wrong": [c("high_conf_wrong", "m1", f"A{i}", 99 - i) for i in range(6)]
             + [c("high_conf_wrong", "m2", "A1", 95)],
             "flip": [c("flip", "m1", "A0", 90), c("flip", "m2", "B1", 90)]}
    chosen = ec.select(pools)
    first3 = chosen[:3]
    assert {x["model_key"] for x in first3} == {"m1", "m2"}          # модели по кругу
    keys = [(x["model_key"], x["task_id"]) for x in chosen]
    assert len(keys) == len(set(keys))                               # (модель, задача) не повторяется
    assert ("m1", "A0") in keys and [x["category"] for x in chosen if x["task_id"] == "A0"] == ["high_conf_wrong"]
    assert sum(x["category"] == "flip" for x in chosen) == 1         # flip m1/A0 уже занят
    assert len(chosen) == 8                                          # все 9 минус дубль, меньше N_MAX


def test_select_limit():
    pools = {"high_conf_wrong": [c("high_conf_wrong", f"m{i % 4}", f"A{i}", 90) for i in range(40)]}
    assert len(ec.select(pools)) == ec.N_MAX


def test_same_answer_and_textbook():
    assert ec.same_answer(0.161, 0.161, 0.01) and not ec.same_answer(0.2, 0.161, 0.01)
    assert ec.same_answer([0.4, 1.0], [0.4, 1], 0.01) and not ec.same_answer(0.4, [0.4, 1], 0.01)
    tasks = {"A1": {"id": "A1", "variant": "original", "pair_id": "P1", "reference": 0.161, "tolerance": 0.01},
             "F1": {"id": "F1", "variant": "modified", "pair_id": "P1", "reference": 0.3, "tolerance": 0.01}}
    main = pd.DataFrame([
        {"task_id": "F1", "model_key": "m1", "task_variant": "modified", "pair_id": "P1", "value": 0.1612,
         "correct": 0.0, "confidence": 90.0, "call_id": "c1"},
        {"task_id": "F1", "model_key": "m2", "task_variant": "modified", "pair_id": "P1", "value": 0.25,
         "correct": 0.0, "confidence": 90.0, "call_id": "c2"}])
    out = ec.textbook_answers(main, tasks)
    assert [x["model_key"] for x in out] == ["m1"] and "A1" in out[0]["note"]


def test_high_conf_and_invented():
    main = pd.DataFrame([
        {"task_id": "A1", "type": "A", "model_key": "m1", "correct": 0.0, "confidence": 95.0, "value": 1.0,
         "answer_kind": "number", "call_id": "a"},
        {"task_id": "A2", "type": "A", "model_key": "m1", "correct": 0.0, "confidence": 80.0, "value": 1.0,
         "answer_kind": "number", "call_id": "b"},
        {"task_id": "A3", "type": "A", "model_key": "m1", "correct": 1.0, "confidence": 99.0, "value": 1.0,
         "answer_kind": "number", "call_id": "c"},
        {"task_id": "F6", "type": "F", "model_key": "m1", "correct": 0.0, "confidence": None, "value": 2.5,
         "answer_kind": "cannot_determine", "call_id": "d"},
        {"task_id": "F7", "type": "F", "model_key": "m1", "correct": 1.0, "confidence": 70.0,
         "value": "cannot_determine", "answer_kind": "cannot_determine", "call_id": "e"}])
    assert [x["call_id"] for x in ec.high_conf_wrong(main)] == ["a"]
    assert [x["call_id"] for x in ec.invented_numbers(main)] == ["d"]


def test_quote_exact_or_head_tail():
    short = "ОТВЕТ: 1\nУВЕРЕННОСТЬ: 90"
    assert short in ec.quote(short)
    long = "a" * ec.QUOTE_HEAD + "b" * 5000 + "c" * ec.QUOTE_TAIL
    q = ec.quote(long)
    assert "a" * ec.QUOTE_HEAD + "\n```" in q and "пропущено 5000 символов" in q
    assert q.rstrip("`\n").endswith("\n" + "c" * ec.QUOTE_TAIL) and "b" not in q.replace("Начало", "").replace("Конец", "")
    assert ec.quote(None) == "(в журнале нет текста ответа)"


def test_real_mode_needs_g2(monkeypatch, tmp_path):
    monkeypatch.setattr(ec, "g2_passed", lambda: False)
    monkeypatch.setattr(ec, "out_path", lambda demo: tmp_path / "error_candidates.md")
    assert ec.main([]) == 1
    assert not (tmp_path / "error_candidates.md").exists()


def test_bank_and_runs_come_from_summary(paths):
    # Состав банка и вызовы моделей отчёт берёт из summary (их считает analysis.ipynb), а не из data/ и манифеста.
    from src import analysis as an, bank
    tasks = bank.load_tasks(bank.BANK_PATH)
    runs = {"calls_by_experiment": {"main": {"records": 7, "ok": 6, "errors": 1, "unresolved_requests": 1}},
            "models": {"m1": {"model_requested": ["M-1"], "model_reported": ["M-1-2026"], "ok": 6, "errors": 1,
                              "first_utc": "2026-11-01T10:00:00", "last_utc": "2026-11-02T10:00:00"}}}
    summary = an.clean({"demo": False, "bank": an.bank_summary(tasks), "runs": runs})
    write_json(paths["summary"], summary)
    text, _ = br.build(demo=False)
    assert f"Задач в банке: {len(tasks)}" in text and "| type | bayes |" in text
    assert "M-1-2026" in text and "2026-11-01 … 2026-11-02" in text and "| main |" in text
    assert "запросы без ответа после повторов: {'main': 1}" in text
    write_json(paths["summary"], {"demo": False})
    assert "Задач в банке" not in br.build(demo=False)[0]


def test_coverage_table_and_incomplete_runs():
    cov = {"typeE": [{"model_key": "m1", "n_expected": 600, "n_got": 600, "complete": True},
                     {"model_key": "m2", "n_expected": 600, "n_got": 0, "complete": False}],
           "cot": [{"model_key": "m3", "n_expected": 16, "n_got": 16, "complete": True}]}
    lines = br.coverage_table(cov).splitlines()
    assert lines[0] == "| модель | тип E | H6, cot |"
    assert lines[3] == "| m2 | 0 / 600 **неполный** | — |" and lines[4] == "| m3 | — | 16 / 16 |"
    assert br.incomplete_runs(cov) == {"typeE": ["m2"]}
    assert br.coverage_table({}) == br.NO_DATA and br.incomplete_runs(None) == {}


def test_yandex_folder_masked_and_criterion_period(paths):
    write_json(paths["manifest"], {"prompt": {"version_config": "v1", "hash_config": "x", "version_code": "v1",
                                              "hash_code": "x"}})
    write_json(paths["summary"], {"demo": False, "runs": {"models": {"yandexgpt-pro-5.1": {
        "model_requested": "gpt://b1gabc123/yandexgpt-5.1", "model_reported": "gpt://b1gabc123/yandexgpt-5.1"}}}})
    text, _ = br.build(demo=False)
    assert "b1gabc123" not in text and "gpt://<YANDEX_FOLDER_ID>/yandexgpt-5.1" in text
    block = br.hypothesis_block(br.Human(""), "H1", None, {"Прогноз": "x", "Критерий решения (тест, порог)": "y."},
                                paths["figures"], False)
    assert "Критерий: y." in block and "y.." not in block
