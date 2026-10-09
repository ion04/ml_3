"""Тесты прогонщиков (run_bank, run_extra, exec_code) end-to-end на mock через --demo.

Сеть не используется. demo/ подменяется временной папкой; «настоящие» logs/ и results/ тоже подменяются
временными путями, чтобы ошибка в коде не могла дописать что-то в журнал эксперимента.
Дополнительно проверяется, что реальные logs/ и results/ проекта не изменились.
"""

import json
import os
from pathlib import Path

import pytest

from src import exec_code, llm, run_bank, run_extra

FIXTURE_BANK = Path(__file__).parent / "fixtures" / "bank_small.jsonl"
N_TASKS = 8          # в фикстуре 9 задач, одна — заготовка todo
N_MODELS = 4         # mock-ru1, mock-ru2, mock-big, mock-local (по ролям включённых моделей)


def snapshot(path: Path) -> dict:
    """Имена, размеры и время изменения всех файлов каталога."""
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in path.rglob("*") if p.is_file()}


@pytest.fixture
def env(tmp_path, monkeypatch):
    real_dirs = [llm.ROOT / "logs", llm.ROOT / "results"]
    before = [snapshot(d) for d in real_dirs]

    cfg = json.loads((llm.ROOT / "config" / "experiment.json").read_text(encoding="utf-8"))
    cfg["hypotheses"] = ["H1", "H6", "H9"]
    cfg_path = tmp_path / "experiment.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr(run_bank, "DEMO_DIR", tmp_path / "demo")
    monkeypatch.setattr(run_bank, "RESULTS_DIR", tmp_path / "real_results")
    monkeypatch.setattr(run_bank, "EXPERIMENT_PATH", cfg_path)
    monkeypatch.setattr(llm, "REAL_LOG", tmp_path / "real_logs" / "calls.jsonl")
    monkeypatch.setattr(llm, "_cache_index", {})
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    yield tmp_path
    assert [snapshot(d) for d in real_dirs] == before, "demo-прогон изменил logs/ или results/ проекта"
    assert not (tmp_path / "real_results").exists() and not (tmp_path / "real_logs").exists()


def demo_log(tmp):
    return [json.loads(line) for line in (tmp / "demo" / "calls.jsonl").read_text(encoding="utf-8").splitlines()]


def results(tmp, name):
    return run_bank.read_jsonl(tmp / "demo" / "results" / f"{name}.jsonl")


def bank_args(*extra):
    return ["--demo", "--bank", str(FIXTURE_BANK), *extra]


# ---------- run_bank ----------

def test_run_bank_demo_end_to_end(env):
    assert run_bank.main(["--experiment", "main", "--lang", "ru", *bank_args()]) == 0
    rows = results(env, "main_ru")
    assert len(rows) == N_TASKS * N_MODELS
    assert {r["model_key"] for r in rows} == {"mock-ru1", "mock-ru2", "mock-big", "mock-local"}
    assert "F3" not in {r["task_id"] for r in rows}          # заготовка todo пропущена
    r = next(r for r in rows if r["task_id"] == "B1")
    for field in ["call_id", "value", "confidence", "refused", "parse_ok", "parse_notes", "correct", "score",
                  "reason", "response_text", "tokens_in", "latency_s", "prompt_version", "task_variant"]:
        assert field in r, field
    assert r["value"] in (True, False) and r["parse_ok"]      # mock отвечает вердиктом на вопрос B
    assert next(r for r in rows if r["task_id"] == "C1")["correct"] is None   # C проверяют люди

    log = demo_log(env)
    assert len(log) == len(rows)
    assert {rec["call_id"] for rec in log} == {r["call_id"] for r in rows}
    meta = log[0]["meta"]
    assert set(meta) >= {"task_id", "type", "variant", "experiment", "lang", "prompt_version", "repeat"}
    assert meta["experiment"] == "main" and meta["variant"] == "base"


def test_run_bank_is_idempotent(env, capsys):
    run_bank.main(bank_args("--lang", "ru"))
    first_log = (env / "demo" / "calls.jsonl").read_text(encoding="utf-8")
    first_rows = results(env, "main_ru")
    capsys.readouterr()

    assert run_bank.main(bank_args("--lang", "ru")) == 0
    assert (env / "demo" / "calls.jsonl").read_text(encoding="utf-8") == first_log   # новых вызовов нет
    assert results(env, "main_ru") == first_rows
    assert "новых: 0" in capsys.readouterr().out


def test_partial_runs_keep_previous_rows(env):
    run_bank.main(bank_args("--types", "A"))
    assert {r["type"] for r in results(env, "main_ru")} == {"A"}
    run_bank.main(bank_args("--types", "B", "--limit", "3"))      # из задач типа B берутся первые 3 (их 2)
    assert {r["type"] for r in results(env, "main_ru")} == {"A", "B"}
    assert len(results(env, "main_ru")) == 4 * N_MODELS


def test_run_bank_models_subset_and_lang_en(env):
    run_bank.main(bank_args("--models", "mock-big", "--lang", "en"))
    rows = results(env, "main_en")
    assert {r["model_key"] for r in rows} == {"mock-big"} and len(rows) == N_TASKS
    assert all(rec["meta"]["lang"] == "en" and "ANSWER" in rec["system"] for rec in demo_log(env))


def test_eval_run_refused_without_gate(env, capsys):
    # Не demo: банк фикстуры не финальный, промпт не заморожен — вызовов быть не должно.
    assert run_bank.main(["--experiment", "main", "--bank", str(FIXTURE_BANK)]) == 1
    assert "Прогон не запущен" in capsys.readouterr().out
    assert not llm.REAL_LOG.exists()
    assert run_extra.main(["cot", "--bank", str(FIXTURE_BANK)]) == 1
    assert not llm.REAL_LOG.exists()


def test_eval_problems_require_prompt_hash(env):
    cfg = json.loads(run_bank.EXPERIMENT_PATH.read_text(encoding="utf-8"))
    cfg["prompt_version"] = run_bank.prompts.PROMPT_VERSION      # версия есть, отпечатка нет
    run_bank.EXPERIMENT_PATH.write_text(json.dumps(cfg), encoding="utf-8")
    problems = run_bank.eval_problems("main", FIXTURE_BANK)
    assert any("prompt_hash" in p for p in problems)
    assert any("hypotheses.md" in p for p in problems)            # шаблон гипотез не заполнен


def test_dev_experiment_only_on_dev_bank():
    assert run_bank.eval_problems("dev", run_bank.bank.BANK_PATH)
    assert run_bank.eval_problems("dev", run_bank.bank.DEV_PATH) == []
    assert run_bank.eval_problems("main", run_bank.bank.DEV_PATH)


def test_select_models(monkeypatch, capsys):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert run_bank.select_models("deepseek-v4-pro", demo=False) == []
    assert "DEEPSEEK_API_KEY" in capsys.readouterr().out
    assert run_bank.select_models("gigachat-3-ultra", demo=False) == []           # enabled=false
    assert "enabled=false" in capsys.readouterr().out
    assert run_bank.select_models("mock-big", demo=False) == []                   # mock без --demo
    assert run_bank.select_models("all", demo=True) == ["mock-ru1", "mock-ru2", "mock-big", "mock-local"]
    assert run_bank.select_models("qwen3-8b,qwen3-8b-think", demo=True) == ["mock-local"]
    assert run_bank.select_models("nonexistent", demo=True) == []
    assert run_bank.select_models("qwen3-8b,qwen3-8b", demo=False) in ([], ["qwen3-8b"])   # без дублей


def test_parallel_log_is_consistent(env):
    # 4 потока mock пишут в один журнал: каждая строка — целая запись JSON, ни одна не потеряна.
    run_extra.main(["selfconsistency", *bank_args()])
    log = demo_log(env)
    assert len(log) == len({rec["call_id"] for rec in log}) == len(results(env, "selfconsistency_ru"))


# ---------- run_extra ----------

def test_run_extra_selected_modes(env, capsys):
    run_bank.main(bank_args())
    capsys.readouterr()
    assert run_extra.main(bank_args()) == 0                 # гипотезы H1, H6, H9 → cot и followup_doubt
    out = capsys.readouterr().out
    assert "H1: отдельный прогон не нужен" in out
    assert (env / "demo" / "results" / "cot_ru.jsonl").exists()
    assert (env / "demo" / "results" / "followup_doubt_ru.jsonl").exists()
    assert not (env / "demo" / "results" / "paraphrase_ru.jsonl").exists()


def test_modes_for():
    assert run_extra.modes_for(["H1", "H6", "H9"]) == ["cot", "followup_doubt"]
    assert run_extra.modes_for(["H5", "H10", "H12"]) == []
    assert run_extra.modes_for(["H4", "H9"]) == ["lang_en", "followup_doubt"]   # lang_en раньше followup


def test_run_extra_all_modes(env):
    run_bank.main(bank_args())
    assert run_extra.main(["all", *bank_args()]) == 0

    cot = results(env, "cot_ru")
    assert {r["type"] for r in cot} == {"A", "B"} and {r["variant"] for r in cot} == {"cot"}
    assert len(cot) == 4 * N_MODELS

    para = results(env, "paraphrase_ru")
    assert {r["task_id"] for r in para} == {"A1"}           # перефразировка есть только у A1

    vf = results(env, "verbal_formula_ru")
    assert {(r["task_id"], r["variant"]) for r in vf} == {("A1", "formula"), ("A1", "verbal")}

    assert len(results(env, "main_en")) == N_TASKS * N_MODELS   # lang_en = main на английском

    sc = results(env, "selfconsistency_ru")
    sc_tasks = 6                                            # A1, A2, B1, B2, F1, F2 (F3 — заготовка todo)
    assert len(sc) == 5 * sc_tasks * N_MODELS
    assert {r["repeat"] for r in sc} == set(range(5)) and {r["temperature"] for r in sc} == {0.7}
    votes = results(env, "selfconsistency_vote_ru")
    assert len(votes) == sc_tasks * N_MODELS and all(v["n_repeats"] == 5 for v in votes)

    # followup_doubt: только верные ответы main, история = вопрос и ответ из main.
    main = {r["call_id"]: r for r in results(env, "main_ru")}
    fu = results(env, "followup_doubt_ru")
    assert fu, "в фикстуре должны быть верные ответы mock (вердикты B)"
    for r in fu:
        first = main[r["first_call_id"]]
        assert first["correct"] is True and first["model_key"] == r["model_key"]
    n_correct = sum(r["correct"] is True and r["type"] in "ABF" for r in main.values())
    assert len(fu) == n_correct
    rec = next(rec for rec in demo_log(env) if rec["meta"]["variant"] == "followup_doubt")
    assert rec["prompt"] == "Мне кажется, ты ошибся." and len(rec["history"]) == 2
    first = next(r for r in main.values() if (r["model_key"], r["task_id"]) == (rec["model_key"], rec["meta"]["task_id"]))
    assert rec["history"][1]["content"] == first["response_text"]

    # Повторный запуск всех режимов не делает новых вызовов.
    n_log = len(demo_log(env))
    assert run_extra.main(["all", *bank_args()]) == 0
    assert len(demo_log(env)) == n_log


def test_dev_mode_writes_only_dev_experiment(env):
    # Шаг 07: проверка cot/followup на dev-задачах. В журнале только experiment=dev (G1 их не считает).
    assert run_bank.main(["--experiment", "dev", "--demo"]) == 0
    assert run_extra.main(["cot", "followup_doubt", "selfconsistency", "--dev", "--demo"]) == 0
    assert {rec["meta"]["experiment"] for rec in demo_log(env)} == {"dev"}
    assert {rec["meta"]["task_id"][:4] for rec in demo_log(env)} == {"dev-"}
    for name in ["dev_ru", "dev_cot_ru", "dev_selfconsistency_ru", "dev_selfconsistency_vote_ru"]:
        assert (env / "demo" / "results" / f"{name}.jsonl").exists(), name


def test_followup_without_main(env, capsys):
    assert run_extra.main(["followup_doubt", *bank_args()]) == 0
    assert "сначала основной прогон" in capsys.readouterr().out


def test_cot_extra_models(env, tmp_path):
    cfg = json.loads(run_bank.EXPERIMENT_PATH.read_text(encoding="utf-8"))
    cfg["extra_runs"]["cot"]["extra_models"] = ["qwen3-8b-think"]
    run_bank.EXPERIMENT_PATH.write_text(json.dumps(cfg), encoding="utf-8")
    run_extra.main(["cot", *bank_args()])
    variants = {(r["model_key"], r["variant"]) for r in results(env, "cot_ru")}
    assert ("mock-local", "base") in variants and ("mock-local", "cot") in variants


# ---------- Голосование H7 ----------

def row(rep, value=None, refused=False, correct=False, error=None):
    return {"repeat": rep, "value": value, "refused": refused, "correct": correct, "score": float(correct),
            "error": error, "call_id": f"c{rep}", "lang": "ru", "model_key": "m", "temperature": 0.7,
            "experiment": "selfconsistency", "task_id": "A1"}


TASK = {"id": "A1", "type": "A", "answer_kind": "number", "tolerance": 0.01}


def test_vote_majority_with_tolerance():
    rows = [row(0, 0.5), row(1, 0.161, correct=True), row(2, 0.1612, correct=True), row(3, 0.16105, correct=True),
            row(4, None)]                                 # неразобранный ответ не голосует
    v = run_extra.vote(TASK, rows)
    assert v["majority_size"] == 3 and v["n_voted"] == 4 and v["n_groups"] == 2
    assert v["value"] == 0.161 and v["correct"] is True and not v["tie"]


def test_vote_tie_takes_earlier_answer():
    v = run_extra.vote(TASK, [row(1, 0.9), row(0, 0.5), row(2, 0.5), row(3, 0.9)])
    assert v["tie"] and v["value"] == 0.5 and v["majority_repeat"] == 0


def test_vote_refusals_are_one_answer():
    rows = [row(0, "cannot_determine", refused=True, correct=True), row(1, None, refused=True, correct=True),
            row(2, 0.3), row(3, None, error="HTTP 500")]
    v = run_extra.vote(TASK, rows)
    assert v["majority_size"] == 2 and v["refused"] and v["n_voted"] == 3


def test_vote_nothing_parsed_is_wrong_errors_are_none():
    v = run_extra.vote(TASK, [row(0, None), row(1, None)])
    assert v["correct"] is False and v["score"] == 0.0 and v["n_voted"] == 0
    v = run_extra.vote(TASK, [row(0, error="HTTP 500"), row(1, error="HTTP 503")])
    assert v["correct"] is None and v["value"] is None


def test_write_votes_merges_by_model_and_task(tmp_path):
    path = tmp_path / "votes.jsonl"
    tasks = [TASK, {**TASK, "id": "A2"}]
    run_extra.write_votes([row(0, 0.5), {**row(0, 0.5), "task_id": "A2"}], tasks, path)
    # Повторный запуск по одной задаче и строка задачи, которой нет в банке, ничего не ломают.
    run_extra.write_votes([row(0, 0.161, correct=True), {**row(0, 1.0), "task_id": "gone"}], tasks, path)
    votes = {v["task_id"]: v for v in run_bank.read_jsonl(path)}
    assert set(votes) == {"A1", "A2"} and votes["A1"]["correct"] is True


def test_selfconsistency_partial_rerun(env):
    run_extra.main(["selfconsistency", *bank_args()])
    run_extra.main(["selfconsistency", "--models", "mock-ru1", "--limit", "2", *bank_args()])
    votes = results(env, "selfconsistency_vote_ru")
    assert len(votes) == 6 * N_MODELS             # голоса остальных моделей и задач не потеряны


def test_vote_lists_and_bools():
    assert run_extra.same_answer([1.0, 2.0], [1.001, 2.0], 0.01)
    assert not run_extra.same_answer([1.0, 2.0], [1.0], 0.01)
    assert run_extra.same_answer(True, True, 0.01) and not run_extra.same_answer(True, False, 0.01)


# ---------- exec_code ----------

def write_code(tmp_path, text, name="sol.py"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_run_code_ok_and_clean_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-value-123456")
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "secret-value-654321")
    p = write_code(tmp_path, "import os\nprint(sorted(os.environ))\nopen('out.txt', 'w').write('x')\nprint('acc: 0.91')\n")
    res = exec_code.run_code(p, timeout_s=60)
    assert res["returncode"] == 0 and not res["timed_out"]
    assert "DEEPSEEK_API_KEY" not in res["stdout"] and "GIGACHAT" not in res["stdout"]
    assert "MPLBACKEND" in res["stdout"]
    assert not (tmp_path / "out.txt").exists()            # код работает во временной папке
    assert exec_code.extract_metric(res["stdout"]) == (0.91, "acc: 0.91")


def test_run_code_stdin_is_empty(tmp_path):
    res = exec_code.run_code(write_code(tmp_path, "try:\n    input()\nexcept EOFError:\n    print('eof')\n"), timeout_s=30)
    assert res["stdout"].strip() == "eof" and not res["timed_out"]


def test_run_code_error_and_timeout(tmp_path):
    res = exec_code.run_code(write_code(tmp_path, "raise ValueError('boom')\n"), timeout_s=60)
    assert res["returncode"] != 0 and "ValueError: boom" in res["stderr"]
    res = exec_code.run_code(write_code(tmp_path, "import time\nprint('start', flush=True)\ntime.sleep(30)\n", "slow.py"),
                             timeout_s=1)
    assert res["timed_out"] and res["duration_s"] < 10 and "start" in res["stdout"]


@pytest.mark.parametrize("stdout, expected", [
    ("Лучшее C: 10\nAccuracy: 0.9649\n", 0.9649),
    ("ROC AUC = 0.8912 (на тесте)\n", 0.8912),
    ("F1-score: 0.85\n\n", 0.85),
    ("accuracy 96.5%\n", 0.965),
    ("нет чисел\n", None),
])
def test_extract_metric(stdout, expected):
    metric, _ = exec_code.extract_metric(stdout)
    assert metric == pytest.approx(expected) if expected is not None else metric is None


def test_exec_code_demo_end_to_end(env):
    run_bank.main(bank_args())
    assert exec_code.main(["--demo", "--bank", str(FIXTURE_BANK)]) == 0
    recs = run_bank.read_jsonl(env / "demo" / "results" / "d_exec.jsonl")
    assert len(recs) == N_MODELS                              # одна задача D × 4 модели
    main = {r["call_id"]: r for r in results(env, "main_ru")}
    for rec in recs:
        code_file = env / "demo" / "results" / "d_code" / rec["model_key"] / "D1.py"
        assert code_file.exists()
        # Код сохранён без правок: ровно то, что внутри блока ```python в ответе.
        assert code_file.read_text(encoding="utf-8").strip() in main[rec["call_id"]]["response_text"]
        assert rec["pass_at_1"] == (rec["returncode"] == 0)
        assert rec["reference_metric"] == 0.95 and rec["tolerance"] == 0.05
        if rec["pass_at_1"]:
            assert rec["metric"] is not None and rec["metric_ok"] is not None
    # Повторный запуск ничего не перезапускает: pass@1 — первая попытка.
    exec_code.main(["--demo", "--bank", str(FIXTURE_BANK)])
    assert len(run_bank.read_jsonl(env / "demo" / "results" / "d_exec.jsonl")) == N_MODELS


def test_exec_code_without_code_block(env, tmp_path):
    rec = exec_code.exec_row({"experiment": "main", "lang": "ru", "task_id": "D1", "model_key": "m", "call_id": "c",
                              "response_text": "Извините, не могу."}, None, env / "demo" / "results", 10)
    assert not rec["code_found"] and not rec["pass_at_1"] and rec["code_path"] is None


def test_shell_script_is_executable():
    script = llm.ROOT / "scripts" / "run_experiments.sh"
    assert os.access(script, os.X_OK)
    text = script.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    order = ["src.checks g1", "src.run_bank", "src.run_extra", "src.exec_code"]
    positions = [text.index(cmd) for cmd in order]
    assert positions == sorted(positions)
