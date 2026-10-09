"""Тесты паспорта прогонов (src.manifest) и дозапроса технических ошибок (код выхода 3).

Всё на mock через --demo во временной папке; реальные logs/ и results/ не должны меняться.
"""

import json
from pathlib import Path

import pytest

from src import exec_code, llm, manifest, run_bank

FIXTURE_BANK = Path(__file__).parent / "fixtures" / "bank_small.jsonl"


def snapshot(path: Path) -> dict:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in path.rglob("*") if p.is_file()}


@pytest.fixture
def env(tmp_path, monkeypatch):
    real_dirs = [llm.ROOT / "logs", llm.ROOT / "results"]
    before = [snapshot(d) for d in real_dirs]
    monkeypatch.setattr(run_bank, "DEMO_DIR", tmp_path / "demo")
    monkeypatch.setattr(run_bank, "RESULTS_DIR", tmp_path / "real_results")
    monkeypatch.setattr(llm, "REAL_LOG", tmp_path / "real_logs" / "calls.jsonl")
    monkeypatch.setattr(llm, "_cache_index", {})
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    yield tmp_path
    assert [snapshot(d) for d in real_dirs] == before, "тест изменил logs/ или results/ проекта"


def demo_args(*extra):
    return ["--demo", "--bank", str(FIXTURE_BANK), *extra]


def rec(key, error=None, ts="2026-10-10T10:00:00+00:00", experiment="main"):
    return {"cache_key": key, "error": error, "ts_utc": ts, "meta": {"experiment": experiment}}


def test_call_stats_retried_and_unresolved():
    records = [rec("a", "HTTP 500"), rec("a"), rec("b", "HTTP 429"), rec("c", ts="2026-10-11T00:00:00+00:00")]
    s = manifest.call_stats(records)
    assert (s["records"], s["ok"], s["errors"]) == (4, 2, 2)
    assert s["retried_requests"] == 1        # a: ошибка, затем ответ
    assert s["unresolved_requests"] == 1     # b: ответа нет
    assert s["last_utc"].startswith("2026-10-11")


def test_demo_manifest(env):
    assert run_bank.main(["--experiment", "main", *demo_args()]) == 0
    assert exec_code.main(["--demo", "--bank", str(FIXTURE_BANK), "--experiment", "main,paraphrase"]) == 0
    assert manifest.main(["--demo"]) == 0

    path = env / "demo" / "results" / "run_manifest.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    log = run_bank.read_jsonl(env / "demo" / "calls.jsonl")
    assert m["demo"] is True and "ДЕМО" in m["_comment"]
    assert m["calls_by_experiment"]["main"]["records"] == len(log)
    assert m["calls_by_experiment"]["main"]["unresolved_requests"] == 0
    assert set(m["models"]) == {"mock-ru1", "mock-ru2", "mock-big", "mock-local"}
    one = m["models"]["mock-ru1"]
    assert one["model_reported"] and one["experiments"] == ["main"] and one["errors"] == 0
    rows = run_bank.read_jsonl(env / "demo" / "results" / "main_ru.jsonl")
    assert m["results"]["main_ru.jsonl"]["rows"] == len(rows)
    assert m["results"]["main_ru.jsonl"]["parse_ok"] == sum(r["parse_ok"] is True for r in rows)
    assert m["d_exec"]["runs"] == sum(r["answer_kind"] == "code" for r in rows)
    assert m["prompt"]["version_code"] and m["environment"]["python"]
    assert not (env / "real_results").exists()


def test_write_keeps_ollama_server(tmp_path):
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps({"ollama_server": {"gpu": "RTX 3090"}, "old": 1}), encoding="utf-8")
    manifest.write({"new": 2}, path)
    m = json.loads(path.read_text(encoding="utf-8"))
    assert m == {"new": 2, "ollama_server": {"gpu": "RTX 3090"}}   # старые поля не тянутся, сервер сохранён


def test_tech_errors_exit_3_then_only_errors_rerequested(env, monkeypatch):
    real_mock = llm._call_mock
    monkeypatch.setattr(llm, "_call_mock", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("сеть упала")))
    assert run_bank.main(["--experiment", "main", *demo_args("--limit", "2")]) == run_bank.EXIT_TECH_ERRORS
    n_failed = len(run_bank.read_jsonl(env / "demo" / "calls.jsonl"))
    assert n_failed > 0

    monkeypatch.setattr(llm, "_call_mock", real_mock)
    assert run_bank.main(["--experiment", "main", *demo_args("--limit", "2")]) == 0
    log = run_bank.read_jsonl(env / "demo" / "calls.jsonl")
    assert len(log) == 2 * n_failed                          # журнал дописан, ошибки остались видны
    assert sum(r["error"] is not None for r in log) == n_failed
    assert run_bank.main(["--experiment", "main", *demo_args("--limit", "2")]) == 0
    assert len(run_bank.read_jsonl(env / "demo" / "calls.jsonl")) == 2 * n_failed   # третий раз — всё из кэша
