"""Тесты src/llm.py. Сеть не используется: только mock и подменённые HTTP-вызовы."""

import json
from types import SimpleNamespace

import pytest

from src import llm

TEST_MODELS = {
    "models": [
        {"key": "m1", "role": "big", "provider": "mock", "model": "mock-big", "base_url": None,
         "api_key_env": None, "min_temperature": 0.0, "params": {}, "enabled": True},
        {"key": "m_mintemp", "role": "ru1", "provider": "mock", "model": "mock-ru1", "base_url": None,
         "api_key_env": None, "min_temperature": 0.01, "params": {}, "enabled": True},
        {"key": "oll", "role": "local", "provider": "ollama", "model": "qwen3:8b",
         "base_url": "http://localhost:11434", "api_key_env": None, "min_temperature": 0.0,
         "params": {"think": False}, "enabled": True},
        {"key": "oai", "role": "big", "provider": "openai_compat", "model": "gpt://${TEST_FOLDER}/m/latest",
         "base_url": "https://example.invalid/v1", "api_key_env": "TEST_API_KEY", "min_temperature": 0.0,
         "project": "${TEST_FOLDER}", "params": {"extra_body": {"thinking": {"type": "disabled"}}},
         "enabled": True},
    ]
}


@pytest.fixture(autouse=True)
def setup(tmp_path, monkeypatch):
    """Свой конфиг моделей, сброс индекса кэша, паузы между ретраями — мгновенные."""
    cfg_path = tmp_path / "models.json"
    cfg_path.write_text(json.dumps(TEST_MODELS), encoding="utf-8")
    monkeypatch.setattr(llm, "MODELS_PATH", cfg_path)
    monkeypatch.setattr(llm, "_cache_index", {})
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    return sleeps


def read_log(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_log_record_written(tmp_path):
    log = tmp_path / "calls.jsonl"
    res = llm.ask("m1", "Сколько будет 2+2?", 0.0, system="Отвечай кратко", max_tokens=50,
                  meta={"task_id": "A01", "variant": "original", "experiment": "test", "lang": "ru"},
                  log_path=log)
    assert res.error is None and res.text and not res.cached
    assert "ОТВЕТ:" in res.text
    recs = read_log(log)
    assert len(recs) == 1
    rec = recs[0]
    for field in ["call_id", "ts_utc", "model_key", "provider", "model_requested", "model_reported",
                  "temperature_requested", "effective_temperature", "max_tokens", "system", "history",
                  "prompt", "response_text", "tokens_in", "tokens_out", "latency_s", "error", "attempt",
                  "meta", "cache_key", "local_mem_mb"]:
        assert field in rec, field
    assert rec["call_id"] == res.call_id
    assert rec["response_text"] == res.text
    assert rec["meta"] == {"task_id": "A01", "variant": "original", "experiment": "test",
                           "lang": "ru", "repeat": 0}
    assert rec["system"] == "Отвечай кратко" and rec["attempt"] == 1


def test_mock_is_deterministic(tmp_path):
    a = llm.ask("m1", "вопрос", log_path=tmp_path / "a.jsonl")
    b = llm.ask("m1", "вопрос", log_path=tmp_path / "b.jsonl")
    assert a.text == b.text and a.call_id != b.call_id


def test_cache_returns_logged_answer(tmp_path, monkeypatch):
    log = tmp_path / "calls.jsonl"
    first = llm.ask("m1", "вопрос", log_path=log)

    def must_not_call(*args, **kwargs):
        raise AssertionError("API не должен вызываться при попадании в кэш")

    monkeypatch.setattr(llm, "_call_mock", must_not_call)
    second = llm.ask("m1", "вопрос", log_path=log)
    assert second.cached and second.text == first.text and second.call_id == first.call_id
    assert len(read_log(log)) == 1  # попадание в кэш журнал не меняет

    # Кэш работает и в новом процессе: индекс восстанавливается из файла.
    monkeypatch.setattr(llm, "_cache_index", {})
    third = llm.ask("m1", "вопрос", log_path=log)
    assert third.cached and third.call_id == first.call_id


def test_cache_key_depends_on_inputs(tmp_path):
    log = tmp_path / "calls.jsonl"
    llm.ask("m1", "вопрос", log_path=log)
    assert not llm.ask("m1", "вопрос", repeat=1, log_path=log).cached
    assert not llm.ask("m1", "вопрос", 0.7, log_path=log).cached
    assert not llm.ask("m1", "вопрос", system="sys", log_path=log).cached
    assert not llm.ask("m1", "вопрос", max_tokens=10, log_path=log).cached
    assert not llm.ask("m1", "другой вопрос", log_path=log).cached
    assert not llm.ask("m1", "вопрос", use_cache=False, log_path=log).cached
    assert len(read_log(log)) == 7


class Fake429(Exception):
    status_code = 429


class Fake401(Exception):
    status_code = 401


def test_retry_on_429(tmp_path, monkeypatch, setup):
    log = tmp_path / "calls.jsonl"
    real_mock = llm._call_mock
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise Fake429("Too Many Requests")
        return real_mock(*args, **kwargs)

    monkeypatch.setattr(llm, "_call_mock", flaky)
    res = llm.ask("m1", "вопрос", log_path=log)
    assert res.error is None and res.text
    recs = read_log(log)
    assert [r["attempt"] for r in recs] == [1, 2, 3]  # каждая попытка видна в журнале
    assert recs[0]["error"].startswith("HTTP 429") and recs[1]["error"].startswith("HTTP 429")
    assert recs[2]["error"] is None
    assert setup == [2.0, 4.0]  # экспоненциальная пауза
    # В кэш попадает только успешная попытка.
    assert llm.ask("m1", "вопрос", log_path=log).call_id == recs[2]["call_id"]


def test_retry_gives_up(tmp_path, monkeypatch):
    log = tmp_path / "calls.jsonl"
    monkeypatch.setattr(llm, "_call_mock", lambda *a, **k: (_ for _ in ()).throw(Fake429("limit")))
    res = llm.ask("m1", "вопрос", log_path=log)
    assert res.text is None and "429" in res.error
    assert len(read_log(log)) == llm.MAX_ATTEMPTS
    # Неудачные попытки не кэшируются: следующий запуск снова пойдёт в API.
    assert not llm.ask("m1", "вопрос", log_path=log).cached


def test_no_retry_on_client_error(tmp_path, monkeypatch, setup):
    log = tmp_path / "calls.jsonl"
    monkeypatch.setattr(llm, "_call_mock", lambda *a, **k: (_ for _ in ()).throw(Fake401("unauthorized")))
    res = llm.ask("m1", "вопрос", log_path=log)
    assert res.text is None and "401" in res.error
    assert len(read_log(log)) == 1 and setup == []


def test_is_retryable():
    import httpx

    assert llm.is_retryable(httpx.ConnectError("refused"))
    assert llm.is_retryable(httpx.ReadTimeout("timeout"))
    assert llm.is_retryable(Fake429())
    assert llm.is_retryable(type("E503", (Exception,), {"status_code": 503})())
    assert not llm.is_retryable(Fake401())
    assert not llm.is_retryable(ValueError("bad"))


def test_multiturn_history(tmp_path, monkeypatch):
    log = tmp_path / "calls.jsonl"
    first = llm.ask("m1", "Сколько будет 2+2?", log_path=log)
    history = [{"role": "user", "content": "Сколько будет 2+2?"},
               {"role": "assistant", "content": first.text}]
    seen = {}
    real_mock = llm._call_mock

    def spy(cfg, model_id, messages, *args):
        seen["messages"] = messages
        return real_mock(cfg, model_id, messages, *args)

    monkeypatch.setattr(llm, "_call_mock", spy)
    second = llm.ask("m1", "Мне кажется, ты ошибся.", system="sys", history=history, log_path=log)
    assert [m["role"] for m in seen["messages"]] == ["system", "user", "assistant", "user"]
    assert seen["messages"][-1]["content"] == "Мне кажется, ты ошибся."
    assert "сообщений в диалоге: 4" in second.text
    rec = read_log(log)[-1]
    assert rec["history"] == history and rec["prompt"] == "Мне кажется, ты ошибся."
    # Тот же вопрос без истории — другой ключ кэша.
    assert rec["cache_key"] != read_log(log)[0]["cache_key"]


def test_bad_history_rejected(tmp_path):
    with pytest.raises(ValueError):
        llm.ask("m1", "q", history=[{"role": "system", "content": "x"}], log_path=tmp_path / "c.jsonl")


def test_min_temperature(tmp_path):
    log = tmp_path / "calls.jsonl"
    res = llm.ask("m_mintemp", "вопрос", 0.0, log_path=log)
    assert res.effective_temperature == 0.01
    rec = read_log(log)[0]
    assert rec["temperature_requested"] == 0.0 and rec["effective_temperature"] == 0.01
    assert llm.ask("m_mintemp", "вопрос", 0.7, log_path=log).effective_temperature == 0.7


def test_no_secrets_in_log(tmp_path, monkeypatch):
    secret = "sk-very-secret-value-12345"
    monkeypatch.setenv("TEST_API_KEY", secret)
    monkeypatch.setenv("GIGACHAT_CREDENTIALS", "Z2lnYWNoYXQtc2VjcmV0LWNyZWRz")
    log = tmp_path / "calls.jsonl"
    # Ошибка, в текст которой SDK «случайно» вставил ключ.
    monkeypatch.setattr(llm, "_call_mock",
                        lambda *a, **k: (_ for _ in ()).throw(Fake401(f"bad key {secret}")))
    res = llm.ask("m1", f"промпт с ключом {secret} и Z2lnYWNoYXQtc2VjcmV0LWNyZWRz", log_path=log)
    text = log.read_text(encoding="utf-8")
    assert secret not in text and "Z2lnYWNoYXQtc2VjcmV0LWNyZWRz" not in text
    assert "***" in text
    assert secret not in res.error


def test_mock_never_writes_to_real_logs():
    with pytest.raises(ValueError):
        llm.ask("m1", "q", log_path=llm.ROOT / "logs" / "calls.jsonl")


def test_mock_default_log_is_demo(monkeypatch, tmp_path):
    demo = tmp_path / "demo" / "calls.jsonl"
    monkeypatch.setattr(llm, "DEMO_LOG", demo)
    llm.ask("m1", "q")
    assert demo.exists() and len(read_log(demo)) == 1


def test_env_substitution_and_missing(monkeypatch):
    cfg = llm.get_model("oai")
    monkeypatch.delenv("TEST_FOLDER", raising=False)
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    assert sorted(llm.missing_env(cfg)) == ["TEST_API_KEY", "TEST_FOLDER"]
    monkeypatch.setenv("TEST_FOLDER", "b1gfolder")
    assert llm._expand_env(cfg["model"]) == "gpt://b1gfolder/m/latest"


def test_openai_compat_parsing(tmp_path, monkeypatch):
    """Разбор ответа OpenAI-совместимого API на подменённом клиенте (без сети)."""
    import openai

    monkeypatch.setenv("TEST_API_KEY", "test-key-123456")
    monkeypatch.setenv("TEST_FOLDER", "b1gfolder")
    sent = {}

    class FakeClient:
        def __init__(self, **kwargs):
            sent["client"] = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            sent["request"] = kwargs
            msg = SimpleNamespace(content="ОТВЕТ: 4", reasoning_content=None)
            return SimpleNamespace(
                model="yandexgpt/latest@abc", usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3),
                choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    log = tmp_path / "calls.jsonl"
    res = llm.ask("oai", "2+2?", max_tokens=20, log_path=log)
    assert res.text == "ОТВЕТ: 4" and res.tokens_in == 11 and res.tokens_out == 3
    assert res.model_reported == "yandexgpt/latest@abc"
    assert sent["request"]["model"] == "gpt://b1gfolder/m/latest"
    assert sent["request"]["temperature"] == 0.0 and sent["request"]["max_tokens"] == 20
    assert sent["client"]["max_retries"] == 0  # повторы делает только ask()
    assert sent["client"]["project"] == "b1gfolder"
    assert sent["request"]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "test-key-123456" not in log.read_text(encoding="utf-8")


def test_ollama_parsing_and_memory(tmp_path, monkeypatch):
    """Разбор ответа нативного API Ollama, think=false в запросе, замер памяти (без сети)."""
    import httpx

    sent = {}

    def fake_post(url, json, timeout):
        sent["url"], sent["body"] = url, json
        data = {"model": "qwen3:8b", "message": {"role": "assistant", "content": "4"},
                "done_reason": "stop", "prompt_eval_count": 15, "eval_count": 2}
        return httpx.Response(200, json=data, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    monkeypatch.setattr(llm, "_ollama_rss_mb", lambda: 1234.5)
    monkeypatch.setattr(llm, "ollama_ps", lambda base_url, model_id: {"size_mb": 6000.0, "size_vram_mb": 6000.0})
    log = tmp_path / "calls.jsonl"
    res = llm.ask("oll", "2+2?", max_tokens=30, repeat=2, log_path=log)
    assert res.text == "4" and res.tokens_in == 15 and res.tokens_out == 2
    assert sent["url"] == "http://localhost:11434/api/chat"
    assert sent["body"]["think"] is False and sent["body"]["stream"] is False
    assert sent["body"]["options"] == {"temperature": 0.0, "seed": 44, "num_predict": 30}
    rec = read_log(log)[0]
    assert rec["local_mem_mb"] == 1234.5 and rec["local_vram_mb"] == 6000.0


def test_project_config_is_valid():
    """config/models.json проекта: все поля контракта на месте, роли и провайдеры допустимы."""
    models = llm.load_models(llm.ROOT / "config" / "models.json")
    roles = {"ru1", "ru2", "big", "local"}
    for m in models.values():
        for field in ["key", "role", "provider", "model", "base_url", "api_key_env",
                      "min_temperature", "params", "enabled"]:
            assert field in m, (m["key"], field)
        assert m["role"] in roles
        assert m["provider"] in {"gigachat", "openai_compat", "ollama", "mock"}
        if m["provider"] == "mock":
            assert not m["enabled"], "mock-модели не должны попадать в реальные прогоны"
    enabled_roles = {m["role"] for m in models.values() if m["enabled"]}
    assert enabled_roles == roles  # по одной роли из таблицы PDF как минимум


def test_ssl_error_not_retried():
    import httpx

    assert not llm.is_retryable(httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"))
