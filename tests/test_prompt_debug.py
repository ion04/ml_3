"""Тесты src/prompt_debug.py: таблица отладки по журналу и заморозка промпта. Сети нет, журнал — временный файл."""

import json

from src import prompt_debug, prompts

V = prompts.PROMPT_VERSION


def rec(model, task_id, text, variant="base", version=V, error=None, finish="stop", shots=None, n=0):
    meta = {"experiment": "dev", "task_id": task_id, "variant": variant, "lang": "ru",
            "prompt_version": version, "repeat": 0}
    if shots is not None:
        meta["shots"] = shots
    return {"model_key": model, "meta": meta, "response_text": text, "error": error,
            "finish_reason": finish, "tokens_in": 100, "tokens_out": 50, "cache_key": f"{model}{task_id}{variant}{n}"}


def write_log(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


GOOD = {
    "dev-A1": "Решение.\nОТВЕТ: 0,6931\nУВЕРЕННОСТЬ: 90",
    "dev-B1": "Обоснование.\nОТВЕТ: верно\nУВЕРЕННОСТЬ: 90",
    "dev-C1": "Обоснование.\nОТВЕТ: kNN\nУВЕРЕННОСТЬ: 80",
    "dev-D1": "```python\nprint(0.97)\n```\nОТВЕТ: код выше\nУВЕРЕННОСТЬ: 80",
    "dev-F1": "Данных нет.\nОТВЕТ: нельзя определить\nУВЕРЕННОСТЬ: 90",
}


def test_format_checks(tmp_path):
    log = tmp_path / "calls.jsonl"
    records = [rec("m1", t, x) for t, x in GOOD.items()]
    records += [
        rec("m2", "dev-A1", "Ответ примерно 0.69"),                                   # нет строки ОТВЕТ
        rec("m2", "dev-B1", "ОТВЕТ: верно\nУВЕРЕННОСТЬ: 90\nНадеюсь, помог!"),       # хвост
        rec("m2", "dev-D1", "```python\nprint(1)\n```\nОТВЕТ:\n```python\nprint(1)\n```\nУВЕРЕННОСТЬ: 80"),
        rec("m2", "dev-F1", None, error="timeout"),
        rec("m2", "E5", "ОТВЕТ: отказ\nУВЕРЕННОСТЬ: 70", variant="classify_row", shots=5),
        {**rec("m1", "dev-A1", GOOD["dev-A1"]), "meta": {"experiment": "main", "task_id": "A1"}},  # не dev
    ]
    write_log(log, records)
    checked, text = prompt_debug.build(log)
    by = {(r["model"], r["task_id"]): r for r in checked}
    assert len(checked) == 10                       # запись main не попала
    assert all(by[("m1", t)]["format_ok"] and by[("m1", t)]["tail_ok"] for t in GOOD)
    assert by[("m2", "dev-A1")]["parse_ok"] is False
    assert by[("m2", "dev-B1")]["format_ok"] and not by[("m2", "dev-B1")]["tail_ok"]
    assert by[("m2", "dev-D1")]["parse_ok"] and not by[("m2", "dev-D1")]["format_ok"]   # два блока кода
    assert by[("m2", "dev-F1")]["error"]
    assert by[("m2", "E5")]["parse_ok"]
    assert f"| m1 | 5/5 · 5/5 |" in text


def test_retry_keeps_successful_attempt(tmp_path):
    log = tmp_path / "calls.jsonl"
    write_log(log, [rec("m1", "dev-A1", None, error="429"), rec("m1", "dev-A1", GOOD["dev-A1"])])
    checked, _ = prompt_debug.build(log)
    assert len(checked) == 1 and not checked[0]["error"] and checked[0]["parse_ok"]


def test_write_doc_keeps_manual_text(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text(f"ручной текст\n{prompt_debug.BEGIN}\nстарое\n{prompt_debug.END}\nхвост\n", encoding="utf-8")
    prompt_debug.write_doc("НОВАЯ ТАБЛИЦА", doc)
    out = doc.read_text(encoding="utf-8")
    assert "ручной текст" in out and "хвост" in out and "НОВАЯ ТАБЛИЦА" in out and "старое" not in out


def test_freeze_requires_all_models(tmp_path, monkeypatch):
    monkeypatch.setattr(prompt_debug.llm, "load_models", lambda: {
        "m1": {"enabled": True, "provider": "ollama"},
        "m2": {"enabled": True, "provider": "gigachat"},
        "mock-x": {"enabled": True, "provider": "mock"},
        "off": {"enabled": False, "provider": "ollama"},
    })
    log = tmp_path / "calls.jsonl"
    cfg = tmp_path / "experiment.json"
    cfg.write_text(json.dumps({"prompt_version": None, "prompt_hash": None}), encoding="utf-8")

    write_log(log, [rec("m1", t, x) for t, x in GOOD.items()])
    checked, _ = prompt_debug.build(log)
    problems = prompt_debug.freeze_problems(checked)
    assert len(problems) == 1 and problems[0].startswith("m2")
    assert prompt_debug.freeze(checked, cfg_path=cfg) == 1
    assert json.loads(cfg.read_text())["prompt_version"] is None      # без --force ничего не записано

    # Ответы старой версии промпта не считаются.
    write_log(log, [rec(m, t, x, version=(V if m == "m1" else "v0")) for m in ("m1", "m2") for t, x in GOOD.items()])
    assert prompt_debug.freeze_problems(prompt_debug.build(log)[0])

    write_log(log, [rec(m, t, x) for m in ("m1", "m2") for t, x in GOOD.items()])
    checked, _ = prompt_debug.build(log)
    assert prompt_debug.freeze(checked, cfg_path=cfg) == 0
    saved = json.loads(cfg.read_text())
    assert saved["prompt_version"] == V and saved["prompt_hash"] == prompts.prompt_hash()
