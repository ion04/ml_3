"""Тесты src/checks.py (гейт G1). Сеть не нужна: smoke-тест не запускается."""

import re
import json
import subprocess
from datetime import datetime, timezone

from src import checks

FILLED_MD = """# Гипотезы

## Гипотеза 1

- ID: H1
- Группа: 1
- Формулировка: учебник против изменённых
- Прогноз: подтвердится
- Обоснование прогноза: модели запоминают
- Метрика: разница точности
- Критерий решения (тест, порог): Макнемар, p < 0.05

## Гипотеза 2

- ID: H6
- Группа: 2
- Формулировка: CoT
- Прогноз: не подтвердится, потому что ...
- Обоснование прогноза: текст
- Метрика: точность
- Критерий решения (тест, порог): Макнемар

## Гипотеза 3

- ID: H9
- Группа: 3
- Формулировка: поддакивание
- Прогноз: подтвердится
- Обоснование прогноза: текст
- Метрика: flip rate
- Критерий решения (тест, порог): CI

---

## Коммит с прогнозом
"""


def test_hypotheses_config_groups():
    assert checks.check_hypotheses_config({"hypotheses": ["H1", "H6", "H9"]}) == []
    assert checks.check_hypotheses_config({"hypotheses": ["H5", "H10", "H12"]}) == []
    assert checks.check_hypotheses_config({"hypotheses": []})
    same_group = checks.check_hypotheses_config({"hypotheses": ["H1", "H2", "H9"]})
    assert any("разных групп" in p for p in same_group)
    assert any("неизвестные" in p for p in checks.check_hypotheses_config({"hypotheses": ["H1", "H6", "H14"]}))
    custom = {"hypotheses": ["H1", "H6", "H14"], "custom_hypothesis_groups": {"H14": 3}}
    assert checks.check_hypotheses_config(custom) == []


def test_hypotheses_md_filled_passes():
    assert checks.check_hypotheses_md(FILLED_MD, ["H1", "H6", "H9"]) == []


def test_template_md_fails():
    # Шаблон строим из FILLED_MD, а не читаем hypotheses.md проекта: он уже заполнен и закоммичен.
    template = re.sub(r"(?m)^(- (?:Прогноз|Обоснование прогноза): ).*$", r"\1TODO(человек)", FILLED_MD)
    problems = checks.check_hypotheses_md(template, ["H1", "H6", "H9"])
    assert any("не заполнено" in p for p in problems)
    assert any("Прогноз" in p for p in problems)


def test_hypotheses_md_problems():
    bad_prediction = FILLED_MD.replace("- Прогноз: подтвердится\n- Обоснование прогноза: модели",
                                       "- Прогноз: скорее да\n- Обоснование прогноза: модели")
    assert any("Прогноз" in p for p in checks.check_hypotheses_md(bad_prediction, ["H1", "H6", "H9"]))
    assert any("не совпадают" in p for p in checks.check_hypotheses_md(FILLED_MD, ["H1", "H6", "H10"]))
    todo = FILLED_MD.replace("- Метрика: flip rate", "- Метрика: TODO(человек)")
    assert any("Метрика" in p for p in checks.check_hypotheses_md(todo, ["H1", "H6", "H9"]))


def write_log(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for ts, exp in records:
            f.write(json.dumps({"ts_utc": ts, "model_key": "m", "meta": {"experiment": exp, "task_id": "A1"}}) + "\n")


def test_early_eval_calls(tmp_path):
    log = tmp_path / "calls.jsonl"
    write_log(log, [("2026-10-01T10:00:00.000+00:00", "smoke"),
                    ("2026-10-01T11:00:00.000+00:00", "dev"),
                    ("2026-10-02T10:00:00.000+00:00", "main"),
                    ("2026-10-04T10:00:00.000+00:00", "extra")])
    commit_time = datetime(2026, 10, 3, tzinfo=timezone.utc)
    early = checks.early_eval_calls(log, commit_time)
    assert len(early) == 1 and "experiment=main" in early[0]
    # Без коммита гипотез любой оценочный вызов — нарушение; smoke и dev — нет.
    assert len(checks.early_eval_calls(log, None)) == 2
    assert checks.early_eval_calls(tmp_path / "нет.jsonl", None) == []


def test_early_eval_calls_compares_timezones(tmp_path):
    """Время коммита git в локальной зоне (+03:00), журнал в UTC."""
    log = tmp_path / "calls.jsonl"
    write_log(log, [("2026-10-03T08:30:00.000+00:00", "main")])   # 11:30 по Москве
    assert checks.early_eval_calls(log, datetime.fromisoformat("2026-10-03T12:00:00+03:00"))
    assert not checks.early_eval_calls(log, datetime.fromisoformat("2026-10-03T11:00:00+03:00"))


def test_git_helpers(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    md = tmp_path / "hypotheses.md"
    md.write_text(FILLED_MD, encoding="utf-8")
    assert checks.last_commit(md, cwd=tmp_path) is None
    assert checks.is_modified(md, cwd=tmp_path)            # не отслеживается
    git("add", "hypotheses.md")
    git("commit", "-q", "-m", "гипотезы")
    sha, when = checks.last_commit(md, cwd=tmp_path)
    assert len(sha) == 40 and when.tzinfo is not None
    assert not checks.is_modified(md, cwd=tmp_path)
    md.write_text(FILLED_MD + "\nправка\n", encoding="utf-8")
    assert checks.is_modified(md, cwd=tmp_path)


def test_check_keys(monkeypatch):
    models = {"a": {"key": "a", "enabled": True, "provider": "openai_compat", "model": "x",
                    "base_url": "u", "api_key_env": "TEST_KEY_FOR_CHECKS"},
              "b": {"key": "b", "enabled": True, "provider": "mock", "model": "m", "base_url": None, "api_key_env": None},
              "c": {"key": "c", "enabled": False, "provider": "openai_compat", "model": "x", "base_url": "u",
                    "api_key_env": "OTHER_MISSING_KEY"}}
    monkeypatch.delenv("TEST_KEY_FOR_CHECKS", raising=False)
    assert checks.check_keys(models) == ["a: в .env нет TEST_KEY_FOR_CHECKS"]
    monkeypatch.setenv("TEST_KEY_FOR_CHECKS", "x")
    assert checks.check_keys(models) == []


def test_g2_stub_and_cli():
    assert checks.main(["g2"]) == 1
    assert checks.main(["нет-такой"]) == 2


def test_hypotheses_md_requires_all_lines_and_accepts_markdown():
    no_line = FILLED_MD.replace("- Обоснование прогноза: текст\n- Метрика: flip rate", "- Метрика: flip rate")
    assert any("нет строк Обоснование" in p for p in checks.check_hypotheses_md(no_line, ["H1", "H6", "H9"]))
    bold = FILLED_MD.replace("- Прогноз: подтвердится\n- Обоснование прогноза: модели",
                             "- Прогноз: **подтвердится**\n- Обоснование прогноза: модели")
    assert checks.check_hypotheses_md(bold, ["H1", "H6", "H9"]) == []


def test_broken_log_line_is_reported(tmp_path):
    log = tmp_path / "calls.jsonl"
    log.write_text('{"ts_utc": "2026-10-01T10:00:00+00:00", "meta": {"experiment": "smoke"}}\n{обрыв\n', encoding="utf-8")
    early = checks.early_eval_calls(log, None)
    assert len(early) == 1 and ":2:" in early[0]
