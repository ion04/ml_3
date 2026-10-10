"""Тесты шага 10: автопроверка (src/autograde.py), слепые листы (src/blind.py), гейт G2, тарифы.

Сеть не нужна. Таблицы ответов пишутся во временную папку; сквозной тест гоняет mock-прогон
через фикстуру env из test_runners (реальные logs/ и results/ не меняются).
"""

import csv
import json
from pathlib import Path

import pytest

from src import autograde, blind, checks, exec_code, llm, run_bank, typeE
from tests.test_runners import FIXTURE_BANK, bank_args, env  # noqa: F401  (env — фикстура pytest)

TASKS = {t["id"]: t for t in run_bank.read_jsonl(FIXTURE_BANK)}
SPLIT = typeE.current_split_hash()      # data/e_split.json закоммичен (шаг 06)


def answer(task_id, text, model="m1", experiment="main", lang="ru", correct=None, error=None, **extra):
    t = TASKS.get(task_id, {"type": "E", "answer_kind": "label"})
    row = {"experiment": experiment, "lang": lang, "task_id": task_id, "type": t["type"], "topic": t.get("topic"),
           "source": t.get("source"), "answer_kind": t["answer_kind"], "task_variant": t.get("variant"),
           "variant": "base", "model_key": model, "model_reported": model + "-v1", "repeat": 0,
           "call_id": f"{model}-{experiment}-{lang}-{task_id}-{extra.get('shots', '')}", "error": error,
           "response_text": text, "correct": correct}
    row.update(extra)
    return row


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def d_exec(row, **kw):
    rec = {"experiment": row["experiment"], "lang": row["lang"], "task_id": row["task_id"],
           "model_key": row["model_key"], "call_id": row["call_id"], "code_found": True, "extract_notes": [],
           "returncode": 0, "timed_out": False, "duration_s": 1.0, "stdout": "accuracy: 0.95\n", "stderr": "",
           "metric": 0.95, "metric_line": "accuracy: 0.95", "metric_ok": True, "pass_at_1": True}
    rec.update(kw)
    return rec


@pytest.fixture
def results(tmp_path):
    """Маленький набор таблиц: main_ru (A, B, C, D, F), cot_ru, typeE_llm, dev_ru (должна игнорироваться)."""
    d_ok = answer("D1", "Я — GigaChat.\n```python\nprint('accuracy: 0.95')\n```\nОТВЕТ: код выше")
    d_nometric = answer("D1", "```python\nprint('готово')\n```", model="m2")
    d_crash = answer("D1", "```python\n1/0\n```", model="m3")
    main = [
        answer("A1", "ОТВЕТ: 0,161\nУВЕРЕННОСТЬ: 90", correct=True),
        answer("A1", "ОТВЕТ: 0.5", model="m2", correct=True),          # при прогоне считалось верным → изменилось
        answer("B1", "As DeepSeek, I think...\nОТВЕТ: неверно\nУВЕРЕННОСТЬ: 80", correct=True),
        answer("C1", "<think>Qwen думает</think>Я бы взял PCA. Qwen3:8b рекомендует.\nОТВЕТ: PCA"),
        answer("F2", "ОТВЕТ: нельзя определить", correct=True),
        answer("F1", None, error="timeout"),
        d_ok, d_nometric, d_crash,
    ]
    write_jsonl(tmp_path / "main_ru.jsonl", main)
    write_jsonl(tmp_path / "cot_ru.jsonl", [answer("B1", "ОТВЕТ: верно", experiment="cot", correct=False)])
    write_jsonl(tmp_path / "typeE_llm.jsonl", [
        answer("E5", "ОТВЕТ: отказ", experiment="typeE", shots=0, row_idx=5, y_true=1, correct=True, split_hash=SPLIT),
        answer("E7", "Отказа нет.", experiment="typeE", shots=5, row_idx=7, y_true=0, correct=True, split_hash=SPLIT),
        answer("E9", "ОТВЕТ: отказ", experiment="typeE", shots=0, row_idx=9, y_true=1, split_hash="old"),
    ])
    write_jsonl(tmp_path / "dev_ru.jsonl", [answer("A1", "ОТВЕТ: 1", experiment="dev")])
    write_jsonl(tmp_path / "d_exec.jsonl", [
        d_exec(d_ok),
        d_exec(d_nometric, stdout="готово\n", metric=None, metric_line=None, metric_ok=None),
        d_exec(d_crash, returncode=1, pass_at_1=False, metric=None, metric_ok=None, stderr="ZeroDivisionError"),
    ])
    return tmp_path


# ---------- Автопроверка ----------

def test_autograde_build(results):
    rows, skipped = autograde.build(results, TASKS)
    by = {(r["check"], r["experiment"], r["model_key"], r["task_id"]): r for r in rows}
    assert skipped["техошибка"] == 1 and skipped["C (слепые листы)"] == 1 and skipped["D без запуска кода"] == 0
    assert skipped["E на старом разбиении"] == 1
    assert not any(r["experiment"] == "dev" for r in rows)
    a1 = by[("answer", "main", "m1", "A1")]
    assert a1["correct"] is True and a1["value"] == pytest.approx(0.161)
    a1_m2 = by[("answer", "main", "m2", "A1")]
    assert a1_m2["correct"] is False and a1_m2["correct_at_run"] is True     # пересчёт по текущему эталону
    b1 = by[("answer", "main", "m1", "B1")]
    assert b1["correct"] is True and b1["needs_manual"] is True
    assert by[("answer", "cot", "m1", "B1")]["needs_manual"] is False       # cot не идёт в слепые листы
    assert by[("answer", "main", "m1", "F2")]["correct"] is True
    assert by[("answer", "typeE", "m1", "E5")]["correct"] is True
    e7 = by[("answer", "typeE", "m1", "E7")]
    assert e7["value"] == 0 and e7["correct"] is True and e7["shots"] == 5
    assert by[("d_exec", "main", "m1", "D1")]["correct"] is True
    manual = by[("d_exec", "main", "m2", "D1")]
    assert manual["correct"] is None and "корректность" in manual["manual_part"]
    crash = by[("d_exec", "main", "m3", "D1")]
    assert crash["correct"] is False and crash["pass_at_1"] is False


def test_autograde_main_and_spotcheck_protection(results, tmp_path, monkeypatch):
    out = tmp_path / "demo"
    monkeypatch.setattr(run_bank, "DEMO_DIR", out)
    monkeypatch.setattr(run_bank, "out_dir", lambda demo: results)
    assert autograde.main(["--demo", "--bank", str(FIXTURE_BANK)]) == 0
    graded = run_bank.read_jsonl(results / "graded_auto.jsonl")
    assert len(graded) == 10                     # 7 ответов A/B/E/F + 3 запуска D
    spot_path = out / "grading" / "parser_spotcheck.csv"
    sheet = autograde.read_csv(spot_path)
    assert len(sheet) == 7 and all(r["human_agrees"] == "" for r in sheet)    # ответов меньше 20 — берутся все
    assert list(sheet[0]) == autograde.SPOTCHECK_COLUMNS
    # Человек начал заполнять — повторный запуск лист не трогает.
    sheet[0]["human_agrees"] = "да"
    autograde.write_csv(spot_path, sheet, autograde.SPOTCHECK_COLUMNS)
    assert autograde.main(["--demo", "--bank", str(FIXTURE_BANK)]) == 0
    assert autograde.read_csv(spot_path)[0]["human_agrees"] == "да"


def test_autograde_no_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(run_bank, "out_dir", lambda demo: tmp_path)
    assert autograde.main(["--demo"]) == 2
    assert not (tmp_path / "graded_auto.jsonl").exists()


def fake_graded(n, **kw):
    base = {"check": "answer", "experiment": "main", "lang": "ru", "model_key": "m", "variant": "base",
            "repeat": 0, "shots": None, "type": "A", "parse_ok": True, "parse_notes": []}
    return [{**base, "task_id": f"T{i:03d}", **kw} for i in range(n)]


def test_spotcheck_sample_priorities():
    rows = (fake_graded(50) + fake_graded(8, type="E", experiment="typeE", parse_notes=[autograde.E_NOTE])
            + fake_graded(12, experiment="cot", parse_ok=False))
    sample = autograde.spotcheck_sample(rows, seed=42)
    sel = [s for s, _ in sample]
    assert len(sample) == 20
    assert sel.count("подозрительный E") == 5 and sel.count("подозрительный") == 5 and sel.count("случайный") == 10
    assert len({id(r) for _, r in sample}) == 20
    again = autograde.spotcheck_sample(rows, seed=42)
    assert [r["task_id"] + r["experiment"] for _, r in sample] == [r["task_id"] + r["experiment"] for _, r in again]
    # Случайная часть — поровну по типам, хотя E в пуле больше всего.
    rows = fake_graded(10) + fake_graded(500, type="E", experiment="typeE") + fake_graded(10, type="F")
    random_types = [r["type"] for s, r in autograde.spotcheck_sample(rows, seed=1) if s == "случайный"]
    assert sorted(set(random_types)) == ["A", "E", "F"] and max(map(random_types.count, "AEF")) <= 7


# ---------- Анонимизация ----------

@pytest.mark.parametrize("text", [
    "Я — GigaChat, модель от Сбера.", "As DeepSeek-V3, I believe", "Я Qwen3:8b, созданный Alibaba Cloud",
    "Я YandexGPT 5 Pro", "Меня зовут Алиса", "Как Yandex GPT, отвечу", "Ответ от ГигаЧат", "I am ChatGPT (GPT-4o)",
    "Модель gigachat-2-max ответила", "deepseek-v4-pro: ответ", "Я Дипсик", "Я — _GigaChat_.", "I am Deep Seek",
    "Я Giga Chat", "DeepSeek‑V3 (дефис U+2011)", "Я Tongyi Qianwen", "我是通义千问", "Подробнее: ya.ru",
])
def test_anonymize_masks_names(text):
    pattern = blind.name_regex(blind.model_names([]))
    masked = blind.anonymize(text, pattern)
    for word in ["gigachat", "giga chat", "сбер", "deepseek", "deep seek", "qwen", "qianwen", "alibaba", "yandex",
                 "алис", "гигачат", "gpt", "дипсик", "通义", "ya.ru", "-v3", "‑v3"]:
        assert word not in masked.lower(), masked
    assert blind.MASK in masked


@pytest.mark.parametrize("text", [
    "Сбережения и сберечь память", "Alice и Bob", "Claude Shannon ввёл энтропию", "df = pd.read_csv('taxi.csv')",
    "Используем GridSearchCV(scoring='f1')",
])
def test_anonymize_no_false_positives(text):
    assert blind.anonymize(text, blind.name_regex(blind.model_names([]))) == text


def test_anonymize_keeps_ml_text_and_strips_think():
    pattern = blind.name_regex(blind.model_names([{"model_reported": "GigaChat-2-Max:2.0.28.2"}]))
    assert blind.anonymize("Ответ.<think>обрезано на полуслове", pattern) == "Ответ."
    text = "<think>я Qwen</think>Используйте Pipeline(StandardScaler(), SVC()), kNN и PCA; accuracy 0.97."
    assert blind.anonymize(text, pattern) == "Используйте Pipeline(StandardScaler(), SVC()), kNN и PCA; accuracy 0.97."
    assert blind.anonymize("Версия GigaChat-2-Max:2.0.28.2.", pattern) == "Версия " + blind.MASK + "."


# ---------- Слепые листы ----------

def test_blind_build(results):
    sheet, key = blind.build(results, ["main", "paraphrase"], TASKS, seed=42)
    assert [r["type"] for r in sheet] == ["B", "C", "D", "D", "D"]
    assert [r["item_id"] for r in sheet] == [k["item_id"] for k in key] == ["B001", "C001", "D001", "D002", "D003"]
    assert len({k["call_id"] for k in key}) == 5
    assert list(sheet[0]) == blind.SHEET_COLUMNS
    flat = json.dumps(sheet, ensure_ascii=False).lower()
    for leak in ["m1", "m2", "m3", "gigachat", "deepseek", "qwen", "<think>", "думает"]:
        assert leak not in flat, leak
    b = sheet[0]
    assert b["score_0_2"] == "" and b["reference"] == "утверждение неверно" and b["d_result_correct"] == blind.NA
    d = {k["model_key"]: s for k, s in zip(key, sheet) if s["type"] == "D"}
    assert d["m1"]["d_result_correct"] == blind.NA and "верно (авто)" in d["m1"]["auto_metric"]
    assert d["m2"]["d_result_correct"] == "" and "не извлечена" in d["m2"]["auto_metric"]
    # Код упал: автопроверка говорит «неверно», но корректность всё равно ставят люди (это лишь подсказка).
    assert d["m3"]["d_result_correct"] == "" and "ZeroDivisionError" in d["m3"]["exec_result"]
    req = {k["item_id"]: k["required"] for k in key}
    assert req["B001"] == "score_0_2" and req[next(k["item_id"] for k in key if k["model_key"] == "m1" and k["type"] == "D")] \
        == ";".join(blind.CHECKLIST)
    assert all(d["m1"][c] == "" for c in blind.CHECKLIST) and d["m1"]["score_0_2"] == blind.NA
    # Детерминированность: тот же seed → те же файлы.
    assert blind.build(results, ["main", "paraphrase"], TASKS, seed=42) == (sheet, key)


def fill(sheet_rows: list[dict], score="2", check="0", correct="1") -> list[dict]:
    out = []
    for r in sheet_rows:
        r = dict(r)
        for col, val in [(blind.SCORE_COL, score), (blind.CORRECT_COL, correct), *[(c, check) for c in blind.CHECKLIST]]:
            if r[col] == "":
                r[col] = val
        out.append(r)
    return out


@pytest.fixture
def grading(results, tmp_path, monkeypatch):
    """Листы, созданные src.blind --demo в tmp_path/demo/grading."""
    monkeypatch.setattr(run_bank, "DEMO_DIR", tmp_path / "demo")
    monkeypatch.setattr(run_bank, "out_dir", lambda demo: results)
    assert blind.main(["--demo", "--bank", str(FIXTURE_BANK)]) == 0
    return tmp_path / "demo" / "grading"


def test_blind_main_writes_identical_sheets_and_protects_grades(grading, results):
    r1, r2 = (autograde.read_csv(grading / s) for s in blind.SHEETS)
    assert r1 == r2 and len(r1) == 5
    assert (grading / blind.KEY_NAME).exists()
    # Начали оценивать → при тех же ответах ничего не меняется.
    autograde.write_csv(grading / "sheet_r1.csv", fill(r1), blind.SHEET_COLUMNS)
    assert blind.main(["--demo", "--bank", str(FIXTURE_BANK)]) == 0
    assert autograde.read_csv(grading / "sheet_r1.csv") == fill(r1)
    # Появился новый ответ → остановка, листы не тронуты.
    rows = run_bank.read_jsonl(results / "main_ru.jsonl") + [answer("C1", "ОТВЕТ: kNN", model="m9")]
    write_jsonl(results / "main_ru.jsonl", rows)
    assert blind.main(["--demo", "--bank", str(FIXTURE_BANK)]) == 1
    assert autograde.read_csv(grading / "sheet_r1.csv") == fill(r1)


# ---------- Гейт G2 ----------

def write_pricing(path: Path, verified: bool):
    models = {k: {"rub_per_1m_in": 1.0, "rub_per_1m_out": 2.0} for k in llm.load_models()}
    path.write_text(json.dumps({"verified_by_human": verified, "models": models}), encoding="utf-8")


def write_spot(path: Path, n: int, value: str):
    autograde.write_csv(path, [{"spot_id": f"S{i:02d}", "human_agrees": value} for i in range(n)],
                        ["spot_id", "human_agrees"])


def test_g2_passes_when_everything_filled(grading, tmp_path, capsys):
    for s in blind.SHEETS:
        autograde.write_csv(grading / s, fill(autograde.read_csv(grading / s)), blind.SHEET_COLUMNS)
    write_spot(grading / autograde.SPOTCHECK_NAME, 20, "Да ")
    write_pricing(tmp_path / "pricing.json", True)
    assert checks.g2(grading, tmp_path / "pricing.json") == 0
    assert "совпадают полностью" in capsys.readouterr().out     # одинаковые оценки → предупреждение


def test_g2_reports_problems(grading, tmp_path, capsys):
    r1 = fill(autograde.read_csv(grading / "sheet_r1.csv"))
    r1[0][blind.SCORE_COL] = "3"                                  # недопустимый балл
    autograde.write_csv(grading / "sheet_r1.csv", r1, blind.SHEET_COLUMNS)
    r2 = fill(autograde.read_csv(grading / "sheet_r2.csv"), check="Н/П", score="1.0")[1:]   # строка удалена
    autograde.write_csv(grading / "sheet_r2.csv", r2, blind.SHEET_COLUMNS)
    write_spot(grading / autograde.SPOTCHECK_NAME, 19, "")
    write_pricing(tmp_path / "pricing.json", False)
    assert checks.g2(grading, tmp_path / "pricing.json") == 1
    out = capsys.readouterr().out
    assert "недопустимые значения: 1" in out and "не совпадают с ключом" in out
    assert "строк 19" in out and "human_agrees не заполнен" in out and "verified_by_human" in out


def test_g2_dash_in_required_cell_and_stale_answer(grading, tmp_path, capsys):
    r1 = fill(autograde.read_csv(grading / "sheet_r1.csv"))
    d_row = next(r for r in r1 if r["type"] == "D")
    d_row["d1_preprocessing_before_split"] = blind.NA          # «—» вместо «н/п» в обязательной клетке
    r1[0]["answer_text"] = "совсем другой ответ"               # лист не от этого набора ответов
    autograde.write_csv(grading / "sheet_r1.csv", r1, blind.SHEET_COLUMNS)
    problems = checks.check_sheet(grading / "sheet_r1.csv", autograde.read_csv(grading / blind.KEY_NAME))
    assert any("не заполнено клеток: 1" in p for p in problems)
    assert any("текст ответа не совпадает" in p for p in problems)


def test_excel_semicolon_sheet_is_read_and_protected(grading):
    """Русский Excel сохраняет CSV с «;» (иногда в cp1251): лист читается, оценки не затираются."""
    rows = fill(autograde.read_csv(grading / "sheet_r1.csv"))
    with open(grading / "sheet_r1.csv", "w", encoding="cp1251", errors="replace", newline="") as f:
        w = csv.DictWriter(f, fieldnames=blind.SHEET_COLUMNS, delimiter=";")
        w.writeheader()
        w.writerows(rows)
    back = autograde.read_csv(grading / "sheet_r1.csv")
    assert [r["score_0_2"] for r in back] == [r["score_0_2"] for r in rows]
    assert blind.main(["--demo", "--bank", str(FIXTURE_BANK)]) == 0      # тот же набор — ничего не меняет
    assert autograde.read_csv(grading / "sheet_r1.csv")[0]["score_0_2"] == rows[0]["score_0_2"]
    assert not any("не заполнено" in p for p in
                   checks.check_sheet(grading / "sheet_r1.csv", autograde.read_csv(grading / blind.KEY_NAME)))


def test_g2_without_sheets(tmp_path, capsys):
    assert checks.g2(tmp_path, tmp_path / "nope.json") == 1
    assert "листы ещё не созданы" in capsys.readouterr().out


def test_norm():
    assert checks.norm(" 2.0 ") == "2" and checks.norm("1,0") == "1" and checks.norm("N/A") == "н/п"
    assert checks.norm("Нет") == "нет" and "согласен" not in checks.AGREE_VALUES


# ---------- Тарифы ----------

def test_pricing_file_structure():
    pricing = json.loads((llm.ROOT / "config" / "pricing.json").read_text(encoding="utf-8"))
    assert isinstance(pricing["verified_by_human"], bool) and pricing["date_checked"]
    assert pricing["usd_rub"]["rate"] > 0 and pricing["usd_rub"]["source_url"].startswith("https://www.cbr.ru")
    for key, cfg in llm.load_models().items():
        if cfg["enabled"] and cfg["provider"] != "mock":
            p = pricing["models"][key]
            assert p["rub_per_1m_in"] is not None and p["rub_per_1m_out"] is not None and p["date_checked"]
            assert p.get("local") or p["source_url"].startswith("https://")
            if cfg["provider"] == "ollama":
                assert p["rub_per_1m_in"] == p["rub_per_1m_out"] == 0
    # USD-цены переведены по записанному курсу.
    ds = pricing["models"]["deepseek-v4-pro"]
    assert ds["rub_per_1m_in"] == pytest.approx(ds["original"]["in_cache_miss"] * pricing["usd_rub"]["rate"], rel=1e-4)


# ---------- Сквозной mock-прогон ----------

def test_end_to_end_demo(env):
    assert run_bank.main(bank_args()) == 0
    assert exec_code.main(["--demo", "--bank", str(FIXTURE_BANK)]) == 0
    assert autograde.main(["--demo", "--bank", str(FIXTURE_BANK)]) == 0
    assert blind.main(["--demo", "--bank", str(FIXTURE_BANK)]) == 0
    demo = env / "demo"
    graded = run_bank.read_jsonl(demo / "results" / "graded_auto.jsonl")
    assert {r["type"] for r in graded} == {"A", "B", "D", "F"}
    sheet = autograde.read_csv(demo / "grading" / "sheet_r1.csv")
    key = autograde.read_csv(demo / "grading" / "key_DO_NOT_OPEN.csv")
    assert len(sheet) == len(key) == 4 * 4          # (B1, B2, C1, D1) × 4 mock-модели
    assert not any("mock-ru1" in r["answer_text"] for r in sheet)
    with open(demo / "grading" / "sheet_r1.csv", encoding="utf-8") as f:
        assert f.read(1) == "﻿"                # BOM: Excel открывает кириллицу
    assert checks.g2(demo / "grading", llm.ROOT / "config" / "pricing.json") == 1   # листы пустые
