"""Тесты src/bank.py: схема задачи, строгая проверка --final, сверка эталонов. Сеть не нужна."""

import copy
import json

from src import bank


def make_task(**over):
    """Минимальная корректная задача типа A в статусе verified."""
    t = {"id": "X1", "type": "A", "topic": "bayes", "source": "свои", "difficulty": 2, "answer_kind": "number",
         "variant": None, "pair_id": None, "text_ru": "Условие", "text_en": "Problem", "reference": 0.5,
         "tolerance": 0.01, "reference_status": "verified", "rubric": None, "reference_code": None,
         "reference_metric": None, "notes": ""}
    t.update(over)
    return t


RUBRIC_B = {"key_arguments": ["a"], "typical_errors": ["b"], "score_2": "x", "score_1": "y", "score_0": "z"}
RUBRIC_C = {"acceptable_methods": ["m"], "required_arguments": ["a"], "typical_errors": ["e"],
            "score_2": "x", "score_1": "y", "score_0": "z"}


def final_bank():
    """Банк, который должен пройти --final: 30 задач, A/B/C/D/F по 6, все темы, половина своих."""
    tasks = []
    topics = bank.TOPICS
    for i in range(6):
        tasks.append(make_task(id=f"A{i}", topic=topics[i % 10], source="учебник"))
        tasks.append(make_task(id=f"B{i}", type="B", topic=topics[(i + 1) % 10], answer_kind="bool",
                               reference=i % 2 == 0, tolerance=None, rubric=RUBRIC_B, source="учебник"))
        tasks.append(make_task(id=f"C{i}", type="C", topic=topics[(i + 2) % 10], answer_kind="text",
                               reference="метод", tolerance=None, rubric=RUBRIC_C))
        tasks.append(make_task(id=f"D{i}", type="D", topic=topics[(i + 3) % 10], answer_kind="code", reference=None,
                               tolerance=0.02, reference_code="data/d_reference/D1.py", reference_metric=0.97))
        tasks.append(make_task(id=f"F{i}", topic=topics[(i + 4) % 10], type="F"))
    # Пара H1: оригинал A0 ↔ изменённая F0.
    tasks[0].update(pair_id="P1", variant="original")
    tasks[4].update(pair_id="P1", variant="modified", reference=0.7)
    return tasks


def errors_of(tasks, dev=(), **kw):
    return bank.validate(tasks, list(dev), **kw)[0]


def test_real_bank_passes_schema_check():
    """Черновик банка в репозитории проходит проверку схемы (без --final)."""
    errors, warnings = bank.validate_files(final=False)
    assert errors == []
    assert any("todo" in w for w in warnings)


def test_real_bank_composition():
    tasks = bank.load_tasks(bank.BANK_PATH)
    counts = {tp: sum(t["type"] == tp for t in tasks) for tp in "ABCDF"}
    assert len(tasks) == 38 and counts == {"A": 8, "B": 8, "C": 6, "D": 6, "F": 10}
    assert sum(t["source"] == "свои" for t in tasks) * 2 >= len(tasks)
    assert {t["topic"] for t in tasks} == set(bank.TOPICS)
    dev = bank.load_tasks(bank.DEV_PATH)
    assert sorted(t["type"] for t in dev) == ["A", "B", "C", "D", "F"]


def test_valid_final_bank_passes():
    assert errors_of(final_bank(), final=True) == []


def test_missing_field_and_bad_enum():
    t = make_task()
    del t["tolerance"]
    assert any("нет полей tolerance" in e for e in errors_of([t]))
    assert any("topic=" in e for e in errors_of([make_task(topic="nn")]))
    assert any("difficulty" in e for e in errors_of([make_task(difficulty=5)]))


def test_reference_must_match_answer_kind():
    assert any("числом" in e for e in errors_of([make_task(reference="0.5")]))
    assert any("списком" in e for e in errors_of([make_task(answer_kind="numbers", reference=[1])]))
    bad_b = make_task(type="B", answer_kind="bool", reference="верно", tolerance=None, rubric=RUBRIC_B)
    assert any("true/false" in e for e in errors_of([bad_b]))
    # answer_kind не подходит для типа.
    assert any("не подходит для типа" in e for e in errors_of([make_task(answer_kind="bool", reference=True)]))


def test_todo_stub_passes_schema_but_not_final():
    stub = make_task(text_ru="TODO(человек): придумать", text_en="TODO", reference=None, reference_status="todo")
    assert errors_of([stub]) == []
    bank_ = final_bank()
    bank_[1] = {**stub, "id": bank_[1]["id"], "type": "B", "answer_kind": "bool", "tolerance": None}
    errs = errors_of(bank_, final=True)
    assert any("verified" in e for e in errs) and any("TODO" in e for e in errs)


def test_verified_task_cannot_contain_todo():
    t = make_task(type="C", answer_kind="text", reference="метод", tolerance=None,
                  rubric={**RUBRIC_C, "score_1": "TODO(человек)"})
    assert any("TODO" in e for e in errors_of([t]))


def test_missing_data_requires_cannot_determine():
    t = make_task(type="F", variant="missing_data", answer_kind="number")
    assert any("missing_data" in e for e in errors_of([t]))
    ok = make_task(type="F", variant="missing_data", answer_kind="cannot_determine", reference="cannot_determine",
                   tolerance=None)
    assert errors_of([ok]) == []


def test_pairs_need_original_and_other_half():
    tasks = final_bank()
    tasks[4].update(pair_id="P2")   # F0 больше не в паре с A0
    errs = errors_of(tasks)
    assert any("пара P1" in e for e in errs) and any("пара P2" in e for e in errs)


def test_duplicate_ids_and_dev_overlap():
    a, b = make_task(id="A1"), make_task(id="A1")
    assert any("повторяются" in e for e in errors_of([a, b]))
    dev = make_task(id="A1")
    assert any("dev-задачами" in e for e in errors_of([make_task(id="A1")], dev=[dev]))
    # dev-задача: todo нельзя никогда; черновик агента — можно до гейта G1, но не в --final.
    assert any("dev" in e for e in errors_of([make_task()], dev=[make_task(id="dev-1", reference_status="todo")]))
    draft = [make_task(id="dev-1", reference_status="agent_draft")]
    assert not any("dev" in e for e in errors_of([make_task()], dev=draft))
    assert any("dev-1" in e for e in errors_of([make_task()], dev=draft, final=True))


def test_final_rules():
    tasks = final_bank()
    assert any("нужно 30–40" in e for e in errors_of(tasks[:25], final=True))
    no_c = [t for t in tasks if t["type"] != "C"] + [make_task(id=f"Z{i}") for i in range(6)]
    assert any("типа C" in e for e in errors_of(no_c, final=True))
    # Все утверждения B ложные — нарушение баланса.
    unbalanced = copy.deepcopy(tasks)
    for t in unbalanced:
        if t["type"] == "B":
            t["reference"] = False
    assert any("ложных" in e for e in errors_of(unbalanced, final=True))
    # Сдвиг баланса на одну задачу тоже ошибка: в задании «половина», а не «примерно половина».
    off_by_one = copy.deepcopy(tasks)
    b_tasks = [t for t in off_by_one if t["type"] == "B"]
    n_false = sum(t["reference"] is False for t in b_tasks)
    assert n_false * 2 == len(b_tasks)              # в final_bank ровно половина
    next(t for t in b_tasks if t["reference"] is True)["reference"] = False
    assert any("ложных" in e for e in errors_of(off_by_one, final=True))
    # Своих меньше половины.
    textbook = copy.deepcopy(tasks)
    for t in textbook:
        t["source"] = "учебник"
    assert any("своих" in e for e in errors_of(textbook, final=True))
    # Не заполнен text_en.
    no_en = copy.deepcopy(tasks)
    no_en[0]["text_en"] = ""
    assert any("text_en" in e for e in errors_of(no_en, final=True))
    # Агентский черновик не проходит --final.
    draft = copy.deepcopy(tasks)
    draft[0]["reference_status"] = "agent_draft"
    assert any("verified" in e for e in errors_of(draft, final=True))


def test_final_depends_on_hypotheses():
    tasks = final_bank()
    assert errors_of(tasks, final=True, hypotheses=["H5", "H10", "H12"]) == []
    assert any("H1:" in e for e in errors_of(tasks, final=True, hypotheses=["H1"]))
    assert any("H8:" in e for e in errors_of(tasks, final=True, hypotheses=["H8"]))
    assert any("H2:" in e for e in errors_of(tasks, final=True, hypotheses=["H2"]))


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "t.jsonl"
    tasks = final_bank()
    bank.save_tasks(tasks, path)
    assert bank.load_tasks(path) == tasks
    assert "свои" in path.read_text(encoding="utf-8")   # кириллица без \u-экранирования


def test_load_reports_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(make_task()) + "\n{битый\n", encoding="utf-8")
    try:
        bank.load_tasks(path)
        assert False, "ожидалась ошибка"
    except ValueError as e:
        assert ":2:" in str(e)


def test_parse_marker_and_close():
    assert bank.parse_marker("шум\nREFERENCE: 0.5; 2\n", "REFERENCE") == [0.5, 2.0]
    assert bank.parse_marker("METRIC: 0.97", "REFERENCE") is None
    assert bank.close(0.161, 0.1610, 0.01) and not bank.close(0.17, 0.161, 0.01)


def test_refs_scripts_match_bank():
    """Эталоны A и dev-A в банке совпадают с тем, что считают скрипты data/refs/*.py (без D: они долгие)."""
    for path in (bank.BANK_PATH, bank.DEV_PATH):
        for t in bank.load_tasks(path):
            script = bank.REFS_DIR / f"{t['id']}.py"
            if script.exists():
                got = bank.parse_marker(bank.run_script(script)[0], "REFERENCE")
                expected = t["reference"] if isinstance(t["reference"], list) else [t["reference"]]
                assert all(bank.close(g, e, t["tolerance"]) for g, e in zip(got, expected)), t["id"]


def test_stats_tables():
    tables = bank.stats_tables(final_bank())
    assert tables["тип × тема"].loc["всего", "всего"] == 30


def test_cli_validate_exit_codes(capsys):
    assert bank.main(["validate"]) == 0
    assert bank.main(["validate", "--final"]) == 1   # в черновике есть заготовки todo
    assert "ОШИБКА" in capsys.readouterr().out


def test_id_must_be_string():
    errs = errors_of([make_task(id=None), make_task(id=5)])
    assert sum("id должен быть" in e for e in errs) == 2
    no_id = make_task()
    del no_id["id"]
    assert any("id должен быть" in e for e in errors_of([no_id], final=True))   # без падения KeyError


def test_close_with_zero_reference():
    assert bank.close(1e-12, 0.0, 0.01) and not bank.close(0.01, 0.0, 0.01)
    assert bank.parse_marker("REFERENCE: 1; 2;", "REFERENCE") == [1.0, 2.0]


def test_modified_must_change_answer():
    tasks = final_bank()
    tasks[4]["reference"] = tasks[0]["reference"]
    assert any("совпадает с оригиналом" in e for e in errors_of(tasks))


def test_todo_stub_may_have_no_tolerance():
    stub = make_task(type="D", answer_kind="code", reference=None, tolerance=None, reference_status="todo",
                     text_ru="TODO(человек)")
    assert errors_of([stub]) == []
    assert any("tolerance" in e for e in errors_of([{**stub, "reference_status": "agent_draft",
                                                       "text_ru": "Условие"}]))


def test_h2_needs_formula_and_verbal():
    tasks = final_bank()
    for t in tasks[:5]:
        t["verbal_ru"] = "словами"
    assert any("H2:" in e for e in errors_of(tasks, final=True, hypotheses=["H2"]))
    for t in tasks[:5]:
        t["formula_ru"] = "P(D) = 0,01"
    assert errors_of(tasks, final=True, hypotheses=["H2"]) == []
    assert any("H2 + en" in e for e in errors_of(tasks, final=True, hypotheses=["H2"], languages=["ru", "en"]))


def test_verify_marks_only_agent_drafts(tmp_path):
    path = tmp_path / "tasks.jsonl"
    tasks = bank.load_tasks()
    bank.save_tasks(tasks, path)
    todo_id = next(t["id"] for t in tasks if t["reference_status"] == "todo")
    draft_id = next(t["id"] for t in tasks if t["reference_status"] == "agent_draft")

    assert bank.main(["verify", "--bank", str(path), todo_id]) == 1       # заготовку нельзя «подтвердить»
    assert bank.main(["verify", "--bank", str(path), draft_id]) == 0
    status = {t["id"]: t["reference_status"] for t in bank.load_tasks(path)}
    assert status[draft_id] == "verified" and status[todo_id] == "todo"
