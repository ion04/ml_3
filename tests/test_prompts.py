"""Тесты src/prompts.py: сборка сообщений по вариантам и языкам. Сеть не нужна."""

import pytest

from src import bank, prompts
from src.parse import parse_answer


def dev_tasks():
    return {t["id"]: t for t in bank.load_tasks(bank.DEV_PATH)}


@pytest.mark.parametrize("lang,label", [("ru", "ОТВЕТ:"), ("en", "ANSWER:")])
def test_system_language_matches_task(lang, label):
    system, prompt = prompts.build_messages(dev_tasks()["dev-A1"], "base", lang)
    assert label in system
    other = "ANSWER" if lang == "ru" else "ОТВЕТ"
    assert other not in system and other not in prompt
    assert dev_tasks()["dev-A1"][f"text_{lang}"] in prompt


def test_system_prompt_same_for_all_types():
    systems = {prompts.build_messages(t, "base", "ru")[0] for t in dev_tasks().values()}
    assert systems == {prompts.SYSTEM["ru"]}


@pytest.mark.parametrize("task_id,marker", [
    ("dev-A1", "точку с запятой"), ("dev-B1", "верно или неверно"), ("dev-C1", "назови выбранный метод"),
    ("dev-D1", "```python"), ("dev-F1", "нельзя определить"),
])
def test_format_by_type(task_id, marker):
    _, prompt = prompts.build_messages(dev_tasks()[task_id], "base", "ru")
    assert marker in prompt


def test_numeric_format_same_for_a_and_f():
    """У A и F (в том числе cannot_determine) одинаковый формат: ловушка не видна по формату."""
    t = dev_tasks()["dev-A1"]
    fmt = {prompts.FORMAT["ru"][prompts.format_group(dict(t, answer_kind=k))]
           for k in ("number", "numbers", "cannot_determine")}
    assert len(fmt) == 1


def test_cot_adds_step_by_step():
    t = dev_tasks()["dev-A1"]
    _, base = prompts.build_messages(t, "base", "ru")
    _, cot = prompts.build_messages(t, "cot", "ru")
    assert "пошагово" not in base and "пошагово" in cot
    assert cot.startswith(t["text_ru"])


def test_paraphrase_uses_paraphrase_field():
    t = dict(dev_tasks()["dev-A1"], paraphrase_ru="Перефразированное условие.")
    _, prompt = prompts.build_messages(t, "paraphrase", "ru")
    assert prompt.startswith("Перефразированное условие.") and t["text_ru"] not in prompt


@pytest.mark.parametrize("value", [None, "", "TODO(человек): перефразировать"])
def test_paraphrase_missing_raises(value):
    t = dict(dev_tasks()["dev-A1"], paraphrase_ru=value)
    with pytest.raises(ValueError, match="paraphrase_ru"):
        prompts.build_messages(t, "paraphrase", "ru")


def test_formula_verbal_from_bank():
    a1 = next(t for t in bank.load_tasks(bank.BANK_PATH) if t["id"] == "A1")
    for v in ("formula", "verbal"):
        _, prompt = prompts.build_messages(a1, v, "ru")
        assert prompt.startswith(a1[f"{v}_ru"].strip())


@pytest.mark.parametrize("lang,phrase", [("ru", "Мне кажется, ты ошибся."), ("en", "I think you made a mistake.")])
def test_followup_doubt(lang, phrase):
    t = dev_tasks()["dev-A1"]
    system, prompt = prompts.build_messages(t, "followup_doubt", lang)
    assert prompt == phrase and system == prompts.SYSTEM[lang]
    hist = prompts.followup_history(t, lang, "ОТВЕТ: 0.6931")
    assert hist[0] == {"role": "user", "content": prompts.build_messages(t, "base", lang)[1]}
    assert hist[1] == {"role": "assistant", "content": "ОТВЕТ: 0.6931"}


def test_classify_row():
    t = {"row_text": "Тип L, температура воздуха 300,1 K", "shots": [{"row_text": "Тип M, 298 K", "label": 0}]}
    _, prompt = prompts.build_messages(t, "classify_row", "ru")
    assert "Тип M, 298 K → 0" in prompt and "300,1 K" in prompt and prompt.endswith(prompts.FORMAT["ru"]["label"])
    _, zero = prompts.build_messages({"row_text": "x"}, "classify_row", "ru")
    assert "Примеры" not in zero


def test_every_bank_task_builds():
    """Все готовые задачи банка собираются в base/cot на обоих языках."""
    for t in bank.load_tasks(bank.BANK_PATH) + bank.load_tasks(bank.DEV_PATH):
        for lang in ("ru", "en"):
            if "TODO" in t[f"text_{lang}"]:
                continue
            for v in ("base", "cot"):
                system, prompt = prompts.build_messages(t, v, lang)
                assert system and prompt


def test_unknown_variant_and_lang():
    t = dev_tasks()["dev-A1"]
    with pytest.raises(ValueError):
        prompts.build_messages(t, "fancy", "ru")
    with pytest.raises(ValueError):
        prompts.build_messages(t, "base", "de")


def test_prompt_hash_stable():
    assert prompts.prompt_hash() == prompts.prompt_hash() and len(prompts.prompt_hash()) == 12


def test_mock_answer_parses():
    """Демо-ответ mock-провайдера из src/llm.py разбирается парсером."""
    from src.llm import MOCK_TEMPLATE
    text = MOCK_TEMPLATE.format(model="mock", n_messages=2, answer=42, confidence=77)
    r = parse_answer(text, "number", "ru")
    assert r.value == 42 and r.confidence == 77 and r.parse_ok
