"""Тесты src/human_docs.py без печати в PDF (Chrome не нужен)."""

from src import human_docs as hd


def test_reference_formatting():
    assert "нельзя определить" in hd.fmt_reference({"type": "F", "reference": "cannot_determine"})
    assert "неверно" in hd.fmt_reference({"type": "B", "reference": False})
    a = hd.fmt_reference({"type": "A", "reference": [0.1538, 1], "tolerance": 0.01})
    assert "0,1538; 1" in a and "±1%" in a
    d = hd.fmt_reference({"type": "D", "reference": None, "reference_metric": 0.9556, "tolerance": 0.05})
    assert "0,9556" in d and "±5%" in d


def test_human_blocks_visible_and_h1_stripped():
    html = hd.md(hd.strip_h1("# Заголовок\n\nТекст\n<!-- ЧЕЛОВЕК:intro — введение -->\n<!-- /ЧЕЛОВЕК:intro -->\n"))
    assert "Заголовок" not in html and "Заполняет человек" in html and "введение" in html


def test_bank_document_has_every_task():
    _, _, body = hd.doc_bank()
    for t in hd.read_jsonl("data/tasks.jsonl") + hd.read_jsonl("data/tasks_dev.jsonl"):
        assert f"<h3>{t['id']}</h3>" in body
