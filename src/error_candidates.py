"""Кандидаты на «интересные ошибки» для раздела 4 отчёта → report/error_candidates.md (шаг 12).

Отбор только по правилам, без оценки причин (разбор пишет человек):
  high_conf_wrong   неверный ответ A/B/F при уверенности ≥ HIGH_CONF;
  flip              верный ответ стал неверным после «Мне кажется, ты ошибся» (H9);
  invented_number   число в задаче с недостающими данными (эталон «нельзя определить», H3);
  textbook_answer   в изменённой задаче (F, modified) ответ совпал с эталоном оригинала из учебника (H1);
  code_method       код D с методологической ошибкой: по оценкам проверяющих (≥ 1 из 2 отметили пункт);
  code_failed       код D не запустился (pass@1 = 0);
  c_zero            выбор метода C на 0 баллов у обоих проверяющих;
  e_confident_miss  тип E: пропущенный отказ при уверенности ≥ HIGH_CONF.
Из каждой категории берётся до QUOTA[категория] случаев (по возможности разные модели и задачи), затем список
добирается остатками до N_MAX. Цитата — точный текст ответа из logs/calls.jsonl по call_id (длинный ответ —
начало и конец, пропуск отмечен).

Реальный режим требует пройденный G2: до конца слепой проверки нельзя показывать ответы с названиями моделей.

  .venv/bin/python -m src.error_candidates [--lang ru] [--demo] [--force]
"""

import argparse
import contextlib
import io
import sys
from pathlib import Path

import pandas as pd

from src import analysis as an
from src import bank, blind, checks, grade, run_bank, typeE, typeE_metrics
from src.build_report import fence

HIGH_CONF = 90          # «высокая уверенность», %
N_MAX = 15
QUOTE_MAX = 2500        # длиннее — цитируются начало и конец
QUOTE_HEAD, QUOTE_TAIL = 700, 1500
QUOTA = {"high_conf_wrong": 3, "flip": 2, "invented_number": 2, "textbook_answer": 2, "code_method": 2,
         "code_failed": 1, "c_zero": 1, "e_confident_miss": 1}
TITLES = {
    "high_conf_wrong": f"Неверный ответ при уверенности ≥ {HIGH_CONF}",
    "flip": "Переворот верного ответа после «Мне кажется, ты ошибся»",
    "invented_number": "Число в задаче с недостающими данными",
    "textbook_answer": "В изменённой задаче ответ совпал с эталоном оригинала из учебника",
    "code_method": "Код D с методологической ошибкой (по оценкам проверяющих)",
    "code_failed": "Код D не запустился",
    "c_zero": "Выбор метода (C) на 0 баллов у обоих проверяющих",
    "e_confident_miss": f"Тип E: пропущенный отказ при уверенности ≥ {HIGH_CONF}",
}


def out_path(demo: bool) -> Path:
    return (run_bank.DEMO_DIR if demo else run_bank.ROOT) / "report" / "error_candidates.md"


def g2_passed() -> bool:
    """checks.g2() без вывода на экран."""
    with contextlib.redirect_stdout(io.StringIO()):
        return checks.g2() == 0


def cand(category: str, row, priority: float, **extra) -> dict:
    """Кандидат: строка ответа (Series или dict) + категория; priority — чем больше, тем раньше в списке."""
    r = dict(row)
    return {"category": category, "task_id": r.get("task_id"), "model_key": r.get("model_key"),
            "experiment": r.get("experiment"), "lang": r.get("lang"), "call_id": r.get("call_id"),
            "value": r.get("value"), "confidence": r.get("confidence"), "reason": r.get("reason"),
            "priority": priority, **extra}


def wrong(x) -> bool:
    return an.binary(x) and x == 0


def high_conf_wrong(main: pd.DataFrame) -> list[dict]:
    df = main[main["type"].isin(["A", "B", "F"]) & main["correct"].apply(wrong)
              & (pd.to_numeric(main["confidence"], errors="coerce") >= HIGH_CONF)]
    return [cand("high_conf_wrong", r, r["confidence"]) for _, r in df.iterrows()]


def flips(main: pd.DataFrame, follow: pd.DataFrame) -> list[dict]:
    if follow.empty or main.empty:
        return []
    first = main.set_index("call_id")["correct"]
    f = follow[follow["first_call_id"].map(first).eq(1) & follow["correct"].apply(wrong)]
    return [cand("flip", r, r["confidence"] if pd.notna(r["confidence"]) else 0,
                 quotes=[("первый ответ (верный)", r["first_call_id"]), ("ответ после реплики", r["call_id"])])
            for _, r in f.iterrows()]


def invented_numbers(main: pd.DataFrame) -> list[dict]:
    df = main[(main["answer_kind"] == "cannot_determine") & main["value"].apply(an.is_number_answer)]
    return [cand("invented_number", r, r["confidence"] if pd.notna(r["confidence"]) else 0)
            for _, r in df.iterrows()]


def same_answer(value, reference, tol: float) -> bool:
    """Ответ совпал с эталоном (число или список чисел) в пределах допуска."""
    if isinstance(reference, list):
        return (isinstance(value, list) and len(value) == len(reference)
                and all(grade.close(v, ref, tol) for v, ref in zip(value, reference)))
    return an.is_number_answer(value) and not isinstance(value, list) and grade.close(value, reference, tol)


def textbook_answers(main: pd.DataFrame, tasks: dict) -> list[dict]:
    originals = {t["pair_id"]: t for t in tasks.values() if t.get("variant") == "original" and t.get("pair_id")}
    out = []
    for _, r in main[(main["task_variant"] == "modified") & main["correct"].apply(wrong)].iterrows():
        orig = originals.get(r["pair_id"])
        if orig is None or not isinstance(orig.get("reference"), (int, float, list)):
            continue
        if same_answer(r["value"], orig["reference"], orig.get("tolerance") or 0.01):
            out.append(cand("textbook_answer", r, r["confidence"] if pd.notna(r["confidence"]) else 0,
                            note=f"эталон оригинала {orig['id']}: {an_fmt(orig['reference'])}"))
    return out


def code_method(main: pd.DataFrame) -> list[dict]:
    d = main[(main["type"] == "D") & (pd.to_numeric(main["any_error_mean"], errors="coerce") >= 0.5)]
    out = []
    for _, r in d.iterrows():
        items = [blind.CHECKLIST[c] for c in blind.CHECKLIST if (r.get(f"{c}_mean") or 0) >= 0.5]
        out.append(cand("code_method", r, r["any_error_mean"],
                        note="отмечено хотя бы одним проверяющим: " + "; ".join(items)))
    return out


def code_failed(main: pd.DataFrame, d_exec: dict) -> list[dict]:
    d = main[(main["type"] == "D") & (main["pass_at_1"] == False)]  # noqa: E712 (NaN не считается)
    out = []
    for _, r in d.iterrows():
        rec = d_exec.get(r["call_id"], {})
        err = [x for x in (rec.get("stderr") or "").splitlines() if x.strip()]
        what = "таймаут" if rec.get("timed_out") else ("код не найден" if rec.get("code_found") is False
                                                       else f"код возврата {rec.get('returncode')}")
        out.append(cand("code_failed", r, 0, note=f"{what}; последняя строка stderr: {err[-1] if err else '—'}"))
    return out


def c_zero(main: pd.DataFrame) -> list[dict]:
    df = main[(main["type"] == "C") & (pd.to_numeric(main["score_mean"], errors="coerce") == 0)]
    return [cand("c_zero", r, r["confidence"] if pd.notna(r["confidence"]) else 0) for _, r in df.iterrows()]


def e_misses(results_dir: Path, lang: str) -> list[dict]:
    split = typeE.load_split()
    rows = typeE_metrics.llm_rows(run_bank.read_jsonl(results_dir / typeE_metrics.LLM_NAME), split["test"],
                                  typeE.split_hash(split))
    data = typeE.load_data()
    out = []
    for r in rows:
        conf = r.get("confidence")
        if (r["error"] is None and r["lang"] == lang and r["y_true"] == 1 and r["value"] == 0
                and conf is not None and conf >= HIGH_CONF):
            out.append(cand("e_confident_miss", r, conf, row_text=typeE.row_text(data.iloc[r["row_idx"]], lang),
                            note=f"{r['shots']} примеров в промпте, строка {r['row_idx']} (UDI {r.get('udi')})"))
    return out


def select(pools: dict[str, list[dict]]) -> list[dict]:
    """По квотам из каждой категории, затем добор до N_MAX. Внутри категории — по priority, модели по кругу;
    одна и та же пара (модель, задача) не повторяется."""
    used, chosen = set(), []

    def take(pool: list[dict], k: int) -> None:
        pool = sorted(pool, key=lambda c: (-float(c["priority"] or 0), str(c["task_id"]), str(c["model_key"])))
        by_model: dict[str, list[dict]] = {}
        for c in pool:
            by_model.setdefault(c["model_key"], []).append(c)
        while k > 0 and any(by_model.values()):
            for m in list(by_model):
                while by_model[m] and (by_model[m][0]["model_key"], by_model[m][0]["task_id"]) in used:
                    by_model[m].pop(0)
                if by_model[m] and k > 0:
                    c = by_model[m].pop(0)
                    used.add((c["model_key"], c["task_id"]))
                    chosen.append(c)
                    k -= 1

    for cat, k in QUOTA.items():
        take(pools.get(cat, []), k)
    for cat in QUOTA:
        take(pools.get(cat, []), N_MAX - len(chosen))
    return chosen


def an_fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list):
        return "; ".join(an_fmt(x) for x in v)
    return str(v)


def short_value(v) -> str:
    """Значение из разбора; код D и текст C длинные — они и так есть в цитате."""
    if isinstance(v, str) and ("\n" in v or len(v) > 80):
        return "(текст ответа — см. цитату)"
    return an_fmt(v)


def quote(text: str | None) -> str:
    """Точная цитата ответа; у длинного — начало и конец с пометкой о пропуске."""
    if text is None:
        return "(в журнале нет текста ответа)"
    if len(text) > QUOTE_MAX:   # два точных фрагмента, пометка о пропуске — вне цитаты
        skipped = len(text) - QUOTE_HEAD - QUOTE_TAIL
        return (f"Начало ответа:\n\n{fence(text[:QUOTE_HEAD])}\n\n_[… пропущено {skipped} символов, полный "
                f"текст — в журнале по call_id …]_\n\nКонец ответа:\n\n{fence(text[-QUOTE_TAIL:])}")
    return fence(text)


def render(chosen: list[dict], tasks: dict, log: dict, demo: bool, lang: str) -> str:
    lines = ["# Кандидаты на разбор ошибок (раздел 4 отчёта)", ""]
    if demo:
        lines += ["> **ДЕМО: mock-модели и случайные оценки. Это не результаты эксперимента.**", ""]
    lines += [
        "Собрано `.venv/bin/python -m src.error_candidates` по `results/` и `logs/calls.jsonl`; руками не править. "
        f"Правила отбора — docstring `src/error_candidates.py` (уверенность ≥ {HIGH_CONF}, до {N_MAX} случаев, "
        "разные модели и задачи). Здесь только факты: причины ошибок и выбор пяти для отчёта — за человеком.",
        "",
        "Сводка по категориям: " + (", ".join(f"{TITLES[c]} — {sum(x['category'] == c for x in chosen)}"
                                              for c in QUOTA if any(x["category"] == c for x in chosen))
                                    or "кандидатов нет"),
    ]
    for n, c in enumerate(chosen, 1):
        rec = log.get(c["call_id"], {})
        lines += ["", f"## {n}. {TITLES[c['category']]} — {c['task_id']}, {c['model_key']}", ""]
        if c["category"] == "e_confident_miss":   # строка теста AI4I, в банке задач её нет
            lines += [f"- Задача: `{c['task_id']}` — строка теста AI4I 2020 (тип E). Строка в промпте: "
                      f"«{c['row_text']}»"]
            reference = "отказ (1)"
        else:
            t = tasks.get(c["task_id"], {})
            text = (t.get(f"text_{lang}") or "").strip()
            lines += [f"- Задача: `{c['task_id']}` (тип {t.get('type')}, тема {t.get('topic')}, источник "
                      f"{t.get('source')}, вариант {t.get('variant') or '—'}). "
                      f"Условие: «{text[:400]}{'…' if len(text) > 400 else ''}»"]
            reference = an_fmt(t.get("reference"))
            if t.get("tolerance") is not None:
                reference += f" (допуск {t['tolerance']})"
            if t.get("reference_metric") is not None:
                reference += f"; эталонная метрика {t['reference_metric']}"
        lines += [f"- Модель: `{c['model_key']}`, model_reported `{rec.get('model_reported')}`, "
                  f"вызов {str(rec.get('ts_utc'))[:19]} UTC; эксперимент `{c['experiment']}`, язык {c['lang']}",
                  f"- Эталон: {reference}",
                  f"- Разбор ответа: значение {short_value(c['value'])}, уверенность {an_fmt(c['confidence'])}, "
                  f"оценка: {c.get('reason') or '—'}"]
        if c.get("note"):
            lines.append(f"- Отмечено: {c['note']}")
        for label, cid in c.get("quotes") or [("ответ модели", c["call_id"])]:
            lines += [f"- call_id ({label}): `{cid}`", "", quote(log.get(cid, {}).get("response_text"))]
    return "\n".join(lines) + "\n"


def build(demo: bool, lang: str = "ru") -> tuple[str, int]:
    p = an.paths(demo)
    tasks = {t["id"]: t for t in bank.load_tasks(bank.BANK_PATH)}
    manual = an.load_manual(p["grading"])
    graded = an.read_table(p["results"], "graded_auto")
    main, _ = an.answers(p["results"], "main", lang, manual, graded)
    follow, _ = an.answers(p["results"], "followup_doubt", lang, manual, graded)
    d_exec = {r["call_id"]: r for r in run_bank.read_jsonl(p["results"] / "d_exec.jsonl")}
    pools = {"e_confident_miss": e_misses(p["results"], lang)}
    if not main.empty:
        pools |= {"high_conf_wrong": high_conf_wrong(main), "flip": flips(main, follow),
                  "invented_number": invented_numbers(main), "textbook_answer": textbook_answers(main, tasks),
                  "code_method": code_method(main), "code_failed": code_failed(main, d_exec),
                  "c_zero": c_zero(main)}
    chosen = select(pools)
    ids = {c["call_id"] for c in chosen} | {cid for c in chosen for _, cid in c.get("quotes") or []}
    log = {r["call_id"]: r for r in run_bank.read_jsonl(p["log"]) if r.get("call_id") in ids}
    return render(chosen, tasks, log, demo, lang), len(chosen)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Кандидаты на разбор ошибок → report/error_candidates.md")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--force", action="store_true", help="не проверять G2 (слепая проверка уже не нужна)")
    args = ap.parse_args(argv)
    if not args.demo and not args.force and not g2_passed():
        print("G2 не пройден (src.checks g2): до конца слепой проверки ответы с названиями моделей не показываем.")
        return 1
    text, n = build(args.demo, args.lang)
    path = out_path(args.demo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Записано: {run_bank.show(path)}; кандидатов {n}.")
    return 0 if n else 2


if __name__ == "__main__":
    sys.exit(main())
