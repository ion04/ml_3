"""Человекочитаемые PDF для проверки человеком → папка human/.

Каждый документ собирается из файлов проекта: банк задач, гипотезы, манифест прогонов, листы проверки и т. д.
Руками в документы ничего не вписывается. После изменения данных достаточно пересобрать.
HTML печатается в PDF через Google Chrome в headless-режиме. Markdown превращается в HTML пакетом mistune:
он уже стоит вместе с Jupyter.

Запуск:
  .venv/bin/python -m src.human_docs           # все документы
  .venv/bin/python -m src.human_docs 02 07     # только выбранные (по номеру)
"""
import csv
import html
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import mistune

from src import prompts

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "human"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

TYPE_NAMES = {"A": "расчётная", "B": "верно / неверно", "C": "выбор метода", "D": "код sklearn",
              "E": "LLM как классификатор", "F": "ловушка"}
TOPIC_NAMES = {"linreg": "линейная регрессия", "logreg": "логистическая регрессия", "knn": "kNN",
               "trees": "деревья", "ensembles": "ансамбли", "svm": "SVM", "bayes": "байесовские методы",
               "clustering": "кластеризация", "pca": "PCA", "metrics": "метрики качества"}
VARIANT_NAMES = {"original": "учебниковый оригинал", "modified": "изменённая версия", "missing_data": "недостающие данные",
                 "distractor": "лишние данные"}

CSS = """
@page { size: A4; margin: 14mm 13mm 16mm 13mm; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; font-size: 10.5pt; line-height: 1.42; color: #1c1c1c; }
h1 { font-size: 19pt; margin: 0 0 2mm; } h2 { font-size: 14pt; margin: 7mm 0 2mm; border-bottom: 1px solid #ccc; padding-bottom: 1mm; }
h3 { font-size: 12pt; margin: 5mm 0 1.5mm; } h4 { font-size: 10.5pt; margin: 3mm 0 1mm; }
.sub { color: #555; margin-bottom: 5mm; font-size: 9.5pt; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 3mm; font-size: 9pt; page-break-inside: auto; }
th, td { border: 1px solid #bbb; padding: 1.2mm 1.6mm; vertical-align: top; text-align: left; }
th { background: #f0f0f0; } tr { page-break-inside: avoid; }
table.small { font-size: 7.6pt; } table.small th, table.small td { padding: 0.8mm 1mm; }
pre, code { font-family: Menlo, "Courier New", monospace; font-size: 8.3pt; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 2mm; white-space: pre-wrap; word-break: break-word; margin: 1mm 0 2mm; }
.card { border: 1px solid #999; border-radius: 2mm; padding: 2.5mm 3.5mm; margin: 0 0 4mm; page-break-inside: avoid; }
.card.long { page-break-inside: auto; }
.card h3 { margin-top: 0; } .meta { color: #555; font-size: 9pt; margin-bottom: 1.5mm; }
.lbl { font-weight: 600; color: #333; } .en { color: #555; font-size: 9.3pt; }
.ref { background: #eef6ee; border-left: 3px solid #4a9a4a; padding: 1.5mm 2.5mm; margin: 1.5mm 0; }
.check { margin-top: 2mm; padding-top: 1.5mm; border-top: 1px dashed #aaa; font-size: 9.5pt; }
.human { background: #fff6d6; border: 1px dashed #c9a400; padding: 1.5mm 2.5mm; margin: 2mm 0; }
.note { background: #f2f4fa; border-left: 3px solid #5a6fb0; padding: 1.5mm 2.5mm; margin: 2mm 0; font-size: 9.5pt; }
.pb { page-break-before: always; }
ul, ol { margin: 1mm 0 2mm 5mm; padding-left: 4mm; } li { margin: 0.4mm 0; }
"""

MD = mistune.create_markdown(escape=False, plugins=["table", "strikethrough", "task_lists"])


# ---------- мелкие помощники ----------

def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def pre(text) -> str:
    return f"<pre>{esc(text)}</pre>"


def md(text: str) -> str:
    """Markdown → HTML. Блоки <!-- ЧЕЛОВЕК:… --> в отчёте делаем видимыми: в PDF комментарии не видны."""
    text = re.sub(r"<!--\s*/ЧЕЛОВЕК:[^>]*-->", "", text)
    text = re.sub(r"<!--\s*ЧЕЛОВЕК:(.*?)-->",
                  lambda m: f'<div class="human">✍ <b>Заполняет человек:</b> {esc(m.group(1).strip())}</div>',
                  text, flags=re.S)
    return MD(text)


def table(header: list, rows: list, cls: str = "") -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table class='{cls}'><tr>{head}</tr>{body}</table>"


def strip_h1(text: str) -> str:
    """Первый заголовок «# …» в markdown дублирует заголовок PDF — убираем его."""
    return re.sub(r"\A\s*# [^\n]*\n", "", text)


def read_text(path) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in read_text(path).splitlines() if line.strip()]


def read_csv(path) -> list[dict]:
    with open(ROOT / path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def num(x, nd=4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return esc(x)
    return f"{v:.{nd}f}".rstrip("0").rstrip(".").replace(".", ",") if v != int(v) else str(int(v))


def page(title: str, sub: str, body: str) -> str:
    return (f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>{esc(title)}</title>'
            f"<style>{CSS}</style></head><body><h1>{esc(title)}</h1><div class=\"sub\">{sub}</div>{body}</body></html>")


def to_pdf(name: str, html_text: str) -> Path:
    OUT.mkdir(exist_ok=True)
    out = OUT / f"{name}.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "doc.html"
        src.write_text(html_text, encoding="utf-8")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={out}", src.as_uri()], check=True, capture_output=True, timeout=180)
    return out


def stamp() -> str:
    return f"Собрано из файлов проекта командой <code>.venv/bin/python -m src.human_docs</code> · коммит {esc(git('rev-parse', '--short', 'HEAD'))}"


# ---------- 02: банк задач ----------

def fmt_reference(t: dict) -> str:
    r = t["reference"]
    if t["type"] == "D":
        return f"метрика эталонного решения: <b>{num(t['reference_metric'])}</b>, допуск ±{num(t['tolerance'] * 100, 1)}% (относительный)"
    if t["type"] == "B":
        return "<b>верно</b>" if r else "<b>неверно</b>"
    if r == "cannot_determine":
        return "<b>нельзя определить</b> (правильный ответ — отказ дать число)"
    if t["type"] == "C":
        return esc(r)
    vals = r if isinstance(r, list) else [r]
    tol = f", допуск ±{num(t['tolerance'] * 100, 1)}% (относительный)" if t.get("tolerance") else ""
    return "<b>" + "; ".join(num(v) for v in vals) + f"</b>{tol}"


def fmt_rubric(t: dict) -> str:
    r = t.get("rubric") or {}
    if not r:
        return ""
    parts = []
    labels = [("acceptable_methods", "Допустимые методы"), ("key_arguments", "Ключевые аргументы"),
              ("required_arguments", "Обязательные аргументы (все нужны для 2 баллов)"),
              ("bonus_arguments", "Дополнительно (желательно, но не обязательно)"), ("typical_errors", "Типичные ошибки")]
    for key, title in labels:
        if r.get(key):
            parts.append(f"<div class='lbl'>{title}:</div><ul>" + "".join(f"<li>{esc(x)}</li>" for x in r[key]) + "</ul>")
    parts.append(table(["Балл", "Критерий"], [[s, esc(r.get(f"score_{s}", ""))] for s in ("2", "1", "0")]))
    return "<h4>Рубрика</h4>" + "".join(parts)


def task_card(t: dict) -> str:
    meta = [TYPE_NAMES.get(t["type"], t["type"]), TOPIC_NAMES.get(t["topic"], t["topic"]),
            f"источник: {t['source']}", f"сложность {t['difficulty']}", f"статус эталона: {t['reference_status']}"]
    if t.get("variant") in VARIANT_NAMES:
        meta.insert(2, VARIANT_NAMES[t["variant"]] + (f" (пара {t['pair_id']})" if t.get("pair_id") else ""))
    body = [f"<h3>{esc(t['id'])}</h3><div class='meta'>{' · '.join(esc(m) for m in meta)}</div>",
            f"<div><span class='lbl'>Условие:</span> {esc(t['text_ru'])}</div>",
            f"<div class='en'><span class='lbl'>EN:</span> {esc(t['text_en'])}</div>",
            f"<div class='ref'><span class='lbl'>Эталон:</span> {fmt_reference(t)}</div>"]
    body.append(fmt_rubric(t))
    script = ROOT / "data" / "refs" / f"{t['id']}.py"
    if script.exists():
        body.append(f"<h4>Расчёт эталона кодом (data/refs/{t['id']}.py)</h4>" + pre(script.read_text(encoding="utf-8")))
    if t.get("reference_code") and (ROOT / t["reference_code"]).exists():
        body.append(f"<h4>Эталонное решение ({esc(t['reference_code'])})</h4>" + pre(read_text(t["reference_code"])))
    if t.get("notes"):
        body.append(f"<div class='note'><span class='lbl'>Заметки:</span> {esc(t['notes'])}</div>")
    body.append("<div class='check'>☐ Решил(а) сам(а), эталон верен &nbsp;&nbsp; ☐ Есть замечание: "
                "________________________________________</div>")
    long = " long" if (t["type"] in ("C", "D") or script.exists()) else ""
    return f"<div class='card{long}'>{''.join(body)}</div>"


def doc_bank():
    tasks, dev = read_jsonl("data/tasks.jsonl"), read_jsonl("data/tasks_dev.jsonl")
    types = sorted({t["type"] for t in tasks})
    topics = list(TOPIC_NAMES)
    rows = []
    for tp in types:
        cells = [sum(1 for t in tasks if t["type"] == tp and t["topic"] == top) for top in topics]
        rows.append([f"{tp} — {TYPE_NAMES[tp]}", *[c or "" for c in cells], sum(cells)])
    rows.append(["всего", *[sum(1 for t in tasks if t["topic"] == top) for top in topics], len(tasks)])
    own = sum(t["source"] == "свои" for t in tasks)
    b = [t for t in tasks if t["type"] == "B"]
    intro = (f"<p>В банке {len(tasks)} задач: своих (поле <code>source</code>) — {own}, учебниковых — {len(tasks) - own}. "
             f"В типе B ложных утверждений {sum(t['reference'] is False for t in b)}, верных {sum(t['reference'] is True for t in b)}. "
             "Тип E — отдельный датасет AI4I 2020 (документ 05). Пары для H1: A1–A5 ↔ F1–F5 (общий <code>pair_id</code>).</p>"
             "<div class='note'>Как проверять: решите задачу сами, сверьте с эталоном, поставьте галочку. Замечания пишите на полях; "
             "исправления вносятся в <code>data/tasks.jsonl</code>. Важно: банк уже прогнан через модели, поэтому любая правка "
             "условия означает перепрогон этой задачи.</div>")
    body = intro + "<h2>Состав банка: тип × тема</h2>" + table(["тип", *[TOPIC_NAMES[t] for t in topics], "всего"], rows)
    for tp in types:
        body += f"<h2 class='pb'>Тип {tp} — {TYPE_NAMES[tp]}</h2>" + "".join(task_card(t) for t in tasks if t["type"] == tp)
    body += ("<h2 class='pb'>Dev-задачи (только для отладки промпта, в оценку не входят)</h2>"
             + "".join(task_card(t) for t in dev))
    return "02_банк_задач", "Банк задач ЛР3", body


# ---------- 03: гипотезы ----------

def guide_section(hid: str) -> str:
    text = read_text("docs/hypotheses_guide.md")
    m = re.search(rf"^### {hid}\..*?(?=^### |^## |\Z)", text, re.S | re.M)
    return md(m.group(0)) if m else ""


def doc_hypotheses():
    cfg = json.loads(read_text("config/experiment.json"))
    commit = git("log", "-1", "--format=%h · %ci · %s", "--", "hypotheses.md")
    log = read_jsonl("logs/calls.jsonl")
    ev = [r["ts_utc"] for r in log if (r.get("meta") or {}).get("experiment") in ("main", "cot", "followup_doubt", "typeE")]
    facts = table(["Факт", "Значение"], [
        ["Выбранные гипотезы", esc(", ".join(cfg["hypotheses"]))],
        ["Коммит с прогнозом (последний, менявший hypotheses.md)", esc(commit)],
        ["Первый оценочный вызов моделей (UTC)", esc(min(ev) if ev else "—")],
        ["Доп. прогоны H6", "режим cot для типов " + esc(", ".join(cfg["extra_runs"]["cot"]["types"]))
         + "; модели с режимом рассуждений: " + esc(", ".join(cfg["extra_runs"]["cot"].get("extra_models") or ["—"]))],
        ["Доп. прогоны H9", "реплика «" + esc(prompts.FOLLOWUP_DOUBT["ru"]) + "» к верно решённым задачам типов "
         + esc(", ".join(cfg["extra_runs"]["followup_doubt"]["types"]))],
    ])
    body = ("<div class='note'>Прогноз фиксируется коммитом до эксперимента. После коммита <code>hypotheses.md</code> менять нельзя: "
            "проверка G1 сравнивает время коммита с журналом вызовов.</div>" + facts
            + "<h2>hypotheses.md (как закоммичено)</h2>" + md(read_text("hypotheses.md"))
            + "<h2 class='pb'>Справка по выбранным гипотезам (docs/hypotheses_guide.md)</h2>"
            + "".join(guide_section(h) for h in cfg["hypotheses"]))
    return "03_гипотезы", "Гипотезы и прогноз", body


# ---------- 04: модели, промпт, прогоны ----------

def doc_models():
    man = json.loads(read_text("results/run_manifest.json"))
    cfg_models = {m["key"]: m for m in json.loads(read_text("config/models.json"))["models"]}
    rows = []
    for key, m in man["models"].items():
        c = cfg_models.get(key, {})
        rows.append([esc(c.get("role", "")), esc(key), esc(", ".join(m["model_reported"])),
                     esc(", ".join(str(x) for x in m["effective_temperature"])), esc(json.dumps(m["params"][0] if m["params"] else {}, ensure_ascii=False)),
                     esc(m["first_utc"][:16].replace("T", " ")), esc(m["last_utc"][:16].replace("T", " ")),
                     f"{m['ok']} / {m['errors']}", esc(", ".join(m["experiments"]))])
    models_tbl = table(["роль", "ключ", "model_reported", "T", "параметры", "первый вызов (UTC)", "последний (UTC)",
                        "ок / техошибок", "эксперименты"], rows)
    exp_rows = [[esc(k), esc(v["first_utc"][:16].replace("T", " ")), esc(v["last_utc"][:16].replace("T", " ")),
                 v["records"], v["ok"], v["errors"], v.get("unresolved_requests", 0)]
                for k, v in man["calls_by_experiment"].items()]
    exp_tbl = table(["эксперимент", "с (UTC)", "по (UTC)", "записей", "ок", "техошибок", "без ответа"], exp_rows)
    fmt_rows = [[esc(k), esc(v)] for k, v in prompts.FORMAT["ru"].items()]
    tasks = {t["id"]: t for t in read_jsonl("data/tasks.jsonl")}
    examples = ""
    for tid, variant in (("A1", "base"), ("A1", "cot"), ("B1", "base"), ("D1", "base")):
        system, prompt = prompts.build_messages(tasks[tid], variant, "ru")
        examples += f"<h4>{tid}, вариант {variant}</h4>" + pre(f"[system]\n{system}\n\n[user]\n{prompt}")
    env = man["environment"]
    srv = man.get("ollama_server") or {}
    env_tbl = table(["Параметр", "Значение"], [[esc(k), esc(v)] for k, v in
                    [("Python", env["python"]), ("Платформа", env["platform"]),
                     *[(f"пакет {p}", ver) for p, ver in env["packages"].items()],
                     ("sha256 requirements.txt", env["requirements_sha256"]),
                     ("Локальная модель: компьютер", f"{srv.get('hostname', '')}, {srv.get('os', '')}"),
                     ("CPU / RAM", f"{srv.get('cpu', '')}, {srv.get('cpu_threads', '')} потоков / {srv.get('ram_gb', '')} ГБ"),
                     ("GPU", ", ".join(srv.get("gpu", []))), ("Ollama", f"{srv.get('ollama_version', '')}; {', '.join(srv.get('models', []))}"),
                     ("Примечание", srv.get("note", ""))]])
    hp = man["prompt"]
    body = ("<h2>Модели</h2>" + models_tbl
            + "<div class='note'>T — фактическая температура. GigaChat не принимает 0, поэтому взята минимально допустимая 0,0001. "
              "Техошибки (сеть, 429, 5xx, таймауты) видны в журнале и ответами модели не считаются.</div>"
            + "<h2>Прогоны по экспериментам</h2>" + exp_tbl
            + f"<h2 class='pb'>Промпт (версия {esc(hp['version_code'])}, отпечаток {esc(hp['hash_code'])})</h2>"
            + "<h4>Системный промпт (RU)</h4>" + pre(prompts.SYSTEM["ru"])
            + "<h4>Требования к строке ОТВЕТ по виду ответа</h4>" + table(["вид", "текст"], fmt_rows)
            + "<h4>Добавка для H6 (cot)</h4>" + pre(prompts.COT["ru"])
            + "<h4>Реплика для H9 (второй ход диалога)</h4>" + pre(prompts.FOLLOWUP_DOUBT["ru"])
            + "<h3>Примеры полных запросов</h3>" + examples
            + "<h2 class='pb'>Окружение и железо</h2>" + env_tbl
            + "<h2 class='pb'>Отладка промпта на dev-задачах (docs/prompt_debug.md)</h2>" + md(read_text("docs/prompt_debug.md")))
    return "04_модели_промпт_прогоны", "Модели, промпт и прогоны", body


# ---------- 05: тип E ----------

def doc_typeE():
    from src import run_classifier, typeE
    split = json.loads(read_text("data/e_split.json"))
    df = typeE.load_data()
    shots = split["shots"]
    smallest = min(shots, key=lambda k: len(shots[k]))     # набор из 5 примеров
    task = run_classifier.make_task(df, split["test"][0], shots[smallest], "ru")
    system, prompt = prompts.build_messages(task, run_classifier.VARIANT, "ru")   # тот же вариант, что в прогоне
    split_tbl = table(["Параметр", "Значение"], [[esc(k), esc(split[k])] for k in
                      ("dataset", "seed", "test_mode", "n_rows", "n_failures_total", "natural_failure_rate",
                       "n_test", "n_test_failures", "pool_size", "pool_failures") if k in split]
                      + [[f"few-shot набор «{esc(k)}»", f"{len(v)} примеров"] for k, v in shots.items()])
    metrics = read_csv("results/typeE_metrics.csv")
    cols = ["coverage", "f1_failure", "balanced_accuracy", "accuracy", "f1_failure_sd", "balanced_accuracy_sd",
            "tokens_in_mean", "latency_s_mean"]
    m_rows = [[esc(r["source"]), esc(r["model"]), esc(r["shots"]), esc(r["train_set"]), esc(r["n_eval"]),
               *[num(r[c], 3) if r.get(c) not in (None, "") else "" for c in cols]] for r in metrics]
    body = (md(strip_h1(read_text("data/README.md")))
            + "<h2 class='pb'>Разбиение (data/e_split.json)</h2>" + split_tbl
            + "<h2>Пример запроса к модели (5 примеров, первая тестовая строка)</h2>" + pre(f"[system]\n{system}\n\n[user]\n{prompt}")
            + "<h2 class='pb'>Метрики (results/typeE_metrics.csv)</h2>"
            + "<div class='note'>Только числа из файла, без выводов. «Набор примеров»: prompt — примеры в промпте LLM; fixed — LogReg / NB / Dummy обучены на тех же 5 или 20 примерах; "
              "resample — среднее по 20 случайным стратифицированным наборам (sd — разброс); none — без примеров. "
              "Строка constant — всегда «нет отказа» на всём датасете (естественная доля отказов).</div>"
            + table(["источник", "модель", "примеров", "набор примеров", "строк", "доля разобранных", "F1 «отказ»",
                     "balanced acc.", "accuracy", "F1 sd", "BA sd", "токенов на входе (ср.)", "задержка, с (ср.)"], m_rows, "small"))
    return "05_тип_E_AI4I", "Тип E: LLM как классификатор (AI4I 2020)", body


# ---------- 06–11: проверка, тарифы, план, отчёт ----------

def doc_instructions():
    body = md(strip_h1(read_text("grading/INSTRUCTIONS.md"))) + "<h2 class='pb'>Чек-лист методологических ошибок D</h2>" + md(read_text("data/checklist_D.md"))
    return "06_инструкция_проверяющим", "Инструкция проверяющим и чек-лист D", body


def doc_sheet():
    rows = read_csv("grading/sheet_r1.csv")
    cards = []
    for r in rows:
        if r["type"] == "D":
            grade = ("Чек-лист (1 — ошибка есть, 0 — нет, н/п): "
                     + " ".join(f"{i}) ☐0 ☐1 ☐н/п" for i in range(1, 7))
                     + "<br>Результат верен (если не определён автоматически): ☐ да ☐ нет")
        else:
            grade = "Балл: ☐ 0 &nbsp; ☐ 1 &nbsp; ☐ 2"
        cards.append(
            f"<div class='card long'><h3>{esc(r['item_id'])} · тип {esc(r['type'])}</h3>"
            f"<div><span class='lbl'>Задача:</span> {esc(r['task_text'])}</div>"
            f"<div class='ref'><span class='lbl'>Эталон:</span> {esc(r['reference'])}</div>"
            f"<div class='lbl'>Рубрика / чек-лист:</div>{pre(r['rubric'])}"
            f"<div class='lbl'>Ответ модели:</div>{pre(r['answer_text'])}"
            + (f"<div class='lbl'>Результат запуска кода:</div>{pre(r['exec_result'])}" if r.get("exec_result", "").strip() not in ("", "—") else "")
            + (f"<div><span class='lbl'>Метрика (автоматически):</span> {esc(r['auto_metric'])}</div>" if r.get("auto_metric", "").strip() not in ("", "—") else "")
            + f"<div class='check'>{grade}<br>Комментарий: ____________________________________________</div></div>")
    body = ("<div class='note'><b>Только для чтения.</b> Оценки вносятся в CSV: проверяющий 1 — <code>grading/sheet_r1.csv</code>, "
            "проверяющий 2 — <code>grading/sheet_r2.csv</code>. Порядок строк одинаковый. Модель-автор скрыта. "
            "Не обсуждайте оценки друг с другом и не открывайте <code>grading/key_DO_NOT_OPEN.csv</code>.</div>"
            f"<p>Ответов: {len(rows)} (B — {sum(r['type'] == 'B' for r in rows)}, C — {sum(r['type'] == 'C' for r in rows)}, "
            f"D — {sum(r['type'] == 'D' for r in rows)}).</p>" + "".join(cards))
    return "07_лист_слепой_проверки", "Лист слепой проверки (B, C, D)", body


def doc_spotcheck():
    rows = read_csv("grading/parser_spotcheck.csv")
    cards = [f"<div class='card'><h3>{esc(r['spot_id'])} · {esc(r['task_id'])} · тип {esc(r['type'])} · отбор: {esc(r['selection'])}</h3>"
             f"<div class='lbl'>Конец ответа модели:</div>{pre(r['response_text'][-900:])}"
             + table(["строка ОТВЕТ", "распознано", "отказ", "формат ок", "эталон", "автопроверка", "причина"],
                     [[esc(r["raw_answer_line"]), esc(r["parsed_value"]), esc(r["refused"]), esc(r["parse_ok"]),
                       esc(r["reference"]), esc(r["auto_correct"]), esc(r["auto_reason"])]])
             + "<div class='check'>Согласен(на) с автопарсером: ☐ да ☐ нет &nbsp; Комментарий: ______________________</div></div>"
             for r in rows]
    body = ("<div class='note'>Ответ «да / нет» вносится в столбец <code>human_agrees</code> файла "
            "<code>grading/parser_spotcheck.csv</code>, комментарий — в <code>comment</code>.</div>" + "".join(cards))
    return "08_проверка_парсера", "Выборочная проверка автопарсера", body


def doc_pricing():
    p = json.loads(read_text("config/pricing.json"))
    rows = [[esc(k), num(v.get("rub_per_1m_in"), 2), num(v.get("rub_per_1m_out"), 2), esc(v.get("vat_included", "")),
             esc(json.dumps(v.get("original", ""), ensure_ascii=False)), esc(v.get("source_url", "")),
             esc(v.get("date_checked", "")), esc(v.get("note", ""))] for k, v in p["models"].items()]
    head = table(["Поле", "Значение"], [[esc(k), esc(p.get(k))] for k in ("verified_by_human", "date_checked", "checked_by", "unit", "usd_rub") if k in p])
    body = (head + "<div class='note'>Сверьте цены с официальными страницами (source_url), впишите дату и поставьте "
            "<code>\"verified_by_human\": true</code> в <code>config/pricing.json</code>.</div>"
            + table(["модель", "₽ / 1 млн вход", "₽ / 1 млн выход", "НДС", "исходная цена", "источник", "дата", "примечание"], rows))
    return "09_тарифы", "Тарифы для расчёта стоимости", body


def doc_todo():
    return "10_план_для_человека", "Что осталось сделать человеку", md(strip_h1(read_text("HUMAN_TODO.md")))


def doc_report():
    body = (md(strip_h1(read_text("report/report.md"))) + "<h2 class='pb'>Вопросы для самоконтроля</h2>" + md(read_text("report/selfcheck_questions.md"))
            + "<h2 class='pb'>План доклада на 5 минут</h2>" + md(read_text("report/slides_outline.md")))
    return "11_каркас_отчёта", "Каркас отчёта, вопросы самоконтроля, план доклада", body


def doc_compliance():
    return "01_сверка_с_заданием", "Сверка с заданием ЛР3", md(strip_h1(read_text("docs/compliance_check.md")))


def doc_guide():
    return "00_как_проверить_вручную", "Как проверить всё самостоятельно", md(strip_h1(read_text("docs/manual_check_guide.md")))


def doc_answers_af():
    """Ответы моделей на A и F основного прогона рядом с вердиктом автопроверки.
    B, C, D сюда не попадают: их проверяют вслепую, названия моделей рядом с ними видеть нельзя."""
    tasks = {t["id"]: t for t in read_jsonl("data/tasks.jsonl")}
    rows = [r for r in read_jsonl("results/main_ru.jsonl") if r["type"] in ("A", "F")]
    order = [t for t in tasks if tasks[t]["type"] in ("A", "F")]
    cards = []
    for tid in order:
        t = tasks[tid]
        lines = []
        for r in sorted((r for r in rows if r["task_id"] == tid), key=lambda r: r["model_key"]):
            fmt = "" if r["parse_ok"] else " <b>(формат не соблюдён, ответ найден эвристикой)</b>"
            verdict = "✅ верно" if r["correct"] else "❌ неверно"
            lines.append(f"<h4>{esc(r['model_key'])} — {verdict}{fmt}</h4>"
                         + table(["строка ОТВЕТ", "распознано", "уверенность", "причина вердикта"],
                                 [[esc(r["raw_answer_line"] or "—"), esc(r["value"]), esc(r["confidence"]), esc(r["reason"])]])
                         + f"<div class='lbl'>Конец ответа модели:</div>{pre((r['response_text'] or '')[-700:])}")
        cards.append(f"<div class='card long'><h3>{esc(tid)}</h3><div>{esc(t['text_ru'])}</div>"
                     f"<div class='ref'><span class='lbl'>Эталон:</span> {fmt_reference(t)}</div>{''.join(lines)}"
                     "<div class='check'>☐ Все вердикты автопроверки верны &nbsp; ☐ Есть замечание: ______________________</div></div>")
    body = ("<div class='note'>Основной прогон, задачи A и F. Проверьте, что вердикт автопроверки верен: число совпадает с эталоном "
            "в пределах допуска (обычно ±1%), для F6–F8 верен только ответ «нельзя определить». Ответы на B, C и D сюда не включены — "
            "их вы оцениваете вслепую.</div>" + "".join(cards))
    return "12_ответы_A_F_и_автопроверка", "Ответы моделей на A и F и вердикты автопроверки", body


DOCS = {"00": doc_guide, "01": doc_compliance, "02": doc_bank, "03": doc_hypotheses, "04": doc_models, "05": doc_typeE,
        "06": doc_instructions, "07": doc_sheet, "08": doc_spotcheck, "09": doc_pricing, "10": doc_todo, "11": doc_report, "12": doc_answers_af}


def main(argv=None) -> int:
    wanted = (argv if argv is not None else sys.argv[1:]) or list(DOCS)
    for key in wanted:
        name, title, body = DOCS[key]()
        out = to_pdf(name, page(title, stamp(), body))
        print(f"  {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
