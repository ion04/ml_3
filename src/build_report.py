"""Каркас отчёта ЛР3 → report/report.md (шаг 12). Структура — «Структура отчёта по ЛР» из PDF.

Откуда берутся числа (руками ничего не вписывается):
  results/summary_metrics.json  все метрики (его пишет analysis.ipynb, раздел 8);
  results/run_manifest.json     модели, model_reported, даты и число вызовов, железо локальной модели, окружение;
  config/*.json                 модели, параметры эксперимента, тарифы;
  data/tasks.jsonl              состав банка (таблица «тип × тема»);
  hypotheses.md                 гипотезы и прогноз — в версии из последнего коммита, изменившего файл;
  src/prompts.py                полный текст промпта.
Если чего-то нет, в отчёте стоит «нет данных».

Текст человека (введение, вердикты, разбор ошибок, ограничения, рекомендации) пишется между строками
  <!-- ЧЕЛОВЕК:<id> — подсказка -->
  ...
  <!-- /ЧЕЛОВЕК:<id> -->
При пересборке отчёта этот текст переносится в новый файл. Блок, которому в новой структуре нет места
(например, гипотезу заменили), не теряется: он попадает в конец отчёта.

  .venv/bin/python -m src.build_report            # → report/report.md
  .venv/bin/python -m src.build_report --demo     # → demo/report/report.md (mock, не результаты)
  .venv/bin/python -m src.build_report check      # какие блоки человека ещё пусты (код 1, если есть)
"""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from src import analysis as an
from src import bank, blind, checks, llm, prompts, run_bank

ROOT = run_bank.ROOT
NO_DATA = "_нет данных_"
# Блок человека: открывающая строка с id и подсказкой, текст, закрывающая строка. Пробелы вокруг «-->» и текст
# на той же строке, что открывающий комментарий, допускаются (редактор может их оставить).
HUMAN_RE = re.compile(r"<!-- ЧЕЛОВЕК:([\w.-]+)(?![\w.-])[^\n]*?-->[ \t]*\n?(.*?)\n?[ \t]*<!-- */ЧЕЛОВЕК:\1\s*-->", re.S)
OPEN_RE = re.compile(r"<!--\s*ЧЕЛОВЕК:[\w.-]")   # сколько блоков начато (для проверки, что все распознаны)

HYP_TITLES = {   # как в заголовках analysis.ipynb, раздел 6
    "H1": "Учебниковые задачи против изменённых чисел",
    "H2": "Игнорирование базовой частоты: формульная vs словесная версия",
    "H3": "Додумывание при недостающих данных",
    "H4": "Язык запроса: RU vs EN",
    "H5": "Маленькая модель: A vs B",
    "H6": "«Думай пошагово»",
    "H7": "Самосогласованность: 5 ответов при T = 0,7 и голосование",
    "H8": "Устойчивость к перефразированию",
    "H9": "Поддакивание",
    "H10": "Калибровка уверенности (ECE, диаграмма надёжности)",
    "H11": "Код запускается, но методологически неверен",
    "H12": "Few-shot LLM против логистической регрессии",
    "H13": "Стоимость правильного ответа",
}
# Рисунки гипотез (имена из analysis.ipynb, save_fig).
HYP_FIGURES = {"H1": "H1_original_vs_modified", "H2": "H2_formula_vs_verbal", "H3": "H3_invented",
               "H4": "H4_ru_vs_en", "H5": "H5_gap", "H6": "H6_cot", "H7": "H7_selfconsistency",
               "H8": "H8_paraphrase", "H9": "H9_flip_rate", "H10": "H10_reliability",
               "H11": "H11_code_errors", "H12": "H12_llm_vs_logreg", "H13": "cost_accuracy"}
TYPE_NAMES = {"A": "расчётные", "B": "«верно / неверно» с обоснованием", "C": "выбор метода",
              "D": "код scikit-learn", "E": "LLM как классификатор (AI4I 2020)",
              "F": "ловушки: изменённые условия, недостающие и лишние данные"}
ERRORS_N = 5


def report_paths(demo: bool) -> dict:
    p = an.paths(demo)
    base = run_bank.DEMO_DIR if demo else ROOT
    return {"report": base / "report" / "report.md", "figures": p["figures"], "summary": p["summary"],
            "manifest": p["manifest"], "errors": base / "report" / "error_candidates.md"}


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---------- Форматирование ----------

def is_p(key: str) -> bool:
    return key in ("p_value", "p_holm", "mcnemar_p") or key.endswith("_p") or key.startswith("p_")


def is_count(key: str) -> bool:
    return key == "n" or key.startswith(("n_", "only_", "both_")) or key in ("tp", "fp", "fn", "tn", "records")


def fmt_key(key: str, v) -> str:
    """fmt с учётом смысла столбца: p-значения — 4 значащие цифры, счётчики — целым числом."""
    if isinstance(v, float) and not math.isnan(v) and not math.isinf(v):
        if is_p(key):
            return f"{v:.4g}"
        if is_count(key) and v.is_integer():
            return str(int(v))
    return fmt(v)


def fmt(v) -> str:
    """Значение для ячейки таблицы. Числа не округляются по смыслу — только до 3 знаков для чтения."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, bool):
        return "да" if v else "нет"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isinf(v):
            return "∞" if v > 0 else "−∞"
        if v != 0 and abs(v) < 1e-3:
            return f"{v:.1e}"
        return f"{v:.3f}"
    if isinstance(v, list):
        return "; ".join(fmt(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v).replace("|", "\\|").replace("\n", " ")


def ci(row: dict, value: str, lo: str = "ci_low", hi: str = "ci_high") -> str:
    """«0.750 [0.500; 0.900]» — значение с 95% доверительным интервалом."""
    if row.get(value) is None:
        return "—"
    return f"{fmt(row[value])} [{fmt(row.get(lo))}; {fmt(row.get(hi))}]"


def md_table(rows: list[dict], columns: list | None = None) -> str:
    """Таблица markdown. columns — список ключей или пар (ключ, заголовок); по умолчанию все ключи первой строки."""
    if not rows:
        return NO_DATA
    columns = columns or list(rows[0].keys())
    cols = [c if isinstance(c, tuple) else (c, c) for c in columns]
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt_key(k, r.get(k)) for k, _ in cols) + " |")
    return "\n".join(lines)


def df_table(df) -> str:
    """Таблица pandas (crosstab) → markdown: индекс становится первым столбцом."""
    flat = df.reset_index()
    flat.columns = [str(c) for c in flat.columns]
    return md_table(flat.to_dict("records"))


def kv_list(d: dict) -> str:
    return "\n".join(f"- `{k}`: {fmt_key(k, v)}" for k, v in d.items()) if d else NO_DATA


def figure(name: str, figures_dir: Path, caption: str) -> str:
    """Ссылка на рисунок из report/figures/ (путь относительно report.md) или пометка, что его нет."""
    if (figures_dir / f"{name}.png").exists():
        return f"![{caption}](figures/{name}.png)\n\n_Рисунок: {caption} (`figures/{name}.png`)._"
    return f"_Рисунок «{caption}» (`figures/{name}.png`): нет данных — создаётся analysis.ipynb._"


def fence(text: str, lang: str = "text") -> str:
    """Блок кода, ограничитель длиннее любой последовательности ` внутри текста."""
    run = max((len(m) for m in re.findall(r"`+", text)), default=0)
    f = "`" * max(3, run + 1)
    return f"{f}{lang}\n{text}\n{f}"


# ---------- Блоки человека ----------

class Human:
    """Собирает блоки человека: берёт уже написанный текст из старого отчёта, иначе текст по умолчанию."""

    def __init__(self, old_text: str):
        self.old = {m.group(1): m.group(2) for m in HUMAN_RE.finditer(old_text)}
        self.n_open = len(OPEN_RE.findall(old_text))     # больше, чем len(old) — какой-то блок испорчен
        self.used: list[tuple[str, bool]] = []    # (id, заполнен ли человеком)

    def __call__(self, block_id: str, hint: str, default: str = "") -> str:
        body = self.old.get(block_id, default)
        self.used.append((block_id, bool(body.strip()) and body.strip() != default.strip()))
        return f"<!-- ЧЕЛОВЕК:{block_id} — {hint} -->\n{body}\n<!-- /ЧЕЛОВЕК:{block_id} -->"

    def orphans(self) -> str:
        """Блоки из старого отчёта, которым нет места в новой структуре: переносятся как есть."""
        used = {b for b, _ in self.used}
        lost = [(b, t) for b, t in self.old.items() if b not in used and t.strip()]
        if not lost:
            return ""
        parts = ["## Сохранённые блоки человека без места в текущей структуре", "",
                 "Структура отчёта изменилась (например, выбрана другая гипотеза). Перенесите текст или удалите блок."]
        for b, t in lost:
            parts += ["", f"<!-- ЧЕЛОВЕК:{b} — блок из прошлой версии отчёта -->\n{t}\n<!-- /ЧЕЛОВЕК:{b} -->"]
        return "\n".join(parts)


# ---------- Разделы ----------

def committed_hypotheses(selected: list[str]) -> tuple[list[dict], str]:
    """Гипотезы из hypotheses.md в версии последнего коммита и строка про коммит."""
    if not selected:
        return [], "Гипотезы не выбраны: `config/experiment.json` → `hypotheses` пуст (нет данных)."
    commit = checks.last_commit(checks.HYPOTHESES_MD)
    if commit is None:
        return [], "`hypotheses.md` не закоммичен (нет данных)."
    sha, when = commit
    text = checks._git(["show", f"{sha}:hypotheses.md"])
    early = checks.early_eval_calls(checks.LOG_PATH, when)
    note = f"Гипотезы и прогноз зафиксированы коммитом `{sha}` ({when.isoformat()})"
    note += (f". **Внимание: в журнале {len(early)} оценочных вызовов раньше этого коммита** (`src.checks g1`)."
             if early else ", раньше всех оценочных вызовов в `logs/calls.jsonl`.")
    if checks.is_modified(checks.HYPOTHESES_MD):
        note += " **Внимание: `hypotheses.md` изменён после коммита; в отчёте — версия из коммита.**"
    return checks.parse_hypotheses_md(text), note


def section_intro(h: Human, cfg: dict) -> str:
    hyps, note = committed_hypotheses(cfg.get("hypotheses") or [])
    cols = [("ID", "ID"), ("Группа", "Группа"), ("Формулировка", "Формулировка"), ("Прогноз", "Прогноз"),
            ("Метрика", "Метрика"), ("Критерий решения (тест, порог)", "Критерий решения")]
    return "\n\n".join([
        "## 1. Введение",
        h("intro", "введение: цель работы, зачем проверять LLM на задачах классического ML, что сделано"),
        "### Гипотезы и прогноз (из коммита)",
        note,
        md_table(hyps, cols) if hyps else NO_DATA,
    ])


def models_rows(models_cfg: dict, manifest: dict | None, demo: bool) -> list[dict]:
    """Строка на модель: идентификатор в запросе, model_reported из ответов API, даты, параметры."""
    used = (manifest or {}).get("models") or {}
    keys = [k for k, m in models_cfg.items() if m["enabled"] and m["provider"] != "mock"] if not demo else []
    keys += [k for k in used if k not in keys]
    rows = []
    for k in an.order_models(keys, models_cfg):
        m, u = models_cfg.get(k, {}), used.get(k, {})
        rows.append({
            "role": m.get("role"), "key": k, "provider": m.get("provider"),
            "model_requested": u.get("model_requested") or m.get("model"),
            "model_reported": u.get("model_reported") or "нет данных",
            "dates": f"{u['first_utc'][:10]} … {u['last_utc'][:10]}" if u.get("first_utc") else "нет данных",
            "temperature": u.get("effective_temperature"), "params": m.get("params"),
            "max_tokens": u.get("max_tokens"),
            "calls": f"{u['ok']} / {u['errors']}" if "ok" in u else "нет данных",
        })
    return rows


def prompt_text(langs: list[str]) -> str:
    """Полный текст всех частей промпта для языков эксперимента."""
    parts = [f"Версия `{prompts.PROMPT_VERSION}`, отпечаток текстов `prompts.prompt_hash()` = `{prompts.prompt_hash()}`.",
             "Сообщение пользователя = условие задачи (+ просьба рассуждать для cot) + требование к формату "
             "по типу ответа. Системный промпт один на язык, общий для всех моделей и типов задач "
             "(`src/prompts.py`)."]
    for lang in langs:
        c = prompts.CLASSIFY[lang]
        parts += [f"#### Язык {lang}: системный промпт", fence(prompts.SYSTEM[lang]),
                  f"#### Язык {lang}: требования к формату (в конце сообщения пользователя)"]
        parts += [f"- `{g}`:\n\n{fence(t)}" for g, t in prompts.FORMAT[lang].items()]
        parts += [f"#### Язык {lang}: варианты",
                  f"- `cot` (H6), добавляется после условия:\n\n{fence(prompts.COT[lang])}",
                  f"- `followup_doubt` (H9), второй ход диалога:\n\n{fence(prompts.FOLLOWUP_DOUBT[lang])}",
                  f"- `classify_row` (тип E): инструкция, примеры, строка-запрос, формат `label`:\n\n"
                  + fence("\n".join(f"{k}: {v}" for k, v in c.items()))]
    return "\n\n".join(parts)


def bank_section(tasks: list[dict], cfg: dict, manifest: dict | None) -> str:
    tables = bank.stats_tables(tasks)
    n = len(tasks)
    own = sum(t["source"] == "свои" for t in tasks)
    status = {s: sum(t["reference_status"] == s for t in tasks) for s in bank.STATUSES}
    e = cfg.get("typeE", {})
    split_hash = (manifest or {}).get("typeE_split_hash")
    return "\n\n".join([
        f"Задач в банке: {n}; придуманы самостоятельно (`source = свои`): {own} ({own / n:.0%})." if n else NO_DATA,
        "Статус эталонов: " + ", ".join(f"{k} — {v}" for k, v in status.items()) + ".",
        "**Тип × тема**", df_table(tables["тип × тема"]),
        "**Тип × источник**", df_table(tables["тип × источник"]),
        "**Тип × вариант** (пары для H1/H3, лишние данные)", df_table(tables["тип × вариант"]),
        f"**Тип E:** {e.get('dataset')}; тест {e.get('n_test')} строк ({e.get('test_mode')}, отказов "
        f"{e.get('n_test_failures')}); примеры в промпте {e.get('shots')}, из них отказов {e.get('shot_failures')}; "
        f"отпечаток разбиения `{split_hash or 'нет данных'}` (`data/e_split.json`, `data/README.md`).",
    ])


def usd_rub(pricing: dict):
    """Курс ЦБ из config/pricing.json: число или {"rate": ..., "date": ...}."""
    v = pricing.get("usd_rub")
    return f"{v['rate']} (ЦБ на {v.get('date')})" if isinstance(v, dict) else fmt(v)


def grading_section(summary: dict | None, pricing: dict, cfg: dict) -> str:
    s = summary or {}
    meta = s.get("pricing") or pricing    # дата и курс тех тарифов, по которым ноутбук считал стоимость
    rules = [
        "- A, F, вердикт B, E — автопроверка (`src/grade.py`, `results/graded_auto.jsonl`): числа сравниваются "
        "с эталоном с относительным допуском задачи (по умолчанию 1%); на задачах с недостающими данными верен "
        "ответ «нельзя определить» или отказ.",
        f"- B (обоснование), C, D — два проверяющих независимо и вслепую (`src/blind.py`, `grading/INSTRUCTIONS.md`): "
        f"B и C — балл 0–2 по рубрике, C «верно» при балле ≥ {an.C_CORRECT_MIN_SCORE}; D — запуск кода без правок "
        f"(pass@1, `src/exec_code.py`) и чек-лист из {len(blind.CHECKLIST)} методологических ошибок "
        "(`data/checklist_D.md`). Итог — среднее двух проверяющих (0; 0,5; 1).",
        f"- Согласие проверяющих — каппа Коэна; порог {an.KAPPA_MIN}.",
        f"- Доверительные интервалы — бутстреп по задачам ({s.get('bootstrap_n', cfg.get('bootstrap_n'))} ресэмплов, "
        f"seed {s.get('seed', cfg.get('seed'))}), для долей в гипотезах — Клоппер–Пирсон; пары моделей — тест "
        "Макнемара (`exact=True`) с поправкой Холма.",
        "- Цена — по официальным тарифам `config/pricing.json` (₽ за 1 млн токенов входа и выхода), "
        f"дата проверки {meta.get('date_checked')}, курс {usd_rub(meta)} ₽/$, "
        f"сверено человеком: {fmt(meta.get('verified_by_human'))}."
        + (" **Внимание: `config/pricing.json` изменён после расчёта стоимости (другая дата проверки) — "
           "перезапустите ноутбук.**" if s.get("pricing") and s["pricing"].get("date_checked") != pricing.get("date_checked")
           else ""),
    ]
    kappa_cols = [("type", "тип"), ("what", "что сравнивается"), ("n_pairs", "n"), ("agreement", "совпадение"),
                  ("kappa", "κ"), ("below_min", f"κ < {an.KAPPA_MIN}")]
    spot_cols = [("selection", "выборка"), ("n", "n"), ("n_yes", "согласен"), ("share_yes", "доля"),
                 ("ci_low", "CI low"), ("ci_high", "CI high")]
    price_rows = [{"model": k, **{f: v.get(f) for f in ("rub_per_1m_in", "rub_per_1m_out", "source_url")}}
                  for k, v in (pricing.get("models") or {}).items()]
    return "\n\n".join(["\n".join(rules),
                        "**Каппа Коэна**", md_table(s.get("kappa") or [], kappa_cols),
                        "**Выборочная проверка автопарсера** (человек согласен с разбором)",
                        md_table(s.get("parser_spotcheck") or [], spot_cols),
                        "**Тарифы** (`config/pricing.json`)", md_table(price_rows)])


def section_methods(h: Human, cfg: dict, models_cfg: dict, manifest: dict | None, summary: dict | None,
                    tasks: list[dict], pricing: dict) -> str:
    man = manifest or {}
    model_cols = [("role", "роль"), ("key", "ключ"), ("provider", "провайдер"),
                  ("model_requested", "модель в запросе"), ("model_reported", "model_reported (ответ API)"),
                  ("dates", "даты вызовов (UTC)"), ("temperature", "T"), ("params", "параметры"),
                  ("max_tokens", "max_tokens"), ("calls", "вызовов ok / ошибок")]
    calls = [{"experiment": k, **v} for k, v in (man.get("calls_by_experiment") or {}).items()]
    pr = man.get("prompt") or {}
    prompt_note = (f"Промпт в config: версия `{pr.get('version_config')}`, отпечаток `{pr.get('hash_config')}`; "
                   f"в коде: `{pr.get('version_code')}`, `{pr.get('hash_code')}`." if pr else NO_DATA)
    if pr and (pr.get("version_config") is None or pr.get("hash_config") != pr.get("hash_code")):
        prompt_note += " **Промпт не заморожен или изменён после заморозки.**"
    params = {k: v for k, v in (man.get("params") or {}).items() if k != "typeE"}
    env = man.get("environment") or {}
    env_line = (f"Python {env.get('python')}, {env.get('platform')}; пакеты: "
                + ", ".join(f"{k} {v}" for k, v in (env.get("packages") or {}).items())
                + f"; sha256 `requirements.txt`: `{str(env.get('requirements_sha256'))[:12]}`; "
                f"git HEAD при сборке манифеста: `{str(man.get('git_head'))[:12]}`." if env else NO_DATA)
    server = man.get("ollama_server") or {}
    server_keys = ["hostname", "os", "cpu", "cpu_threads", "ram_gb", "ram_modules", "gpu", "ollama_version",
                   "models", "collected_utc", "note"]
    mem = ((summary or {}).get("local_model") or {}).get("memory") or []
    return "\n\n".join([
        "## 2. Методика",
        "### 2.1. Модели",
        md_table(models_rows(models_cfg, manifest, bool(man.get("demo"))), model_cols),
        "Даты, `model_reported`, температура и число вызовов — из журнала `logs/calls.jsonl` "
        "(`results/run_manifest.json`, раздел `models`; вызовы smoke и dev не входят).",
        "**Вызовы по экспериментам** (все записи журнала, включая повторы после технических ошибок)",
        md_table(calls, ["experiment", "first_utc", "last_utc", "records", "ok", "errors", "retried_requests",
                         "unresolved_requests"]) if calls else NO_DATA,
        "### 2.2. Параметры и окружение",
        kv_list(params),
        prompt_note,
        env_line,
        "### 2.3. Железо для локальной модели",
        kv_list({k: server.get(k) for k in server_keys if k in server}),
        "**Память локальной модели по журналу** (`/api/ps` Ollama)",
        md_table(mem),
        "### 2.4. Промпт",
        prompt_text(cfg.get("languages") or ["ru"]),
        "### 2.5. Банк задач",
        bank_section(tasks, cfg, manifest),
        "### 2.6. Способ проверки",
        grading_section(summary, pricing, cfg),
        h("methods_extra", "дополнения к методике: состав команды, кто проверял вслепую, решения по TLS и тарифам; "
                           "можно оставить пустым"),
    ])


ACC_COLS = [("n", "n"), ("n_graded", "оценено"), ("acc_ci", "точность [95% CI]"),
            ("refusal_rate", "доля отказов"), ("accuracy_answered", "точность без отказов")]


def acc_rows(rows: list[dict], key: str) -> list[dict]:
    return [{key: r.get(key), **r, "acc_ci": ci(r, "accuracy")} for r in rows]


def model_type_pivot(rows: list[dict]) -> str:
    """Точность «модель × тип» с CI: строки — модели, столбцы — типы."""
    if not rows:
        return NO_DATA
    types = [t for t in an.BANK_TYPES if any(r["type"] == t for r in rows)]
    models = list(dict.fromkeys(r["model_key"] for r in rows))
    cell = {(r["model_key"], r["type"]): ci(r, "accuracy") for r in rows}
    return md_table([{"model_key": m, **{t: cell.get((m, t), "—") for t in types}} for m in models],
                    ["model_key", *types])


def hypothesis_block(h: Human, hid: str, data: dict | None, prediction: dict | None, figures_dir: Path,
                     selected: bool) -> str:
    parts = [f"#### {hid}. {HYP_TITLES.get(hid, '')}" + ("" if selected else " (не выбрана, разведочный анализ)")]
    if prediction:
        parts.append(f"Прогноз из коммита: **{prediction.get('Прогноз', '—')}**. Критерий: "
                     f"{prediction.get('Критерий решения (тест, порог)', '—')}.")
    if not data:
        parts.append(NO_DATA)
    else:
        scalars = {k: v for k, v in data.items() if not isinstance(v, (list, dict))}   # H11: плоский словарь
        if scalars:
            parts.append(kv_list(scalars))
        for name, value in data.items():
            if name in scalars:
                continue
            parts.append(f"`{name}`")
            if isinstance(value, list):
                parts.append(md_table(value))
            elif isinstance(value, dict) and all(isinstance(v, list) for v in value.values()):  # H10 bins
                parts += [f"{k}:\n\n{md_table(v)}" for k, v in value.items()]
            elif isinstance(value, dict):
                parts.append(kv_list(value))
            else:
                parts.append(fmt(value))
    if hid in HYP_FIGURES:
        parts.append(figure(HYP_FIGURES[hid], figures_dir, f"{hid}: {HYP_TITLES.get(hid, '')}"))
    if selected:
        parts.append(h(f"verdict.{hid}", f"вердикт по {hid}: подтвердилась / опровергнута / данных недостаточно; "
                                         "сравнение с прогнозом, опора на числа выше"))
    return "\n\n".join(parts)


def section_results(h: Human, summary: dict | None, cfg: dict, figures_dir: Path) -> str:
    if not summary:
        missing = ("_Нет данных: `results/summary_metrics.json` ещё не создан. Он появится после прогонов "
                   "(шаги 08–09), ручной проверки (G2) и `scripts/run_analysis.sh` (шаг 11)._")
    s = summary or {}
    parts = ["## 3. Результаты"]
    if not summary:
        parts.append(missing)
    counts = s.get("counts") or {}
    parts += ["### 3.1. Объём данных", kv_list(counts)]

    parts += ["### 3.2. Точность по моделям (основной прогон, RU; A, B, C, D, F)",
              md_table(acc_rows(s.get("accuracy_by_model") or [], "model_key"), [("model_key", "модель"), *ACC_COLS]),
              figure("accuracy_ci", figures_dir, "точность моделей с 95% доверительными интервалами"),
              "### 3.3. Точность «модель × тип задачи» [95% CI]",
              model_type_pivot(s.get("accuracy_by_model_type") or []),
              figure("heatmap_model_type", figures_dir, "тепловая карта «модель × тип задачи»"),
              "**По типам задач**",
              md_table(acc_rows(s.get("accuracy_by_type") or [], "type"), [("type", "тип"), *ACC_COLS]),
              "**Баллы 0–2 по рубрике (B, C), среднее двух проверяющих**",
              md_table(s.get("rubric_scores") or []),
              "### 3.4. Значимость различий моделей: тест Макнемара",
              md_table(s.get("mcnemar_models") or [],
                       [("model_a", "модель 1"), ("model_b", "модель 2"), ("n", "n"),
                        ("only_first", "верно только 1"), ("only_second", "верно только 2"),
                        ("acc_first", "точность 1"), ("acc_second", "точность 2"), ("p_value", "p"),
                        ("p_holm", "p (Холм)")]),
              "### 3.5. Тип D: код scikit-learn",
              md_table(s.get("typeD") or []),
              figure("typeD_checklist", figures_dir, "тип D: методологические ошибки по чек-листу"),
              "### 3.6. Тип E: LLM как классификатор",
              md_table(s.get("typeE") or [],
                       ["source", "model", "model_reported", "lang", "shots", "train_set", "eval_set", "n_eval", "coverage", "f1_failure",
                        "balanced_accuracy", "accuracy", "f1_failure_sd", "balanced_accuracy_sd", "complete"]),
              figure("typeE_shots", figures_dir, "тип E: F1 «отказа» и balanced accuracy от числа примеров"),
              "### 3.7. Цена и скорость",
              md_table(s.get("cost") or [],
                       ["model_key", "n", "accuracy", "latency_s_mean", "tokens_in_mean", "tokens_out_mean",
                        "cost_rub_total", "rub_per_correct", "rub_per_correct_ci_low", "rub_per_correct_ci_high"]),
              "**Тип E: токены и стоимость**", md_table(s.get("cost_typeE") or []),
              figure("cost_accuracy", figures_dir, "«стоимость – точность» для каждой модели")]

    # Гипотезы: сначала три выбранные (с вердиктом человека), затем остальные, по которым есть числа.
    selected = cfg.get("hypotheses") or []
    hyp_data = s.get("hypotheses") or {}
    predictions = {p.get("ID", "").replace("*", "").strip(): p for p in committed_hypotheses(selected)[0]}
    parts.append("### 3.8. Гипотезы")
    if not selected:
        parts.append("_Гипотезы не выбраны (нет данных)._")
    for hid in selected:
        parts.append(hypothesis_block(h, hid, hyp_data.get(hid), predictions.get(hid), figures_dir, True))
    others = [hid for hid in hyp_data if hid not in selected]
    if others:
        parts += ["### 3.9. Прочие гипотезы (разведочный анализ, не входят в три выбранные)",
                  "_Посчитаны ноутбуком, потому что для них нашлись данные. Оставить или удалить — решает человек._"]
        parts += [hypothesis_block(h, hid, hyp_data[hid], None, figures_dir, False) for hid in others]
    return "\n\n".join(parts)


def section_errors(h: Human, errors_path: Path) -> str:
    link = (f"Кандидаты, отобранные по журналу: [`{errors_path.name}`]({errors_path.name}) "
            "(`src/error_candidates.py`)." if errors_path.exists()
            else f"_Кандидатов пока нет: `{errors_path.name}` создаётся после ручной проверки (G2) "
                 "командой `.venv/bin/python -m src.error_candidates`._")
    parts = ["## 4. Разбор пяти ошибок моделей", link]
    for i in range(1, ERRORS_N + 1):
        parts += [f"### Ошибка {i}",
                  h(f"error.{i}", "задача, модель, call_id, точная цитата ответа из журнала; в каком месте "
                                  "рассуждения модель сбилась и почему")]
    return "\n\n".join(parts)


def limitation_facts(cfg: dict, manifest: dict | None, summary: dict | None, pricing: dict,
                     tasks: list[dict]) -> list[str]:
    """Факты из файлов, которые могут пригодиться для раздела ограничений (не формулировки ограничений)."""
    e = cfg.get("typeE", {})
    man, s = manifest or {}, summary or {}
    facts = [f"тип E: тест `{e.get('test_mode')}`, {e.get('n_test')} строк, из них отказов "
             f"{e.get('n_test_failures')} (`config/experiment.json` → `typeE`)",
             f"банк: {len(tasks)} задач, по типам " + ", ".join(
                 f"{t} — {sum(x['type'] == t for x in tasks)}" for t in bank.FINAL_TYPES),
             f"тарифы сверены человеком: {fmt(pricing.get('verified_by_human'))} (`config/pricing.json`)"]
    note = (man.get("ollama_server") or {}).get("note")
    if note:
        facts.append(f"железо локальной модели: {note} (`run_manifest.json` → `ollama_server.note`)")
    unresolved = {k: v.get("unresolved_requests") for k, v in (man.get("calls_by_experiment") or {}).items()
                  if v.get("unresolved_requests")}
    if unresolved:
        facts.append(f"запросы без ответа после повторов: {unresolved}")
    if s.get("kappa_below_min"):
        facts.append(f"κ ниже {an.KAPPA_MIN} у типов: {s['kappa_below_min']}")
    return facts


def section_limits_recs(h: Human, facts: list[str]) -> str:
    rec_default = "\n".join(
        ["| Тип задачи | Можно доверять / проверять выборочно / не доверять | На какие числа опирается |",
         "|---|---|---|"] + [f"| {t} — {name} |  |  |" for t, name in TYPE_NAMES.items()])
    return "\n\n".join([
        "## 5. Ограничения эксперимента",
        "Справка (факты из файлов проекта, не формулировки ограничений):\n\n" + "\n".join(f"- {f}" for f in facts),
        h("limitations", "не меньше трёх ограничений, своими словами"),
        "## 6. Практические рекомендации",
        h("recommendations", "заполнить таблицу: вердикт по каждому типу задачи и числа, на которые он опирается",
          rec_default),
        "## Заключение",
        h("conclusion", "заключение: главное, что показал эксперимент (пишет человек)"),
    ])


def build(demo: bool, old_text: str = "") -> tuple[str, list[tuple[str, bool]]]:
    """Текст отчёта и список блоков человека (id, заполнен ли)."""
    p = report_paths(demo)
    summary, manifest = read_json(p["summary"]), read_json(p["manifest"])
    if summary is not None and bool(summary.get("demo")) != demo:
        raise ValueError(f"{p['summary']}: demo={summary.get('demo')}, а отчёт собирается с demo={demo}")
    cfg = run_bank.load_experiment()
    models_cfg = llm.load_models()
    pricing = an.load_pricing(an.paths(demo)["pricing"])
    tasks = bank.load_tasks(bank.BANK_PATH)
    h = Human(old_text)

    summary_sha = (hashlib.sha256(p["summary"].read_bytes()).hexdigest()[:12] if summary else "нет данных")
    header = ["# ЛР3. Можно ли доверять LLM в задачах классического машинного обучения"]
    if demo:
        header.append("> **ДЕМО: mock-модели и случайные оценки. Это не результаты эксперимента.**")
    header += [
        "> Файл собирает `.venv/bin/python -m src.build_report` из `results/summary_metrics.json` "
        f"(sha256 `{summary_sha}`), `results/run_manifest.json`, `config/*.json`, `data/tasks.jsonl`, "
        "`hypotheses.md` (версия из коммита) и `src/prompts.py`. Числа руками не править: после пересчёта "
        "`analysis.ipynb` пересоберите отчёт. Свой текст пишите только между строками `<!-- ЧЕЛОВЕК:… -->` и "
        "`<!-- /ЧЕЛОВЕК:… -->`: при пересборке он сохраняется.",
        h("authors", "авторы: ФИО, группа, кто проверял вслепую; дата"),
    ]
    sections = [
        "\n\n".join(header),
        section_intro(h, cfg),
        section_methods(h, cfg, models_cfg, manifest, summary, tasks, pricing),
        section_results(h, summary, cfg, p["figures"]),
        section_errors(h, p["errors"]),
        section_limits_recs(h, limitation_facts(cfg, manifest, summary, pricing, tasks)),
        "## Приложение: воспроизводимость\n\nПорядок «с нуля до отчёта» — `README.md`; проверка в чистом окружении "
        "— `docs/reproducibility.md` (`scripts/check_reproducibility.sh`). Вопросы самоконтроля — "
        "[`selfcheck_questions.md`](selfcheck_questions.md), план доклада — [`slides_outline.md`](slides_outline.md).",
    ]
    orphans = h.orphans()
    if orphans:
        sections.append(orphans)
    return "\n\n".join(sections) + "\n", h.used


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Каркас отчёта → report/report.md")
    ap.add_argument("command", nargs="?", default="build", choices=["build", "check"])
    ap.add_argument("--demo", action="store_true", help="demo/: mock-модели, не результаты")
    args = ap.parse_args(argv)
    path = report_paths(args.demo)["report"]
    old = path.read_text(encoding="utf-8") if path.exists() else ""

    if args.command == "check":
        if not old:
            print(f"Нет {path}: сначала .venv/bin/python -m src.build_report")
            return 1
        _, blocks = build(args.demo, old)   # блок не заполнен, если в нём текст по умолчанию
        empty = [b for b, filled in blocks if not filled]
        print("Блоки человека: всего", len(blocks), "| не заполнены:", ", ".join(empty) if empty else "нет")
        return 1 if empty else 0

    h = Human(old)
    if h.n_open != len(h.old):
        print(f"{path}: начато блоков «<!-- ЧЕЛОВЕК:» {h.n_open}, распознано {len(h.old)}. Какой-то блок испорчен "
              "(нет закрывающей строки «<!-- /ЧЕЛОВЕК:id -->» или опечатка в id). Отчёт НЕ перезаписан: "
              "исправьте блок и повторите.")
        return 1
    text, blocks = build(args.demo, old)
    path.parent.mkdir(parents=True, exist_ok=True)
    if old and old != text:   # прошлая версия на всякий случай (report/*.bak не коммитится)
        path.with_name(path.name + ".bak").write_text(old, encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    filled = sum(f for _, f in blocks)
    print(f"Записано: {run_bank.show(path)}; блоков человека {len(blocks)}, заполнено {filled}.")
    if not report_paths(args.demo)["summary"].exists():
        print("Нет summary_metrics.json: разделы с результатами помечены «нет данных».")
    return 0


if __name__ == "__main__":
    sys.exit(main())
