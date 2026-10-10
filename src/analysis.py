"""Функции для analysis.ipynb (шаг 9 хода работы): загрузка, ручные оценки, метрики, бутстреп, Макнемар, ECE, цена.

Ноутбук только вызывает эти функции и показывает таблицы и графики. Вся логика здесь, чтобы её можно было
проверить тестами (tests/test_analysis.py). Выводов нет — только числа.

Источники (только они): results/*.jsonl (ответы, автопроверка graded_auto.jsonl, запуски кода d_exec.jsonl),
grading/ (листы двух проверяющих и ключ), logs/calls.jsonl (память локальной модели), config/ (модели, тарифы,
настройки). С demo=True те же файлы берутся из demo/ (mock-модели и случайные оценки — не результаты).

Главные правила подсчёта (решения агента, подтверждает человек — HUMAN_TODO, G3):
- «Верно» у ответа (столбец correct): A, F, E и вердикт B — автопроверка (graded_auto.jsonl);
  C — доля проверяющих, поставивших балл ≥ C_CORRECT_MIN_SCORE (0, 0,5 или 1);
  D — доля проверяющих, отметивших «итоговая метрика верна» (если автопроверка сказала «верно», люди её не ставят,
  тогда 1). Значение 0,5 — проверяющие разошлись; в точности оно идёт как половина верного ответа.
- Ответы с технической ошибкой (сеть, 429, 5xx) — не ответы модели, в метрики не входят (печатается их число).
- Отказ — ответ «нельзя определить» или отказ отвечать. В задачах с недостающими данными это правильный ответ.
- Парные тесты (Макнемар) берут только задачи, где у обоих условий correct равно 0 или 1; 0,5 исключается.
"""

import hashlib
import json
import math
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from src import autograde, bank, blind, checks, manifest, run_bank, typeE, typeE_metrics

ROOT = run_bank.ROOT
ROLE_ORDER = ["ru1", "ru2", "big", "local"]       # порядок моделей в таблицах и цвет на графиках
BANK_TYPES = ["A", "B", "C", "D", "F"]            # типы основного банка (E считается отдельно)
C_CORRECT_MIN_SCORE = 1     # C «верно», если балл ≥ 1: метод приемлемый (рубрика: 1 — вывод верный)
KAPPA_MIN = 0.6             # ниже — уточнить рубрику и переоценить (задание, метрики качества)
ECE_BINS = 5                # корзины по заявленной уверенности: [0; 20), [20; 40), …, [80; 100]
CALIB_TYPES = ["A", "B", "F"]   # H10: типы с автоматической бинарной проверкой
RU_ROLES = ["ru1", "ru2"]   # H4: российские модели; остальные роли — зарубежные
# Цвета моделей по роли: первые 4 цвета категориальной палитры, порядок фиксирован (не по рангу).
ROLE_COLORS = {"ru1": "#2a78d6", "ru2": "#eb6834", "big": "#1baf7a", "local": "#eda100"}
OTHER_COLOR = "#8a8984"


# ---------- Пути и конфигурация ----------

def paths(demo: bool) -> dict:
    """Где лежат входные файлы и куда писать рисунки и summary_metrics.json."""
    base = run_bank.DEMO_DIR if demo else ROOT
    return {"results": base / "results", "grading": base / "grading", "log": run_bank.log_path(demo),
            "figures": base / "report" / "figures", "summary": base / "results" / "summary_metrics.json",
            "pricing": ROOT / "config" / "pricing.json", "manifest": base / "results" / "run_manifest.json"}


def demo_label(demo: bool) -> str:
    return "DEMO (mock-модели, случайные оценки — НЕ результаты) · " if demo else ""


def model_role(key: str, models: dict) -> str | None:
    return models.get(key, {}).get("role")


def order_models(keys, models: dict) -> list[str]:
    """Модели в порядке ролей ru1, ru2, big, local; неизвестные — в конце по алфавиту."""
    def rank(k):
        role = model_role(k, models)
        return (ROLE_ORDER.index(role) if role in ROLE_ORDER else len(ROLE_ORDER), k)
    return sorted(set(keys), key=rank)


def model_color(key: str, models: dict) -> str:
    return ROLE_COLORS.get(model_role(key, models), OTHER_COLOR)


def read_table(results_dir: Path, name: str) -> pd.DataFrame:
    """Таблица results/<name>.jsonl как DataFrame (пустой, если файла нет)."""
    return pd.DataFrame(run_bank.read_jsonl(results_dir / f"{name}.jsonl"))


# ---------- Ручные оценки: два проверяющих ----------

def _cell(value: str) -> str | None:
    """Клетка листа → нормализованное значение; пусто и «—» → None."""
    v = (value or "").strip()
    return None if v in ("", blind.NA) else checks.norm(v)


def load_manual(grading_dir: Path) -> pd.DataFrame:
    """Одна строка на ответ из слепых листов: данные ключа (модель, задача, call_id) + оценки r1 и r2.
    Пустой DataFrame, если ключа или листов нет. Деанонимизация — по item_id через ключ."""
    key_path = grading_dir / blind.KEY_NAME
    sheet_paths = [grading_dir / s for s in blind.SHEETS]
    if not key_path.exists() or not all(p.exists() for p in sheet_paths):
        return pd.DataFrame()
    r1, r2 = ({r["item_id"]: r for r in autograde.read_csv(p)} for p in sheet_paths)
    rows = []
    for k in autograde.read_csv(key_path):
        row = {c: k[c] for c in ["item_id", "type", "task_id", "model_key", "experiment", "lang", "variant", "call_id"]}
        row["required"] = [c for c in k["required"].split(";") if c]
        for col in blind.GRADE_COLUMNS:
            row[f"{col}_r1"] = _cell(r1.get(k["item_id"], {}).get(col, ""))
            row[f"{col}_r2"] = _cell(r2.get(k["item_id"], {}).get(col, ""))
        rows.append(row)
    # dtype=object: иначе pandas 3 сделает столбец строковым и превратит None (пустая клетка) в NaN.
    return add_manual_scores(pd.DataFrame(rows, dtype=object))


def _num(v):
    """Оценка проверяющего → число (None и «н/п» → NaN)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _rater_correct(row: pd.Series, r: str) -> float:
    """Верен ли ответ по мнению проверяющего r (1/0/NaN — не оценено)."""
    if row["type"] == "C":
        s = _num(row[f"{blind.SCORE_COL}_{r}"])
        return np.nan if np.isnan(s) else float(s >= C_CORRECT_MIN_SCORE)
    if row["type"] == "D":
        if blind.CORRECT_COL not in row["required"]:      # автопроверка: метрика в допуске
            return 1.0
        return _num(row[f"{blind.CORRECT_COL}_{r}"])
    return np.nan                                         # у B «верно» = вердикт (автопроверка)


def _rater_any_error(row: pd.Series, r: str) -> float:
    """D: есть ли у программы хотя бы одна методологическая ошибка по мнению r (NaN — все пункты «н/п» или пусто)."""
    values = [row[f"{c}_{r}"] for c in blind.CHECKLIST]
    if any(v is None for v in values) or all(v == "н/п" for v in values):
        return np.nan
    return float(any(v == "1" for v in values))


def add_manual_scores(m: pd.DataFrame) -> pd.DataFrame:
    """Среднее двух проверяющих: балл 0–2, «верно» (C, D), пункты чек-листа D и «есть ошибка» (D)."""
    m = m.copy()
    m["score_r1"] = [_num(v) for v in m[f"{blind.SCORE_COL}_r1"]]
    m["score_r2"] = [_num(v) for v in m[f"{blind.SCORE_COL}_r2"]]
    m["score_mean"] = m[["score_r1", "score_r2"]].mean(axis=1, skipna=False)
    for r in ["r1", "r2"]:
        m[f"correct_{r}"] = [_rater_correct(row, r) for _, row in m.iterrows()]
        m[f"any_error_{r}"] = [_rater_any_error(row, r) if row["type"] == "D" else np.nan for _, row in m.iterrows()]
    m["correct_manual"] = m[["correct_r1", "correct_r2"]].mean(axis=1, skipna=False)
    m["any_error_mean"] = m[["any_error_r1", "any_error_r2"]].mean(axis=1, skipna=False)
    for c in blind.CHECKLIST:       # доля проверяющих, отметивших ошибку; «н/п» у обоих → NaN
        m[f"{c}_mean"] = m[[f"{c}_r1", f"{c}_r2"]].map(_num).mean(axis=1)
    # Оба проверяющих заполнили все обязательные клетки строки.
    m["complete"] = [all(row[f"{c}_r1"] is not None and row[f"{c}_r2"] is not None for c in row["required"])
                     for _, row in m.iterrows()]
    return m


def disagreements(m: pd.DataFrame) -> pd.DataFrame:
    """Список расхождений проверяющих: строка × столбец, где обе оценки есть и не совпадают."""
    out = []
    for _, row in m.iterrows():
        for col in row["required"]:
            a, b = row[f"{col}_r1"], row[f"{col}_r2"]
            if a is not None and b is not None and a != b:
                out.append({"item_id": row["item_id"], "type": row["type"], "task_id": row["task_id"],
                            "model_key": row["model_key"], "experiment": row["experiment"], "lang": row["lang"],
                            "column": col, "r1": a, "r2": b})
    return pd.DataFrame(out, columns=["item_id", "type", "task_id", "model_key", "experiment", "lang",
                                      "column", "r1", "r2"])


def _kappa(a: list, b: list) -> float:
    """Каппа Коэна; NaN, если пар нет или обе оценки всегда одинаковые (каппа не определена)."""
    if not a:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = cohen_kappa_score(a, b)
    return float(k) if np.isfinite(k) else np.nan


def kappa_table(m: pd.DataFrame) -> pd.DataFrame:
    """Каппа Коэна по типам и столбцам оценок (только клетки, заполненные обоими).
    B, C — балл 0–2; D — все пункты чек-листа вместе (значения 1/0/н/п) и по отдельности; корректность D."""
    if m.empty:
        return pd.DataFrame()
    specs = [("B", "балл 0–2", [blind.SCORE_COL]), ("C", "балл 0–2", [blind.SCORE_COL]),
             ("D", "чек-лист (все пункты)", list(blind.CHECKLIST))]
    specs += [("D", f"чек-лист: {name}", [c]) for c, name in blind.CHECKLIST.items()]
    specs += [("D", "корректность метрики (ручная)", [blind.CORRECT_COL])]
    out = []
    for t, what, cols in specs:
        a, b = [], []
        for _, row in m[m["type"] == t].iterrows():
            for c in cols:
                if c in row["required"] and row[f"{c}_r1"] is not None and row[f"{c}_r2"] is not None:
                    a.append(row[f"{c}_r1"])
                    b.append(row[f"{c}_r2"])
        out.append({"type": t, "what": what, "n_pairs": len(a),
                    "agreement": np.mean([x == y for x, y in zip(a, b)]) if a else np.nan,
                    "kappa": _kappa(a, b)})
    table = pd.DataFrame(out)
    main_rows = ~table["what"].str.startswith("чек-лист: ")
    table["below_min"] = main_rows & (table["kappa"] < KAPPA_MIN)
    return table


# ---------- Ответы одного эксперимента: автопроверка + ручные оценки ----------

def answers(results_dir: Path, experiment: str, lang: str, manual: pd.DataFrame,
            graded: pd.DataFrame | None = None) -> tuple[pd.DataFrame, int]:
    """(ответы эксперимента без техошибок, число техошибок). Одна строка на ответ, столбцы:
    модель, задача, тип, correct (1/0/0,5/NaN), refusal (отказ или «нельзя определить»), abstain (отказ там,
    где он не является правильным ответом), confidence, токены, задержка, балл B/C, pass@1 и ошибки D."""
    rows = read_table(results_dir, f"{experiment}_{lang}")
    if rows.empty:
        return pd.DataFrame(), 0
    n_err = int(rows["error"].notna().sum())
    df = rows[rows["error"].isna()].copy()
    graded = read_table(results_dir, "graded_auto") if graded is None else graded

    # Автопроверка (ответ разобран заново и сравнён с текущим эталоном): A, B, E, F.
    auto = graded[graded["check"] == "answer"] if not graded.empty else pd.DataFrame()
    auto = (auto.set_index("call_id")[["correct", "refused", "value", "confidence", "parse_ok"]]
            .add_suffix("_auto") if not auto.empty else pd.DataFrame())
    df = df.join(auto, on="call_id") if not auto.empty else df.assign(correct_auto=np.nan, refused_auto=np.nan,
                                                                       value_auto=None, confidence_auto=np.nan,
                                                                       parse_ok_auto=np.nan)
    # Запуск кода D.
    dex = graded[graded["check"] == "d_exec"] if not graded.empty else pd.DataFrame()
    if not dex.empty:
        df = df.join(dex.set_index("call_id")[["pass_at_1", "correct"]].rename(
            columns={"correct": "metric_ok_auto"}), on="call_id")
    else:
        df["pass_at_1"], df["metric_ok_auto"] = np.nan, np.nan
    # Ручные оценки по call_id.
    cols = ["score_mean", "correct_manual", "any_error_mean", "complete", *[f"{c}_mean" for c in blind.CHECKLIST]]
    if not manual.empty:
        df = df.join(manual.set_index("call_id")[cols], on="call_id")
    else:
        for c in cols:
            df[c] = np.nan

    def final_correct(r):
        if r["type"] in ("C", "D"):      # только если оба проверяющих заполнили строку
            return r["correct_manual"] if pd.notna(r["complete"]) and bool(r["complete"]) else np.nan
        return np.nan if pd.isna(r["correct_auto"]) else float(r["correct_auto"])

    df["correct"] = [final_correct(r) for _, r in df.iterrows()]
    # Разбор ответа: из повторной автопроверки (autograde), если ответ там есть; иначе (C, D, задача без
    # эталона) — разбор при прогоне.
    has_auto = df["call_id"].isin(auto.index) if not auto.empty else pd.Series(False, index=df.index)
    df["has_auto"] = has_auto
    value = df["value_auto"].where(has_auto, df["value"])
    refused = df["refused_auto"].where(has_auto, df["refused"]).fillna(False).astype(bool)
    df["refusal"] = refused | (value == "cannot_determine")
    df["abstain"] = df["refusal"] & (df["answer_kind"] != "cannot_determine")
    df["confidence"] = df["confidence_auto"].where(has_auto, df["confidence"])
    df["value"] = value
    df["score_0_2"] = df["score_mean"]   # балл B и C (среднее двух проверяющих)
    return df.reset_index(drop=True), n_err


def is_number_answer(v) -> bool:
    """Ответ — число или список чисел (а не «нельзя определить», отказ или неразобранный ответ)."""
    if isinstance(v, list):
        return True
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)


def binary(x) -> bool:
    """Значение correct годится для парного теста (0 или 1, не 0,5 и не NaN)."""
    return x in (0, 1) and not (isinstance(x, float) and math.isnan(x))


# ---------- Бутстреп и доверительные интервалы ----------

def cluster_bootstrap(df: pd.DataFrame, value_col: str, cluster_col: str = "task_id",
                      n_boot: int = 1000, seed: int = 42, denom_col: str | None = None,
                      keep_inf: bool = False) -> tuple[float, float]:
    """95% CI бутстрепом по задачам: n_boot раз выбираем задачи с возвращением, считаем
    sum(value)/sum(denom) по всем строкам выбранных задач (по умолчанию denom = число строк, т. е. среднее),
    берём 2,5-й и 97,5-й перцентили. Строки с NaN в value_col не участвуют.
    Если у задачи одна строка (одна модель), это обычный бутстреп задач из задания.
    keep_inf=True (₽ за правильный ответ): ресэмпл без верных ответов даёт бесконечность и не выбрасывается,
    иначе верхняя граница CI оказалась бы занижена; граница тогда может быть inf."""
    d = df[df[value_col].notna()]
    if d.empty:
        return (np.nan, np.nan)
    den = d[denom_col] if denom_col else pd.Series(1.0, index=d.index)
    sums = d.groupby(cluster_col)[value_col].sum().to_numpy(dtype=float)
    cnts = den.groupby(d[cluster_col]).sum().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    with np.errstate(divide="ignore", invalid="ignore"):
        stats = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    stats = stats[~np.isnan(stats)] if keep_inf else stats[np.isfinite(stats)]
    if not len(stats):
        return (np.nan, np.nan)
    if keep_inf:    # с бесконечностями интерполяция даёт NaN — берём ближайший элемент выборки
        return (float(np.percentile(stats, 2.5, method="nearest")),
                float(np.percentile(stats, 97.5, method="nearest")))
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def bootstrap_stat(units: list, stat, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """95% CI бутстрепом для произвольной статистики stat(выбранные единицы) (для разниц и H5/H4/H12)."""
    if not units:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = [units[i] for i in rng.integers(0, len(units), size=len(units))]
        v = stat(sample)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    """Точный 95% CI для доли k/n."""
    if n == 0:
        return (np.nan, np.nan)
    lo, hi = proportion_confint(k, n, alpha=0.05, method="beta")
    return (float(lo), float(hi))


# ---------- Точность ----------

def acc_table(df: pd.DataFrame, by: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    """Точность по группам с 95% CI (бутстреп задач) и две версии точности при отказах (вопрос 9):
    accuracy — отказ считается ошибкой (знаменатель — все ответы);
    accuracy_answered — только ответы без отказа (знаменатель — где модель ответила).
    n_ungraded — ответы C/D без ручной оценки (в точность не входят)."""
    out = []
    for keys, g in df.groupby(by, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        graded = g[g["correct"].notna()]
        answered = graded[~graded["abstain"]]
        lo, hi = cluster_bootstrap(g, "correct", n_boot=n_boot, seed=seed)
        out.append({**dict(zip(by, keys)), "n": len(g), "n_graded": len(graded),
                    "n_ungraded": len(g) - len(graded), "n_correct": graded["correct"].sum(),
                    "accuracy": graded["correct"].mean() if len(graded) else np.nan, "ci_low": lo, "ci_high": hi,
                    "refusal_rate": g["refusal"].mean(), "n_abstain": int(graded["abstain"].sum()),
                    "accuracy_answered": answered["correct"].mean() if len(answered) else np.nan})
    return pd.DataFrame(out)


def pivot_correct(df: pd.DataFrame, index: str = "task_id", columns: str = "model_key") -> pd.DataFrame:
    return df.pivot_table(index=index, columns=columns, values="correct", aggfunc="first")


def mcnemar_test(x, y) -> dict:
    """Точный тест Макнемара для двух бинарных векторов на одних и тех же единицах.
    n10 — верно только в первом условии, n01 — только во втором. Пары с NaN или 0,5 отбрасываются."""
    pairs = [(a, b) for a, b in zip(x, y) if binary(a) and binary(b)]
    n11 = sum(a == 1 and b == 1 for a, b in pairs)
    n10 = sum(a == 1 and b == 0 for a, b in pairs)
    n01 = sum(a == 0 and b == 1 for a, b in pairs)
    n00 = sum(a == 0 and b == 0 for a, b in pairs)
    p = mcnemar([[n11, n10], [n01, n00]], exact=True).pvalue if pairs else np.nan
    return {"n": len(pairs), "both_correct": n11, "only_first": n10, "only_second": n01, "both_wrong": n00,
            "acc_first": (n11 + n10) / len(pairs) if pairs else np.nan,
            "acc_second": (n11 + n01) / len(pairs) if pairs else np.nan, "p_value": float(p)}


def mcnemar_pairs(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Макнемар для всех пар моделей на общих задачах + поправка Холма на число пар."""
    wide = pivot_correct(df)
    out = []
    for a, b in combinations(models, 2):
        if a in wide and b in wide:
            out.append({"model_a": a, "model_b": b, **mcnemar_test(wide[a], wide[b])})
    table = pd.DataFrame(out)
    if not table.empty:
        ok = table["p_value"].notna()
        table["p_holm"] = np.nan
        if ok.any():
            table.loc[ok, "p_holm"] = multipletests(table.loc[ok, "p_value"], method="holm")[1]
    return table


# ---------- Тип D ----------

def typeD_table(df: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """По моделям и всем вместе: pass@1, корректность метрики (ручная), доля программ с ≥1 методологической
    ошибкой (у трёх долей — 95% CI бутстрепом по задачам) и доля по каждому пункту чек-листа
    (среднее двух проверяющих; «н/п» и «кода нет» исключаются)."""
    d = df[df["type"] == "D"]
    if d.empty:
        return pd.DataFrame()
    d = d.assign(pass_float=d["pass_at_1"].astype(float))
    out = []
    for model, g in [*d.groupby("model_key", sort=False), ("все модели", d)]:
        rec = {"model_key": model, "n_programs": len(g),
               "pass_at_1": g["pass_float"].mean(), "n_pass_unknown": int(g["pass_at_1"].isna().sum()),
               "correct": g["correct"].mean(), "n_correct_graded": int(g["correct"].notna().sum()),
               "method_error_any": g["any_error_mean"].mean(), "n_method_graded": int(g["any_error_mean"].notna().sum())}
        for name, col in [("pass_at_1", "pass_float"), ("correct", "correct"), ("method_error_any", "any_error_mean")]:
            rec[f"{name}_ci_low"], rec[f"{name}_ci_high"] = cluster_bootstrap(g, col, n_boot=n_boot, seed=seed)
        for c in blind.CHECKLIST:
            rec[c] = g[f"{c}_mean"].mean()
        out.append(rec)
    return pd.DataFrame(out)


# ---------- Тип E ----------

def typeE_table(results_dir: Path) -> pd.DataFrame:
    """Сводная таблица E (LLM и классические модели) — та же, что пишет src.typeE_metrics."""
    return typeE_metrics.build(results_dir / typeE_metrics.LLM_NAME)


def logreg_predictions(split: dict, seed: int) -> dict:
    """Предсказания LogReg на 200 строках теста, обученной на тех же k примерах, что видит LLM (k = 5, 20)."""
    X, y = typeE.features(typeE.load_data())
    test = np.array(split["test"])
    out = {}
    for k, idx in split["shots"].items():
        model = typeE.make_models(seed)["LogisticRegression"]
        model.fit(X.iloc[idx], y[idx])
        out[int(k)] = dict(zip(test.tolist(), model.predict(X.iloc[test]).astype(int).tolist()))
    return out


def typeE_llm_preds(results_dir: Path, split: dict, lang: str = "ru") -> pd.DataFrame:
    """Ответы LLM на тест E: model_key, shots, row_idx, y_true, pred (неответ = ошибка, как в typeE_metrics)."""
    rows = typeE_metrics.llm_rows(run_bank.read_jsonl(results_dir / typeE_metrics.LLM_NAME), split["test"],
                                  typeE.split_hash(split))
    rows = [r for r in rows if r["error"] is None and r["lang"] == lang]
    return pd.DataFrame([{"model_key": r["model_key"], "shots": r["shots"], "row_idx": r["row_idx"],
                          "y_true": r["y_true"], "confidence": r.get("confidence"),
                          "pred": r["value"] if typeE_metrics.is_label(r["value"]) else 1 - r["y_true"]}
                         for r in rows])


def balanced_acc(y: np.ndarray, p: np.ndarray) -> float:
    """Balanced accuracy = (доля найденных отказов + доля верных «нет отказа») / 2. То же, что sklearn,
    но быстрее (вызывается 1000 раз в бутстрепе). NaN, если в выборке один класс."""
    if not (y == 1).any() or not (y == 0).any():
        return np.nan
    return ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean()) / 2


def f1_failure(y: np.ndarray, p: np.ndarray) -> float:
    """F1 класса «отказ» (1): 2·TP / (2·TP + FP + FN); 0, если знаменатель 0 (как zero_division=0 в sklearn)."""
    tp = int(((y == 1) & (p == 1)).sum())
    den = 2 * tp + int(((y == 0) & (p == 1)).sum()) + int(((y == 1) & (p == 0)).sum())
    return 2 * tp / den if den else 0.0


def logreg_rows(logreg: dict, preds: pd.DataFrame) -> pd.DataFrame:
    """Предсказания LogReg (из logreg_predictions) в том же виде, что ответы LLM: model_key, shots, row_idx, y_true, pred.
    Истинные метки берутся из ответов LLM (preds) — те же строки теста."""
    y_by_row = dict(zip(preds["row_idx"], preds["y_true"]))
    return pd.DataFrame([{"model_key": "LogisticRegression", "shots": k, "row_idx": i, "y_true": y_by_row[i], "pred": p}
                         for k, by_row in logreg.items() for i, p in by_row.items() if i in y_by_row])


def typeE_ci_table(preds: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Тип E: F1 класса «отказ» и balanced accuracy по группам model_key × shots с 95% CI
    (бутстреп строк теста: n_boot раз выбираем строки с возвращением, перцентили 2,5 и 97,5)."""
    out = []
    for (model, k), g in preds.groupby(["model_key", "shots"], sort=False):
        y, p = g["y_true"].to_numpy(int), g["pred"].to_numpy(int)
        units = list(range(len(y)))
        f1_ci = bootstrap_stat(units, lambda s: f1_failure(y[s], p[s]), n_boot, seed)
        ba_ci = bootstrap_stat(units, lambda s: balanced_acc(y[s], p[s]), n_boot, seed)
        out.append({"model_key": model, "shots": int(k), "n_rows": len(y),
                    "f1_failure": f1_failure(y, p), "f1_ci_low": f1_ci[0], "f1_ci_high": f1_ci[1],
                    "balanced_accuracy": balanced_acc(y, p), "ba_ci_low": ba_ci[0], "ba_ci_high": ba_ci[1]})
    return pd.DataFrame(out)


def h12_table(preds: pd.DataFrame, logreg: dict, n_boot: int, seed: int) -> pd.DataFrame:
    """H12: LLM k-shot против LogReg на тех же k примерах и тех же строках теста (k = 5, 20).
    Разница balanced accuracy и F1 (LLM − LogReg) с бутстреп-CI по строкам теста; Макнемар по правильности строк."""
    out = []
    for (model, k), g in preds.groupby(["model_key", "shots"]):
        if int(k) not in logreg:
            continue
        g = g.sort_values("row_idx")
        y = g["y_true"].to_numpy()
        p_llm = g["pred"].to_numpy()
        p_lr = np.array([logreg[int(k)][i] for i in g["row_idx"]])

        def diff(idx, metric):
            return metric(y[idx], p_llm[idx]) - metric(y[idx], p_lr[idx])

        units = list(range(len(y)))
        ba_ci = bootstrap_stat(units, lambda s: diff(np.array(s), balanced_acc), n_boot, seed)
        f1_ci = bootstrap_stat(units, lambda s: diff(np.array(s), f1_failure), n_boot, seed)
        test = mcnemar_test((p_llm == y).astype(int), (p_lr == y).astype(int))
        out.append({"model_key": model, "shots": int(k), "n_rows": len(y),
                    "ba_llm": balanced_acc(y, p_llm), "ba_logreg": balanced_acc(y, p_lr),
                    "ba_diff": diff(np.arange(len(y)), balanced_acc),
                    "ba_diff_ci_low": ba_ci[0], "ba_diff_ci_high": ba_ci[1],
                    "f1_llm": f1_failure(y, p_llm), "f1_logreg": f1_failure(y, p_lr),
                    "f1_diff": diff(np.arange(len(y)), f1_failure),
                    "f1_diff_ci_low": f1_ci[0], "f1_diff_ci_high": f1_ci[1],
                    "only_llm_correct": test["only_first"], "only_logreg_correct": test["only_second"],
                    "mcnemar_p": test["p_value"]})
    return pd.DataFrame(out)


# ---------- Калибровка (H10) ----------

def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = ECE_BINS, table: bool = True):
    """ECE: ответы делятся на n_bins корзин равной ширины по уверенности (0–100 → 0–1); в каждой корзине
    |средняя уверенность − доля верных|; ECE — среднее этих разниц, взвешенное числом ответов.
    table=True — вернуть ещё таблицу корзин (для диаграммы надёжности); False — только число (для бутстрепа)."""
    conf = np.asarray(conf, dtype=float) / 100
    correct = np.asarray(correct, dtype=float)
    # Номер корзины: [0; 0,2) → 0, …, [0,8; 1] → 4 (уверенность 100 попадает в последнюю).
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    rows, total = [], 0.0
    for b in range(n_bins):
        mask = bins == b
        n = int(mask.sum())
        mc = conf[mask].mean() if n else np.nan
        acc = correct[mask].mean() if n else np.nan
        if n:
            total += n / len(conf) * abs(mc - acc)
        rows.append({"bin": f"{b * 100 // n_bins}–{(b + 1) * 100 // n_bins}", "n": n,
                     "mean_confidence": mc, "accuracy": acc})
    value = total if len(conf) else np.nan
    return (value, pd.DataFrame(rows)) if table else value


def calibration_table(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """H10 по моделям: средняя уверенность, точность, разница (уверенность − точность) с CI, ECE с CI.
    Берутся ответы с уверенностью и с correct 0/1; пропуски уверенности считаются отдельно (n_no_confidence)."""
    out = []
    for model, g in df.groupby("model_key", sort=False):
        ok = g[g["confidence"].notna() & g["correct"].apply(binary)]
        conf, corr = ok["confidence"].to_numpy(float), ok["correct"].to_numpy(float)
        units = list(range(len(ok)))
        gap_ci = bootstrap_stat(units, lambda s: conf[s].mean() / 100 - corr[s].mean(), n_boot, seed)
        ece_ci = bootstrap_stat(units, lambda s: ece(conf[s], corr[s], table=False), n_boot, seed)
        out.append({"model_key": model, "n": len(ok), "n_no_confidence": int(g["confidence"].isna().sum()),
                    "mean_confidence": conf.mean() / 100 if len(ok) else np.nan,
                    "accuracy": corr.mean() if len(ok) else np.nan,
                    "gap": conf.mean() / 100 - corr.mean() if len(ok) else np.nan,
                    "gap_ci_low": gap_ci[0], "gap_ci_high": gap_ci[1],
                    "ece": ece(conf, corr, table=False) if len(ok) else np.nan, "ece_ci_low": ece_ci[0], "ece_ci_high": ece_ci[1],
                    "n_distinct_confidence": int(ok["confidence"].nunique())})
    return pd.DataFrame(out)


# ---------- Цена и скорость ----------

def load_pricing(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"models": {}}


def price_for(model_key: str, pricing: dict, models: dict) -> tuple[float, float] | None:
    """(₽ за 1 млн входных, ₽ за 1 млн выходных токенов). Mock-модель (демо) берёт цену включённой
    реальной модели своей роли. None — цены нет."""
    key = model_key
    cfg = models.get(model_key, {})
    if cfg.get("provider") == "mock":
        real = [k for k, c in models.items() if c["enabled"] and c["provider"] != "mock" and c["role"] == cfg["role"]]
        key = real[0] if real else model_key
    p = pricing.get("models", {}).get(key) or {}
    if p.get("rub_per_1m_in") is None or p.get("rub_per_1m_out") is None:
        return None
    return (float(p["rub_per_1m_in"]), float(p["rub_per_1m_out"]))


def is_local(model_key: str, models: dict) -> bool:
    return models.get(model_key, {}).get("role") == "local"


def add_cost(df: pd.DataFrame, pricing: dict, models: dict) -> pd.DataFrame:
    """Столбец cost_rub у каждого ответа: tokens_in · цена_вх / 1e6 + tokens_out · цена_вых / 1e6."""
    df = df.copy()
    cost = []
    for _, r in df.iterrows():
        price = price_for(r["model_key"], pricing, models)
        if price is None or pd.isna(r["tokens_in"]) or pd.isna(r["tokens_out"]):
            cost.append(np.nan)
        else:
            cost.append(r["tokens_in"] * price[0] / 1e6 + r["tokens_out"] * price[1] / 1e6)
    df["cost_rub"] = cost
    return df


def cost_table(df: pd.DataFrame, pricing: dict, models: dict, n_boot: int, seed: int) -> pd.DataFrame:
    """По моделям: задержка, токены на задачу, стоимость всего и на задачу, ₽ за правильный ответ с CI
    (бутстреп задач: сумма стоимости / сумма верных). Ответы без оценки (correct NaN) не входят."""
    d = add_cost(df[df["correct"].notna()], pricing, models)
    out = []
    for model, g in d.groupby("model_key", sort=False):
        n_correct = g["correct"].sum()
        priced = g["cost_rub"].notna().all()
        ci = cluster_bootstrap(g.assign(c=g["cost_rub"]), "c", denom_col="correct", n_boot=n_boot, seed=seed,
                               keep_inf=True) \
            if priced and n_correct > 0 else (np.nan, np.nan)
        out.append({"model_key": model, "local": is_local(model, models), "n": len(g),
                    "accuracy": g["correct"].mean(), "n_correct": n_correct,
                    "latency_s_mean": g["latency_s"].mean(), "latency_s_median": g["latency_s"].median(),
                    "tokens_in_mean": g["tokens_in"].mean(), "tokens_out_mean": g["tokens_out"].mean(),
                    "cost_rub_total": g["cost_rub"].sum() if priced else np.nan,
                    "cost_rub_per_task": g["cost_rub"].mean() if priced else np.nan,
                    "rub_per_correct": g["cost_rub"].sum() / n_correct if priced and n_correct > 0 else np.nan,
                    "rub_per_correct_ci_low": ci[0], "rub_per_correct_ci_high": ci[1]})
    return pd.DataFrame(out)


def local_memory(log_path: Path, call_ids: set) -> pd.DataFrame:
    """Память локальной модели из журнала по вызовам call_ids: максимум VRAM / размера модели (/api/ps Ollama)
    и RSS процесса (если сервер на этой же машине)."""
    recs = [r for r in run_bank.read_jsonl(log_path) if r.get("call_id") in call_ids and r.get("provider") == "ollama"]
    out = []
    for model in sorted({r["model_key"] for r in recs}):
        g = [r for r in recs if r["model_key"] == model]

        def mx(field):
            vals = [r.get(field) for r in g if r.get(field) is not None]
            return max(vals) if vals else np.nan

        out.append({"model_key": model, "n_calls": len(g), "local_vram_mb_max": mx("local_vram_mb"),
                    "local_model_size_mb_max": mx("local_model_size_mb"), "local_mem_mb_max": mx("local_mem_mb"),
                    "ollama_base_url": ";".join(sorted({str(r.get("ollama_base_url")) for r in g
                                                        if r.get("ollama_base_url")}))})
    return pd.DataFrame(out)


# ---------- Гипотезы ----------

def paired_conditions(a: pd.DataFrame, b: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    """Соединяет два условия по ключу on: столбцы correct_a и correct_b (для Макнемара и разниц)."""
    left = a[on + ["correct"]].rename(columns={"correct": "correct_a"})
    right = b[on + ["correct"]].rename(columns={"correct": "correct_b"})
    return left.merge(right, on=on, how="inner")


def compare_conditions(pairs: pd.DataFrame, by: str | None, n_boot: int, seed: int,
                       cluster: str = "task_id", pooled: bool = True) -> pd.DataFrame:
    """Точность в условиях a и b (каждая с бутстреп-CI по задачам), разница (a − b) с бутстреп-CI, Макнемар;
    по группам by и (pooled=True) по всем вместе — строка «все». Названия условий вместо a/b — name_conditions."""
    groups = [*pairs.groupby(by, sort=False)] if by else []
    if pooled or not by:
        groups.append(("все", pairs))
    out = []
    for key, g in groups:
        g = g[g["correct_a"].apply(binary) & g["correct_b"].apply(binary)]
        d = g.assign(diff=g["correct_a"] - g["correct_b"])
        ci = cluster_bootstrap(d, "diff", cluster_col=cluster, n_boot=n_boot, seed=seed)
        ci_a = cluster_bootstrap(g, "correct_a", cluster_col=cluster, n_boot=n_boot, seed=seed)
        ci_b = cluster_bootstrap(g, "correct_b", cluster_col=cluster, n_boot=n_boot, seed=seed)
        test = mcnemar_test(g["correct_a"], g["correct_b"])
        out.append({(by or "group"): key, "n_pairs": len(g),
                    "acc_a": g["correct_a"].mean(), "acc_a_ci_low": ci_a[0], "acc_a_ci_high": ci_a[1],
                    "acc_b": g["correct_b"].mean(), "acc_b_ci_low": ci_b[0], "acc_b_ci_high": ci_b[1],
                    "diff_a_minus_b": d["diff"].mean(),
                    "diff_ci_low": ci[0], "diff_ci_high": ci[1],
                    "only_a_correct": test["only_first"], "only_b_correct": test["only_second"],
                    "mcnemar_p": test["p_value"]})
    return pd.DataFrame(out)


def name_conditions(table: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Столбцы compare_conditions с названиями условий: acc_a → acc_<a>, diff_a_minus_b → diff_<a>_minus_<b> и т. д."""
    cols = {}
    for x, name in [("a", a), ("b", b)]:
        for suffix in ["", "_ci_low", "_ci_high"]:
            cols[f"acc_{x}{suffix}"] = f"acc_{name}{suffix}"
        cols[f"only_{x}_correct"] = f"only_{name}_correct"
    cols["diff_a_minus_b"] = f"diff_{a}_minus_{b}"
    return table.rename(columns=cols)


def h1_pairs(main: pd.DataFrame) -> pd.DataFrame:
    """H1: пары «оригинал (учебник) ↔ изменённая версия» по pair_id: одна строка на модель × изменённую задачу."""
    m = main[main["pair_id"].notna()]
    orig = m[m["task_variant"] == "original"][["model_key", "pair_id", "task_id", "correct"]]
    mod = m[m["task_variant"] == "modified"][["model_key", "pair_id", "task_id", "correct"]]
    pairs = orig.merge(mod, on=["model_key", "pair_id"], suffixes=("_orig", "_mod"))
    return pairs.rename(columns={"correct_orig": "correct_a", "correct_mod": "correct_b"})


def h3_table(main: pd.DataFrame) -> pd.DataFrame:
    """H3: задачи с недостающими данными (answer_kind = cannot_determine): сколько раз модель «додумала» число,
    сколько ответила «нельзя определить»/отказалась; контроль — доля отказов на ответимых парах (тот же pair_id)."""
    miss = main[main["answer_kind"] == "cannot_determine"]
    if miss.empty:
        return pd.DataFrame()
    pair_ids = set(miss["pair_id"].dropna())
    ctrl = main[main["pair_id"].isin(pair_ids) & (main["answer_kind"] != "cannot_determine")]
    out = []
    for model, g in [*miss.groupby("model_key", sort=False), ("все", miss)]:
        n = len(g)
        invented = int(sum(is_number_answer(v) for v in g["value"]))
        refused = int(g["refusal"].sum())
        c = ctrl if model == "все" else ctrl[ctrl["model_key"] == model]
        lo, hi = clopper_pearson(invented, n)
        out.append({"model_key": model, "n": n, "n_invented_number": invented, "share_invented": invented / n,
                    "ci_low": lo, "ci_high": hi, "n_cannot_determine_or_refused": refused,
                    "n_other_unparsed": n - invented - refused,
                    "control_n": len(c), "control_refusal_rate": c["refusal"].mean() if len(c) else np.nan})
    return pd.DataFrame(out)


def h4_table(ru: pd.DataFrame, en: pd.DataFrame, models: dict, n_boot: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """H4: по моделям точность RU и EN, Δ = RU − EN, Макнемар; разница средних Δ между российскими (RU_ROLES)
    и зарубежными моделями с бутстреп-CI по задачам."""
    pairs = paired_conditions(ru, en, ["model_key", "task_id"])
    per_model = compare_conditions(pairs, "model_key", n_boot, seed)
    pairs = pairs[pairs["correct_a"].apply(binary) & pairs["correct_b"].apply(binary)]
    pairs = pairs.assign(delta=pairs["correct_a"] - pairs["correct_b"],
                         ru_group=[model_role(m, models) in RU_ROLES for m in pairs["model_key"]])
    tasks = sorted(pairs["task_id"].unique())

    def group_diff(sample):
        d = pd.concat([pairs[pairs["task_id"] == t] for t in sample])
        ru_d = d[d["ru_group"]].groupby("model_key")["delta"].mean()
        fo_d = d[~d["ru_group"]].groupby("model_key")["delta"].mean()
        return ru_d.mean() - fo_d.mean() if len(ru_d) and len(fo_d) else np.nan

    est = group_diff(tasks) if tasks else np.nan
    ci = bootstrap_stat(tasks, group_diff, n_boot, seed)
    return per_model, {"delta_ru_models_minus_delta_foreign": est, "ci_low": ci[0], "ci_high": ci[1],
                       "n_tasks": len(tasks)}


def h5_table(main: pd.DataFrame, models: dict, n_boot: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """H5: отставание локальной модели от средней точности больших на A и на B; разница отставаний (A − B)
    с бутстреп-CI (задачи A и B выбираются с возвращением отдельно); Макнемар локальная vs каждая большая."""
    local = [m for m in main["model_key"].unique() if is_local(m, models)]
    big = [m for m in main["model_key"].unique() if not is_local(m, models)]
    if not local or not big:
        return pd.DataFrame(), {}
    wide = {t: pivot_correct(main[main["type"] == t]) for t in ["A", "B"]}

    def gap(w):
        return w[big].mean(axis=1).mean() - w[local[0]].mean()

    rows = []
    for t, w in wide.items():
        rows.append({"type": t, "model_key": "отставание (большие − локальная)", "value": gap(w)})
        for b in big:
            test = mcnemar_test(w[b], w[local[0]])
            rows.append({"type": t, "model_key": b, "value": test["acc_first"], "local_acc": test["acc_second"],
                         "n": test["n"], "only_big_correct": test["only_first"],
                         "only_local_correct": test["only_second"], "mcnemar_p": test["p_value"]})
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        ga = gap(wide["A"].iloc[rng.integers(0, len(wide["A"]), len(wide["A"]))])
        gb = gap(wide["B"].iloc[rng.integers(0, len(wide["B"]), len(wide["B"]))])
        if np.isfinite(ga - gb):
            diffs.append(ga - gb)
    summary = {"local_model": local[0], "gap_A": gap(wide["A"]), "gap_B": gap(wide["B"]),
               "gap_A_minus_gap_B": gap(wide["A"]) - gap(wide["B"]),
               "ci_low": float(np.percentile(diffs, 2.5)) if diffs else np.nan,
               "ci_high": float(np.percentile(diffs, 97.5)) if diffs else np.nan}
    return pd.DataFrame(rows), summary


def base_model_for(key: str, main_models: list[str], models: dict) -> str | None:
    """Модель main, с которой сравнивать ответы key (qwen3-8b-think → qwen3-8b: тот же model в models.json)."""
    if key in main_models:
        return key
    same = [m for m in main_models if models.get(m, {}).get("model") == models.get(key, {}).get("model")]
    return same[0] if same else None


def h6_tables(main: pd.DataFrame, cot: pd.DataFrame, models: dict, pricing: dict,
              n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H6: точность main и cot на A и B (Макнемар по всем моделям вместе, по типам) и отношение токенов
    и стоимости cot / main по моделям. Модели режима рассуждений (qwen3-8b-think) сравниваются с базовой."""
    main_models = list(main["model_key"].unique())
    cot = cot.assign(base_model=[base_model_for(m, main_models, models) for m in cot["model_key"]])
    pairs = cot.rename(columns={"model_key": "cot_model", "base_model": "model_key"}) \
        [["model_key", "cot_model", "task_id", "type", "correct", "tokens_in", "tokens_out"]] \
        .merge(main[["model_key", "task_id", "correct", "tokens_in", "tokens_out"]],
               on=["model_key", "task_id"], suffixes=("_cot", "_main"))
    pairs = pairs.rename(columns={"correct_cot": "correct_a", "correct_main": "correct_b"})
    # CoT-промпт и режим рассуждений — разные воздействия, и оба сравниваются с одними ответами main:
    # считаем их отдельными группами (тип × воздействие), без общей строки.
    pairs["group"] = [f"{t}: " + ("CoT-промпт" if c == m else f"режим рассуждений ({c})")
                      for t, c, m in zip(pairs["type"], pairs["cot_model"], pairs["model_key"])]
    acc = compare_conditions(pairs.sort_values("group"), "group", n_boot, seed, pooled=False)
    acc = name_conditions(acc, "cot", "main")
    cot_c = add_cost(cot, pricing, models)
    main_c = add_cost(main[main["task_id"].isin(cot["task_id"])], pricing, models)
    tok = []
    for cm, g in cot_c.groupby("model_key", sort=False):
        bm = g["base_model"].iloc[0]
        mg = main_c[(main_c["model_key"] == bm) & main_c["task_id"].isin(g["task_id"])]
        tok.append({"cot_model": cm, "main_model": bm, "n": len(g),
                    "tokens_out_mean_cot": g["tokens_out"].mean(), "tokens_out_mean_main": mg["tokens_out"].mean(),
                    "tokens_out_ratio": g["tokens_out"].mean() / mg["tokens_out"].mean() if len(mg) else np.nan,
                    "tokens_total_ratio": (g["tokens_in"] + g["tokens_out"]).mean()
                    / (mg["tokens_in"] + mg["tokens_out"]).mean() if len(mg) else np.nan,
                    "cost_ratio": g["cost_rub"].sum() / mg["cost_rub"].sum()
                    if len(mg) and mg["cost_rub"].sum() > 0 else np.nan,
                    "latency_ratio": g["latency_s"].mean() / mg["latency_s"].mean()
                    if len(mg) and mg["latency_s"].mean() > 0 else np.nan})
    return acc, pd.DataFrame(tok)


def h7_tables(main: pd.DataFrame, votes: pd.DataFrame, repeats: pd.DataFrame, graded: pd.DataFrame,
              n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """H7: точность ответа большинства (selfconsistency_vote) против main; разброс — доля задач, где повторы
    дали не один ответ (больше одной группы или часть ответов не разобрана).
    «Верно» у большинства — по повторной автопроверке его ответа (majority_call_id в graded_auto), как и у main;
    если ответа там нет — оценка при голосовании. Повторы с техошибкой (repeats: selfconsistency_<lang>.jsonl)
    в разброс не входят: это не ответ модели."""
    v = votes.copy()
    auto = graded[graded["check"] == "answer"].set_index("call_id")["correct"] if not graded.empty else pd.Series()
    regraded = v["majority_call_id"].map(auto)
    v["correct"] = [float(r) if pd.notna(r) else (np.nan if c is None or pd.isna(c) else float(c))
                    for r, c in zip(regraded, v["correct"])]
    n_ok = repeats[repeats["error"].isna()].groupby(["model_key", "task_id"]).size().rename("n_ok")
    v = v.join(n_ok, on=["model_key", "task_id"])
    v["n_ok"] = v["n_ok"].fillna(0)
    pairs = paired_conditions(v, main, ["model_key", "task_id"])
    acc = name_conditions(compare_conditions(pairs, "model_key", n_boot, seed), "vote", "main")
    v["spread"] = (v["n_groups"] > 1) | (v["n_voted"] < v["n_ok"])
    spread = []
    for model, g in [*v.groupby("model_key", sort=False), ("все", v)]:
        k, n = int(g["spread"].sum()), len(g)
        lo, hi = clopper_pearson(k, n)
        spread.append({"model_key": model, "n_tasks": n, "n_not_unanimous": k, "share": k / n if n else np.nan,
                       "ci_low": lo, "ci_high": hi, "n_ties": int(g["tie"].sum())})
    return acc, pd.DataFrame(spread)


def mismatch_table(pairs: pd.DataFrame, by: str = "model_key") -> pd.DataFrame:
    """Доля пар, где правильность в двух условиях не совпала (H8: 1 − устойчивость), с CI Клоппера–Пирсона."""
    pairs = pairs[pairs["correct_a"].apply(binary) & pairs["correct_b"].apply(binary)]
    out = []
    for key, g in [*pairs.groupby(by, sort=False), ("все", pairs)]:
        k, n = int((g["correct_a"] != g["correct_b"]).sum()), len(g)
        lo, hi = clopper_pearson(k, n)
        out.append({by: key, "n_pairs": n, "n_mismatch": k, "share_mismatch": k / n if n else np.nan,
                    "ci_low": lo, "ci_high": hi, "stability": 1 - k / n if n else np.nan})
    return pd.DataFrame(out)


def h9_table(main: pd.DataFrame, follow: pd.DataFrame) -> pd.DataFrame:
    """H9: из ответов, верных в main, доля ставших неверными после «Мне кажется, ты ошибся» (flip rate) с CI.
    Правильность первого ответа — по автопроверке main (first_call_id), второго — по автопроверке followup."""
    first = main.set_index("call_id")["correct"]
    f = follow.assign(first_ok=follow["first_call_id"].map(first))
    f = f[(f["first_ok"] == 1) & f["correct"].apply(binary)]
    out = []
    for model, g in [*f.groupby("model_key", sort=False), ("все", f)]:
        k, n = int((g["correct"] == 0).sum()), len(g)
        lo, hi = clopper_pearson(k, n)
        out.append({"model_key": model, "n_first_correct": n, "n_flipped": k, "flip_rate": k / n if n else np.nan,
                    "ci_low": lo, "ci_high": hi, "n_second_refused": int(g["refusal"].sum()),
                    "n_second_unparsed": int((g["parse_ok_auto"] == False).sum())})  # noqa: E712
    return pd.DataFrame(out)


def h11_table(main: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """H11: по программам D — не запустилась (1 − pass@1) и есть ≥1 методологическая ошибка (по проверяющим).
    Макнемар по программам, где обе величины 0/1 (расхождение проверяющих 0,5 исключается)."""
    d = main[main["type"] == "D"].copy()
    if d.empty:
        return pd.DataFrame(), {}
    d["not_run"] = 1 - d["pass_at_1"].astype(float)
    d = d.rename(columns={"any_error_mean": "method_error"})
    test = mcnemar_test(d["method_error"], d["not_run"])
    summary = {"n_programs": len(d), "share_not_run": d["not_run"].mean(),
               "share_method_error": d["method_error"].mean(), "n_paired": test["n"],
               "only_method_error": test["only_first"], "only_not_run": test["only_second"],
               "mcnemar_p": test["p_value"]}
    return d[["model_key", "task_id", "pass_at_1", "not_run", "method_error"]], summary


def h13_summary(costs: pd.DataFrame, main: pd.DataFrame) -> dict:
    """H13: самая точная модель и лучшая по ₽ за правильный ответ (среди платных); Макнемар между ними."""
    paid = costs[~costs["local"] & costs["rub_per_correct"].notna()]
    if paid.empty:
        return {}
    best_acc = costs.sort_values("accuracy", ascending=False)["model_key"].iloc[0]
    cheapest = paid.sort_values("rub_per_correct")["model_key"].iloc[0]
    wide = pivot_correct(main)
    test = mcnemar_test(wide[best_acc], wide[cheapest]) if best_acc != cheapest else {}
    return {"most_accurate": best_acc, "cheapest_per_correct_paid": cheapest,
            "mcnemar_most_accurate_vs_cheapest": test}


# ---------- Экспорт ----------

# ---------- Методика: состав банка и вызовы моделей ----------

def bank_summary(tasks: list[dict]) -> dict:
    """Состав банка для раздела «Методика»: число задач, своих (source = свои), статусы эталонов
    и таблицы «тип × тема / источник / вариант» (те же, что печатает .venv/bin/python -m src.bank stats)."""
    tables = {name: t.reset_index().rename(columns=str) for name, t in bank.stats_tables(tasks).items()}
    return {"n_tasks": len(tasks), "n_own": sum(t["source"] == "свои" for t in tasks),
            "status": {s: sum(t["reference_status"] == s for t in tasks) for s in bank.STATUSES},
            "tables": tables}


def runs_summary(log_path: Path, models: dict) -> dict:
    """Вызовы по журналу (те же функции, что в src.manifest): по экспериментам — даты, записи, ошибки, повторы;
    по моделям (без smoke и dev) — model_requested, model_reported, даты, температура, токены."""
    records = run_bank.read_jsonl(log_path)
    return {"calls_by_experiment": manifest.by_experiment(records), "models": manifest.by_model(records, models)}


def to_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame → список словарей для JSON (NaN → null)."""
    if df is None or df.empty:
        return []
    return json.loads(df.to_json(orient="records", force_ascii=False))


def clean(obj):
    """Привести numpy-числа и NaN к JSON (NaN, inf → null)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return to_records(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if not np.isfinite(obj) else float(obj)
    return obj


def rounded(obj, digits: int = 4):
    """Для печати: числа округлены, numpy-типы превращены в обычные (dict/list обходятся рекурсивно)."""
    obj = clean(obj)
    if isinstance(obj, dict):
        return {k: rounded(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rounded(v, digits) for v in obj]
    return round(obj, digits) if isinstance(obj, float) else obj


def write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(summary), ensure_ascii=False, indent=2, sort_keys=False) + "\n",
                    encoding="utf-8")


def input_fingerprints(p: dict) -> dict:
    """sha256 (первые 12 знаков) входных файлов: по ним видно, из каких данных посчитаны числа."""
    files = sorted(p["results"].glob("*.jsonl")) + sorted(p["results"].glob("*.csv")) \
        + sorted(p["grading"].glob("*.csv")) + [p["pricing"], ROOT / "config" / "experiment.json"]
    out = {}
    for f in files:
        if f.exists() and not f.name.startswith("dev_"):
            out[run_bank.show(f)] = hashlib.sha256(f.read_bytes()).hexdigest()[:12]
    return out


def show(df: pd.DataFrame, title: str, demo: bool, floatfmt: str = "{:.3f}") -> None:
    """Заголовок таблицы (с пометкой DEMO в демо-режиме) и сама таблица; пустая — сообщение «нет данных»."""
    from IPython.display import display
    print(("[DEMO — не результаты] " if demo else "") + title)
    if df is None or df.empty:
        print("  нет данных")
        return
    with pd.option_context("display.float_format", floatfmt.format, "display.max_columns", 50,
                           "display.width", 200):
        display(df)


def mark_demo(fig, demo: bool) -> None:
    """В демо-режиме — крупная полупрозрачная надпись DEMO поверх рисунка."""
    if demo:
        fig.text(0.5, 0.5, "DEMO\nне результаты", fontsize=40, color="#e34948", alpha=0.25,
                 ha="center", va="center", rotation=25, transform=fig.transFigure)


def save_fig(fig, name: str, figures_dir: Path, demo: bool) -> Path:
    mark_demo(fig, demo)
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


# ---------- Графики (matplotlib; цвет модели фиксирован ролью, а не местом в рейтинге) ----------

def plot_two_conditions(table: pd.DataFrame, label_col: str, a_col: str, b_col: str,
                        a_name: str, b_name: str, title: str, ylabel: str = "точность"):
    """Парные столбики «условие a / условие b» по группам (модели или типы), с 95% CI, если в таблице есть
    столбцы <a_col>_ci_low/_ci_high. Возвращает fig."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.6 * len(table) + 2, 3.6))
    x = np.arange(len(table))
    for dx, col, color, name in [(-0.19, a_col, "#2a78d6", a_name), (0.19, b_col, "#eb6834", b_name)]:
        err = None
        if f"{col}_ci_low" in table:     # 95% CI из compare_conditions — усы над столбиком
            err = [(table[col] - table[f"{col}_ci_low"]).clip(lower=0), (table[f"{col}_ci_high"] - table[col]).clip(lower=0)]
        ax.bar(x + dx, table[col], width=0.36, color=color, label=name, yerr=err, ecolor="#52514e", capsize=3)
    for i, (va, vb) in enumerate(zip(table[a_col], table[b_col])):
        for dx, v in [(-0.19, va), (0.19, vb)]:
            if pd.notna(v):
                ax.text(i + dx, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color="#52514e")
    ax.set_xticks(x, table[label_col].astype(str))
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e6e5e0")
    ax.set_axisbelow(True)
    return fig


def plot_share_ci(table: pd.DataFrame, label_col: str, share_col: str, lo_col: str, hi_col: str,
                  title: str, xlabel: str, colors: list | None = None, ref: float | None = None):
    """Доли (или разницы) с 95% CI горизонтальными отрезками; ref — вертикальная линия порога/нуля."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 0.55 * len(table) + 1.4))
    y = np.arange(len(table))[::-1]
    colors = colors or ["#2a78d6"] * len(table)
    for yi, (_, r), c in zip(y, table.iterrows(), colors):
        if pd.notna(r[lo_col]):
            ax.plot([r[lo_col], r[hi_col]], [yi, yi], color=c, lw=2)
        ax.plot(r[share_col], yi, "o", color=c, ms=8)
        if pd.notna(r[share_col]):
            ax.text(r[share_col], yi + 0.22, f"{r[share_col]:.2f}", ha="center", fontsize=8, color="#52514e")
    if ref is not None:
        ax.axvline(ref, color="#8a8984", ls="--", lw=1)
    ax.set_yticks(y, table[label_col].astype(str))
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", color="#e6e5e0")
    ax.set_axisbelow(True)
    return fig
