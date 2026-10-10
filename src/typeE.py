"""Тип E «LLM как классификатор»: датасет AI4I 2020, разбиение, текст строки, классические модели.

Запуск (по порядку; повторный запуск даёт те же файлы):
  .venv/bin/python -m src.typeE download    # data/ai4i2020.csv (ucimlrepo id=601, запасной вариант — zip с UCI)
  .venv/bin/python -m src.typeE split       # data/e_split.json: тест, few-shot 5 ⊂ 20, наборы для разброса
  .venv/bin/python -m src.typeE classical   # results/typeE_classical.csv: Dummy, LogReg, GaussianNB
  .venv/bin/python -m src.typeE all         # всё сразу
  .venv/bin/python -m src.typeE show 0      # строка 0 текстом (ru и en)

Решения (подробнее в data/README.md):
- Утечки удаляются: TWF, HDF, PWF, OSF, RNF — это виды отказа, из них и складывается Machine failure.
  UDI и Product ID — идентификаторы, не свойства станка. Остаются 6 признаков (FEATURES).
- Номер строки (row_idx) — позиция строки в data/ai4i2020.csv, считая с 0 (UDI = row_idx + 1).
- Тест не пересекается с пулом, из которого берутся примеры few-shot.
- Few-shot выбирается стратифицированно и вложенно: 20 = 5 отказов + 15 без отказа, 5 = 2 + 3 из тех же 20.
  При случайном выборе из пула (3,4% отказов) 20 примеров часто оказались бы одного класса
  (вопрос самоконтроля № 8), и сравнение с LogReg/NB на тех же примерах потеряло бы смысл.
- Тест по умолчанию обогащён (40 отказов из 200), иначе в нём было бы ~7 отказов и F1 редкого класса
  считался бы по семи строкам. Это меняет долю классов, поэтому accuracy на тесте нельзя сравнивать
  с естественной: отдельно считается константа при естественной доле отказов (eval_set=full).
"""

import argparse
import hashlib
import io
import json
import sys
import zipfile
from datetime import date

import httpx
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import llm, prompts

ROOT = llm.ROOT
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
SPLIT_PATH = ROOT / "data" / "e_split.json"
EXPERIMENT_PATH = ROOT / "config" / "experiment.json"
CLASSICAL_PATH = ROOT / "results" / "typeE_classical.csv"

UCI_ID = 601
UCI_ZIP = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"

TARGET = "Machine failure"
NUMERIC = ["Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
FEATURES = ["Type"] + NUMERIC
LEAK_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]     # виды отказа = разложение целевой переменной
ID_COLS = ["UDI", "Product ID"]

# Значения по умолчанию для config/experiment.json → typeE (если поля там нет).
DEFAULTS = {"test_mode": "enriched", "n_test": 200, "n_test_failures": 40,
            "shots": [0, 5, 20], "shot_failures": {"5": 2, "20": 5}, "n_resamples": 20, "n_dev_rows": 6}


# ---------- Загрузка ----------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """В zip с UCI колонки с единицами («Air temperature [K]») и UDI, в ucimlrepo — без единиц и UID.
    Приводим к одному виду: без единиц, идентификатор строки — UDI."""
    df = df.rename(columns=lambda c: c.split(" [")[0].strip())
    return df.rename(columns={"UID": "UDI"})


def download() -> str:
    """Скачивает AI4I 2020 в data/ai4i2020.csv (все 14 колонок как есть). Возвращает источник."""
    try:
        from ucimlrepo import fetch_ucirepo
        df = fetch_ucirepo(id=UCI_ID).data.original
        source = f"ucimlrepo fetch_ucirepo(id={UCI_ID})"
    except Exception as e:      # сайт UCI иногда отдаёт ошибку API — тогда берём архив напрямую
        print(f"ucimlrepo не сработал ({type(e).__name__}: {e}), качаю zip: {UCI_ZIP}")
        resp = httpx.get(UCI_ZIP, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            df = pd.read_csv(z.open(name))
        source = UCI_ZIP
    df = normalize_columns(df)
    assert len(df) == 10000 and TARGET in df.columns, f"неожиданный файл: {df.shape}, {list(df.columns)}"
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Сохранено {DATA_PATH.relative_to(ROOT)}: {df.shape[0]} строк, {df.shape[1]} колонок; "
          f"источник: {source}; дата: {date.today().isoformat()}")
    return source


def load_data() -> pd.DataFrame:
    """Полный датасет (все колонки). Признаки для моделей брать через features()."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"нет {DATA_PATH.relative_to(ROOT)}: .venv/bin/python -m src.typeE download")
    return pd.read_csv(DATA_PATH)


def features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """X — 6 признаков без утечек и идентификаторов, y — Machine failure (1 = отказ)."""
    return df[FEATURES], df[TARGET].to_numpy()


# ---------- Настройки ----------

def load_config() -> dict:
    """config/experiment.json → typeE + seed, недостающие поля — из DEFAULTS."""
    with open(EXPERIMENT_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    return {**DEFAULTS, **cfg.get("typeE", {}), "seed": cfg["seed"]}


# ---------- Разбиение ----------

def stratified_pick(rng, idx_fail, idx_ok, n_fail, n_ok) -> list[int]:
    """n_fail строк с отказом и n_ok без отказа, без повторов."""
    return ([int(i) for i in rng.choice(idx_fail, n_fail, replace=False)]
            + [int(i) for i in rng.choice(idx_ok, n_ok, replace=False)])


def nested_shots(rng, pool_fail, pool_ok, cfg) -> dict:
    """Наборы few-shot: большой (20) стратифицированно из пула, малый (5) — его часть.
    Порядок примеров в промпте случайный (фиксирован seed), у малого — тот же относительный порядок."""
    sizes = sorted(int(k) for k in cfg["shot_failures"])            # [5, 20]
    big = sizes[-1]
    big_fail = cfg["shot_failures"][str(big)]
    fails = [int(i) for i in rng.choice(pool_fail, big_fail, replace=False)]
    oks = [int(i) for i in rng.choice(pool_ok, big - big_fail, replace=False)]
    order = [int(i) for i in rng.permutation(fails + oks)]
    sets = {str(big): order}
    for k in sizes[:-1]:
        k_fail = cfg["shot_failures"][str(k)]
        chosen = set(fails[:k_fail] + oks[:k - k_fail])               # первые из уже выбранных → вложенность
        sets[str(k)] = [i for i in order if i in chosen]
    return sets


def make_split(df: pd.DataFrame, cfg: dict) -> dict:
    _, y = features(df)
    rng = np.random.default_rng(cfg["seed"])
    all_fail = np.flatnonzero(y == 1)
    all_ok = np.flatnonzero(y == 0)
    n_test = cfg["n_test"]
    if cfg["test_mode"] == "enriched":
        n_test_fail = cfg["n_test_failures"]
    elif cfg["test_mode"] == "natural":
        n_test_fail = round(n_test * len(all_fail) / len(y))          # естественная доля, ~7 из 200
    else:
        raise ValueError(f"test_mode должен быть enriched или natural, сейчас {cfg['test_mode']!r}")
    test = stratified_pick(rng, all_fail, all_ok, n_test_fail, n_test - n_test_fail)
    test = [int(i) for i in rng.permutation(test)]     # перемешан: --limit 20 берёт начало с обоими классами

    in_test = np.zeros(len(y), bool)
    in_test[test] = True
    pool_fail = np.flatnonzero((y == 1) & ~in_test)
    pool_ok = np.flatnonzero((y == 0) & ~in_test)
    shots = nested_shots(rng, pool_fail, pool_ok, cfg)

    # Строки для отладки промпта типа E (шаг 07, experiment=dev): из пула, не из теста и не из примеров.
    used = set(shots[max(shots, key=int)])
    dev_fail = [i for i in pool_fail if i not in used]
    dev_ok = [i for i in pool_ok if i not in used]
    n_dev = cfg["n_dev_rows"]
    dev = [int(i) for i in rng.permutation(stratified_pick(rng, dev_fail, dev_ok, n_dev // 2, n_dev - n_dev // 2))]

    # Ещё n_resamples случайных стратифицированных наборов (тот же размер и состав) — только для LogReg/NB,
    # чтобы показать, насколько их качество зависит от того, какие примеры попали в обучение.
    rng_r = np.random.default_rng(cfg["seed"] + 1)
    resamples = [nested_shots(rng_r, pool_fail, pool_ok, cfg) for _ in range(cfg["n_resamples"])]

    return {
        "_comment": "Тип E, AI4I 2020. Индексы — позиции строк в data/ai4i2020.csv с 0 (UDI = индекс + 1). "
                    "Создаётся .venv/bin/python -m src.typeE split; подробности в data/README.md.",
        "dataset": "AI4I 2020 Predictive Maintenance (UCI id 601)",
        "data_file": "data/ai4i2020.csv",
        "seed": cfg["seed"], "test_mode": cfg["test_mode"],
        "n_rows": int(len(y)), "n_failures_total": int(len(all_fail)),
        "natural_failure_rate": round(len(all_fail) / len(y), 6),
        "n_test": len(test), "n_test_failures": int(y[test].sum()),
        "pool_size": int(len(pool_fail) + len(pool_ok)), "pool_failures": int(len(pool_fail)),
        "shot_failures": cfg["shot_failures"],
        "test": test,
        "shots": shots,
        "dev_rows": dev,
        "resamples": resamples,
    }


def write_split(split: dict) -> None:
    old = json.loads(SPLIT_PATH.read_text(encoding="utf-8")) if SPLIT_PATH.exists() else None
    with open(SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=1)
        f.write("\n")
    if old is not None and old != split:
        print("ВНИМАНИЕ: разбиение изменилось по сравнению с прежним data/e_split.json. "
              "Если LLM-прогоны типа E уже были, их строки относятся к старому разбиению.")
    print(f"Сохранено {SPLIT_PATH.relative_to(ROOT)}: тест {split['n_test']} строк "
          f"({split['n_test_failures']} отказов, режим {split['test_mode']}), пул {split['pool_size']}, "
          f"few-shot {', '.join(f'{k}: {len(v)}' for k, v in split['shots'].items())}, "
          f"наборов для разброса {len(split['resamples'])}, dev-строк {len(split['dev_rows'])}")


def load_split() -> dict:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"нет {SPLIT_PATH.relative_to(ROOT)}: .venv/bin/python -m src.typeE split")
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


def split_hash(split: dict) -> str:
    """Короткий отпечаток разбиения. Пишется в каждую строку LLM-результатов: если разбиение поменяли,
    старые строки не смешаются с новыми."""
    blob = json.dumps(split, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def current_split_hash() -> str | None:
    """Отпечаток текущего data/e_split.json; None, если разбиения ещё нет."""
    try:
        return split_hash(load_split())
    except FileNotFoundError:
        return None


def check_split(split: dict, y: np.ndarray) -> None:
    """Инварианты разбиения: тест и примеры не пересекаются, 5 ⊂ 20, состав классов как задумано."""
    test = set(split["test"])
    assert len(test) == len(split["test"]) == split["n_test"]
    assert int(y[split["test"]].sum()) == split["n_test_failures"]
    for sets in [split["shots"], *split["resamples"]]:
        sizes = sorted(sets, key=int)
        for small, big in zip(sizes, sizes[1:]):
            assert set(sets[small]) < set(sets[big]), "малый набор few-shot должен входить в большой"
        for k, idx in sets.items():
            assert not test & set(idx), "примеры few-shot пересекаются с тестом"
            assert len(set(idx)) == int(k) and int(y[idx].sum()) == split["shot_failures"][k]
    assert not test & set(split["dev_rows"]) and not set(split["dev_rows"]) & set(split["shots"][max(split["shots"], key=int)])


# ---------- Строка → текст ----------

def fmt_decimal(x: float, lang: str) -> str:
    """300.1 → «300,1» (ru) / «300.1» (en): одна цифра после запятой, как в датасете."""
    s = f"{x:.1f}"
    return s.replace(".", ",") if lang == "ru" else s


def fmt_int(n, lang: str) -> str:
    """1551 → «1 551» (ru, пробел между разрядами) / «1,551» (en)."""
    s = f"{int(n):,}"
    return s.replace(",", " ") if lang == "ru" else s


def row_text(row, lang: str = "ru") -> str:
    """Строка датасета → описание станка по шаблону prompts.CLASSIFY[lang]["row"] (образец из PDF)."""
    return prompts.CLASSIFY[lang]["row"].format(
        type=row["Type"],
        air=fmt_decimal(row["Air temperature"], lang),
        process=fmt_decimal(row["Process temperature"], lang),
        speed=fmt_int(row["Rotational speed"], lang),
        torque=fmt_decimal(row["Torque"], lang),
        wear=fmt_int(row["Tool wear"], lang),
    )


def shots_list(df: pd.DataFrame, idx: list[int], lang: str) -> list[dict]:
    """Примеры для промпта: [{"row_text", "label"}] в порядке из e_split.json."""
    return [{"row_text": row_text(df.iloc[i], lang), "label": int(df.iloc[i][TARGET])} for i in idx]


# ---------- Классические модели ----------

def make_models(seed: int) -> dict:
    """Константа, логистическая регрессия и наивный Байес. Type кодируется one-hot, числа стандартизуются.
    handle_unknown="ignore": в 5 примерах может не оказаться какого-то типа изделия."""
    def prep():
        return ColumnTransformer([("type", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
                                  ("num", StandardScaler(), NUMERIC)])
    return {
        "Dummy(most_frequent)": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": Pipeline([("prep", prep()),
                                        ("clf", LogisticRegression(max_iter=1000, random_state=seed))]),
        # GaussianNB требует плотную матрицу: OneHotEncoder по умолчанию выдаёт разреженную.
        "GaussianNB": Pipeline([("prep", prep().set_params(sparse_threshold=0)), ("clf", GaussianNB())]),
    }


def metrics(y_true, y_pred) -> dict:
    """F1 и precision/recall по классу «отказ» (1), balanced accuracy, accuracy, матрица ошибок."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n_eval": len(y_true), "n_eval_failures": int(np.sum(y_true)),
        "f1_failure": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision_failure": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_failure": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def evaluate(X, y, train_idx, test_idx, model_names, seed, info) -> list[dict]:
    """Обучает модели на train_idx и оценивает на тесте (eval_set=test) и на всех строках вне обучения
    с естественной долей отказов (eval_set=natural)."""
    rows = []
    rest = np.setdiff1d(np.arange(len(y)), train_idx)
    for name, model in make_models(seed).items():
        if name not in model_names:
            continue
        model.fit(X.iloc[train_idx], y[train_idx])
        for eval_set, idx in [("test", test_idx), ("natural", rest)]:
            rows.append({"model": name, **info, "eval_set": eval_set,
                         "n_train": len(train_idx), "n_train_failures": int(y[train_idx].sum()),
                         **metrics(y[idx], model.predict(X.iloc[idx]))})
    return rows


def classical(df: pd.DataFrame, split: dict, seed: int) -> pd.DataFrame:
    X, y = features(df)
    check_split(split, y)
    test = np.array(split["test"])
    rows = []
    all_models = list(make_models(seed))
    for k, idx in split["shots"].items():                    # те же примеры, что видит LLM
        rows += evaluate(X, y, np.array(idx), test, all_models, seed,
                         {"shots": int(k), "train_set": "fixed", "resample_id": None})
    for r, sets in enumerate(split["resamples"]):             # разброс по случайным наборам
        for k, idx in sets.items():
            rows += evaluate(X, y, np.array(idx), test, ["LogisticRegression", "GaussianNB"], seed,
                             {"shots": int(k), "train_set": "resample", "resample_id": r})
    # Константа «всегда нет отказа» на всём датасете — accuracy при естественной доле отказов.
    rows.append({"model": "constant(нет отказа)", "shots": None, "train_set": "none", "resample_id": None,
                 "eval_set": "full", "n_train": 0, "n_train_failures": 0, **metrics(y, np.zeros_like(y))})
    table = pd.DataFrame(rows)
    table[["shots", "resample_id"]] = table[["shots", "resample_id"]].astype("Int64")   # целые с пропусками
    table["seed"] = seed
    table["test_mode"] = split["test_mode"]
    return table


def write_classical(table: pd.DataFrame) -> None:
    CLASSICAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CLASSICAL_PATH, index=False, float_format="%.6f")
    print(f"Сохранено {CLASSICAL_PATH.relative_to(ROOT)}: {len(table)} строк")
    cols = ["model", "shots", "eval_set", "f1_failure", "balanced_accuracy", "accuracy"]
    fixed = table[table["train_set"].isin(["fixed", "none"])]
    print(fixed[cols].to_string(index=False, float_format="%.3f"))
    res = table[(table["train_set"] == "resample") & (table["eval_set"] == "test")]
    print(f"\nРазброс по {res['resample_id'].nunique()} случайным стратифицированным наборам (тест):")
    print(res.groupby(["model", "shots"])[["f1_failure", "balanced_accuracy"]]
          .agg(["mean", "min", "max"]).to_string(float_format="%.3f"))


# ---------- CLI ----------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Тип E: AI4I 2020, разбиение, классические модели")
    ap.add_argument("command", choices=["download", "split", "classical", "all", "show"])
    ap.add_argument("row", nargs="?", type=int, help="для show: номер строки (с 0)")
    args = ap.parse_args(argv)

    if args.command in ("download", "all"):
        download()
    if args.command == "show":
        row = load_data().iloc[args.row]
        print(row.to_string() + "\n\nru: " + row_text(row, "ru") + "\nen: " + row_text(row, "en"))
        return 0
    cfg = load_config()
    if args.command in ("split", "all"):
        df = load_data()
        split = make_split(df, cfg)
        check_split(split, features(df)[1])
        write_split(split)
    if args.command in ("classical", "all"):
        write_classical(classical(load_data(), load_split(), cfg["seed"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
