"""Случайные оценки в демо-листах — только чтобы проверить analysis.ipynb без настоящей ручной проверки.

Запуск:
  .venv/bin/python -m src.demo_grades

Пишет ТОЛЬКО в demo/grading/ (sheet_r1.csv, sheet_r2.csv, parser_spotcheck.csv), которые создал
`src.blind --demo` и `src.autograde --demo`. Оценки случайные (seed из config/experiment.json): это не проверка
ответов и не результаты. Проверяющий 2 совпадает с проверяющим 1, кроме ~15% клеток, — чтобы каппа и список
расхождений в ноутбуке были не пустыми. Настоящие листы grading/ заполняют два человека вслепую.
"""

import random
import sys

from src import autograde, blind, run_bank

DEMO_GRADING = run_bank.DEMO_DIR / "grading"
DISAGREE = 0.15     # доля клеток, где «проверяющий 2» ставит другое значение
VALUES = {blind.SCORE_COL: ["0", "1", "2", "2"], blind.CORRECT_COL: ["0", "1", "1"],
          **{c: ["0", "0", "0", "1", "н/п"] for c in blind.CHECKLIST}}
NOTE = "DEMO: случайная оценка, не проверка"


def fill(grading_dir=DEMO_GRADING, seed: int = 42) -> int:
    """Заполняет обязательные клетки демо-листов и столбец human_agrees выборки парсера. Код 0 / 2 (нет листов)."""
    if not grading_dir.resolve().is_relative_to(run_bank.DEMO_DIR.resolve()):
        raise ValueError("demo_grades пишет только в demo/: настоящие листы заполняют люди")
    key_path = grading_dir / blind.KEY_NAME
    if not key_path.exists():
        print("Нет demo/grading/key_DO_NOT_OPEN.csv: сначала scripts/run_experiments.sh --demo")
        return 2
    rng = random.Random(seed)
    key = {k["item_id"]: k for k in autograde.read_csv(key_path)}
    r1 = autograde.read_csv(grading_dir / blind.SHEETS[0])
    r2 = []
    for row in r1:
        other = dict(row)
        for col in filter(None, key[row["item_id"]]["required"].split(";")):
            row[col] = rng.choice(VALUES[col])
            other[col] = row[col] if rng.random() > DISAGREE else rng.choice(VALUES[col])
        row["comment"] = other["comment"] = NOTE
        r2.append(other)
    autograde.write_csv(grading_dir / blind.SHEETS[0], r1, blind.SHEET_COLUMNS)
    autograde.write_csv(grading_dir / blind.SHEETS[1], r2, blind.SHEET_COLUMNS)

    spot_path = grading_dir / autograde.SPOTCHECK_NAME
    if spot_path.exists():
        spot = autograde.read_csv(spot_path)
        for row in spot:
            row["human_agrees"] = "да" if rng.random() < 0.9 else "нет"
            row["comment"] = NOTE
        autograde.write_csv(spot_path, spot, autograde.SPOTCHECK_COLUMNS)
    print(f"Демо-листы заполнены случайными оценками: {len(r1)} строк ({run_bank.show(grading_dir)})")
    return 0


if __name__ == "__main__":
    sys.exit(fill(seed=run_bank.load_experiment().get("seed", 42)))
