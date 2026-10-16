#!/usr/bin/env bash
# Выполняет analysis.ipynb целиком (как Restart & Run All).
#
#   scripts/run_analysis.sh          # реальные данные: ноутбук выполняется на месте, выходы сохраняются в нём;
#                                    # рисунки → report/figures/, числа → results/summary_metrics.json,
#                                    # затем report/error_candidates.md (после G2) и каркас report/report.md
#   scripts/run_analysis.sh --demo   # проверка на demo/: случайные оценки в demo/grading/ (src.demo_grades),
#                                    # копия demo/analysis_demo.ipynb с DEMO = True; analysis.ipynb не меняется
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

if [[ "${1:-}" == "--demo" ]]; then
  $PY -m src.demo_grades
  # Копия ноутбука с DEMO = True в первой ячейке кода.
  $PY - <<'PYEOF'
import nbformat
nb = nbformat.read("analysis.ipynb", as_version=4)
cell = next(c for c in nb.cells if c.cell_type == "code")
assert cell.source.startswith("DEMO = False"), "первая ячейка кода должна начинаться с DEMO = False"
cell.source = cell.source.replace("DEMO = False", "DEMO = True", 1)
nbformat.write(nb, "demo/analysis_demo.ipynb")
PYEOF
  # Ноутбук лежит в demo/, а пакет src — в корне проекта: PYTHONPATH указывает ядру, где его искать.
  PYTHONPATH="$PWD" .venv/bin/jupyter nbconvert --to notebook --execute --inplace demo/analysis_demo.ipynb
  $PY -m src.error_candidates --demo || echo "demo/report/error_candidates.md не обновлён (см. сообщение выше)"
  $PY -m src.build_report --demo
  echo "Готово: demo/analysis_demo.ipynb, demo/report/ (report.md, error_candidates.md, figures/), demo/results/summary_metrics.json (DEMO, не результаты)"
else
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
  # Кандидаты на разбор ошибок — только после слепой проверки (G2); код 1/2 не останавливает сборку отчёта.
  $PY -m src.error_candidates || echo "report/error_candidates.md не обновлён (см. сообщение выше)"
  $PY -m src.build_report
  echo "Готово: analysis.ipynb (с выходами), report/figures/, results/summary_metrics.json, report/report.md"
fi
