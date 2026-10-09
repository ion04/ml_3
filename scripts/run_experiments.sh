#!/usr/bin/env bash
# Весь эксперимент одной командой. Останавливается на первой ошибке.
#
#   scripts/run_experiments.sh          # реальные прогоны (нужны ключи и пройденный гейт G1)
#   scripts/run_experiments.sh --demo   # то же на mock-моделях, всё пишется в demo/
#
# Повторный запуск безопасен: ответы берутся из журнала (кэш), новых вызовов нет, кроме smoke-теста
# внутри g1 (он проверяет связь сейчас) и дозапроса вызовов с технической ошибкой.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

DEMO=""
if [[ "${1:-}" == "--demo" ]]; then
  DEMO="--demo"
  echo "=== DEMO: mock-модели, результаты в demo/ (это не результаты эксперимента) ==="
fi

if [[ -z "$DEMO" ]]; then
  echo "=== 1–2. Гейт G1: ключи, smoke-тест всех моделей, банк, гипотезы закоммичены до прогонов ==="
  $PY -m src.checks g1      # smoke-тест запускается внутри g1 (src.smoke_test), отдельно не нужен
fi

echo "=== 3. Основной прогон банка (RU; английский — режим lang_en для H4) ==="
$PY -m src.run_bank --experiment main --models all --lang ru $DEMO

echo "=== 4. Дополнительные прогоны под выбранные гипотезы ==="
if [[ -n "$DEMO" ]]; then
  $PY -m src.run_extra all $DEMO        # в демо проверяем все режимы
else
  $PY -m src.run_extra selected
fi

echo "=== 5. Запуск кода из ответов типа D ==="
$PY -m src.exec_code --experiment main $DEMO

echo "=== 6. Тип E: LLM-классификатор AI4I ==="
# (шаг 06 добавит команду src.run_classifier: сначала пробный прогон на 20 строках, потом полный)

echo "=== 7. Подготовка проверки: автопроверка, слепые листы ==="
# (шаг 10 добавит команды)

echo "=== Готово ==="
