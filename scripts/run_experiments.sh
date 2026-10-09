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

# Код 3 у прогонщиков = остались вызовы с технической ошибкой (сеть, 429, 5xx). Тогда команда
# повторяется: из кэша берутся готовые ответы, дозапрашиваются только ошибочные (повтор виден в журнале).
# Содержательно плохие ответы не перезапрашиваются: они уже в кэше. Другие коды — сразу остановка.
TECH_RETRIES=3
TECH_PAUSE_S=60
with_retries() {
  local attempt=1 rc
  while true; do
    rc=0
    "$@" || rc=$?
    if [[ $rc -ne 3 ]]; then return $rc; fi
    if [[ $attempt -ge $TECH_RETRIES ]]; then
      echo "Техошибки остались после $TECH_RETRIES запусков. Запустите скрипт позже ещё раз."
      return 3
    fi
    echo "--- остались техошибки: пауза ${TECH_PAUSE_S} с и дозапрос (запуск $((attempt + 1)) из $TECH_RETRIES) ---"
    sleep "$TECH_PAUSE_S"
    attempt=$((attempt + 1))
  done
}

DEMO=""
if [[ "${1:-}" == "--demo" ]]; then
  DEMO="--demo"
  echo "=== DEMO: mock-модели, результаты в demo/ (это не результаты эксперимента) ==="
fi

if [[ -z "$DEMO" ]]; then
  echo "=== 0. Локальная модель на ПК gpu: проверка или запуск туннеля ==="
  scripts/ensure_gpu_ollama.sh   # без ответа Ollama на gpu останавливаемся: на ноутбуке не считаем
  echo "=== 1–2. Гейт G1: ключи, smoke-тест всех моделей, банк, гипотезы закоммичены до прогонов ==="
  $PY -m src.checks g1      # smoke-тест запускается внутри g1 (src.smoke_test), отдельно не нужен
fi

echo "=== 3. Основной прогон банка (RU; английский — режим lang_en для H4) ==="
with_retries $PY -m src.run_bank --experiment main --models all --lang ru $DEMO

echo "=== 4. Дополнительные прогоны под выбранные гипотезы ==="
if [[ -n "$DEMO" ]]; then
  with_retries $PY -m src.run_extra all $DEMO        # в демо проверяем все режимы
else
  with_retries $PY -m src.run_extra selected
fi

echo "=== 5. Запуск кода из ответов типа D ==="
$PY -m src.exec_code --experiment main,paraphrase $DEMO   # paraphrase — если H8 и у задач D есть перефразировки

echo "=== 6. Тип E: LLM-классификатор AI4I ==="
# Данные и разбиение лежат в data/ (ai4i2020.csv, e_split.json); классические модели вызовов не делают.
$PY -m src.typeE classical
echo "--- 6a. Пробный прогон: 20 строк теста × 0/5/20 примеров ---"
with_retries $PY -m src.run_classifier --models all --shots 0,5,20 --limit 20 $DEMO
echo "--- 6b. Полный прогон: 200 строк (первые 20 возьмутся из кэша) ---"
with_retries $PY -m src.run_classifier --models all --shots 0,5,20 $DEMO

echo "=== 7. Подготовка проверки: автопроверка, слепые листы ==="
# (шаг 10 добавит команды)

echo "=== 8. Паспорт прогонов: даты, model_reported, промпт, параметры, вызовы и ошибки ==="
$PY -m src.manifest $DEMO

echo "=== Готово ==="
