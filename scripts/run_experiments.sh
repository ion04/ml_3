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
# Формат ответов пробного прогона: код 4, если parse_ok < 90% хоть в одной группе модель × примеры.
# Тогда скрипт останавливается: промпт правится только на dev (шаг 07), не здесь.
$PY -m src.typeE_metrics check --limit 20 $DEMO
echo "--- 6b. Полный прогон: 200 строк (первые 20 возьмутся из кэша) ---"
with_retries $PY -m src.run_classifier --models all --shots 0,5,20 $DEMO
echo "--- 6c. Таблица метрик LLM и классических моделей: typeE_metrics.csv ---"
$PY -m src.typeE_metrics $DEMO

echo "=== 7. Подготовка проверки: автопроверка, слепые листы ==="
# graded_auto.jsonl (A, вердикт B, E, F, корректность метрики D) + grading/parser_spotcheck.csv
# Код 2 = проверять нечего (нет таблиц или ответов B/C/D) — это не ошибка, идём дальше.
rc=0; $PY -m src.autograde $DEMO || rc=$?
if [[ $rc -ne 0 && $rc -ne 2 ]]; then exit $rc; fi
# grading/sheet_r1.csv, sheet_r2.csv, key_DO_NOT_OPEN.csv. Уже заполняемые листы не перезаписываются;
# если набор ответов изменился, а оценки есть — код 1 и остановка (решает человек).
rc=0; $PY -m src.blind $DEMO || rc=$?
if [[ $rc -ne 0 && $rc -ne 2 ]]; then exit $rc; fi

echo "=== 8. Паспорт прогонов: даты, model_reported, промпт, параметры, вызовы и ошибки ==="
$PY -m src.manifest $DEMO

echo "=== 9. Анализ: analysis.ipynb ==="
if [[ -n "$DEMO" ]]; then
  scripts/run_analysis.sh --demo   # случайные оценки в demo/grading/, копия ноутбука с DEMO = True в demo/
else
  # Реальный анализ — после ручной проверки (G2): ноутбук выполняется на месте, выходы остаются в нём.
  echo "Когда .venv/bin/python -m src.checks g2 даст код 0: scripts/run_analysis.sh"
fi

echo "=== Готово ==="
