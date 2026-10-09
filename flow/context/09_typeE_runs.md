STATUS: deferred
## Сделано
- Реальный прогон E НЕ запускался: ключей GigaChat/Yandex/DeepSeek нет (smoke 1 из 4, ответила только qwen3-8b на gpu), шаг 07 deferred (промпт не заморожен), гипотезы не закоммичены. `run_classifier` без --demo: «не запущен», код 1, вызовов нет. Журнал 34 → 35 записей (+1 smoke от моей проверки).
- Новый `src/typeE_metrics.py`:
  - `table` (по умолчанию) → `results/typeE_metrics.csv` (`--demo` → `demo/results/`);
  - `check --limit 20` — формат пробного прогона по группам модель × примеры: parse_ok, ответы 0/1, «нельзя определить», None, техошибки, finish_reason=length. Код 4 (`EXIT_FORMAT`), если parse_ok < 90% (`--min-parse-ok`); код 2, если строк нет.
- `scripts/run_experiments.sh`, шаг 6: 6a пробный прогон → `typeE_metrics check --limit 20` → 6b полный → 6c `typeE_metrics`.
- `tests/test_typeE_metrics.py` (5 тестов), README п. 6, HUMAN_TODO G1 «Тип E» (+3 пункта шага 09).
## Ключевые файлы и точки входа
- `.venv/bin/python -m src.typeE_metrics [check --limit N] [--demo] [--min-parse-ok 0.9]`.
- `typeE_metrics.build(llm_path, classical_path, split) -> DataFrame` (колонки `COLUMNS`), `llm_table(rows, split)`, `llm_group(rows)`, `classical_table(csv_df)`, `format_table`, `check`.
- Строки CSV:
  - `source=llm`: model × shots × lang, eval_set=test, train_set none|prompt;
  - `source=classical`: fixed (LogReg/NB/Dummy на 5 и 20 примерах), resample (среднее по 20 + `*_sd`, `n_resamples`), constant eval_set=full.
- Колонки: n_eval, n_eval_failures, n_answered, coverage, n_cannot_determine, n_refused_or_unparsed, n_tech_error, complete, f1_failure, balanced_accuracy, accuracy, precision/recall_failure, tp/fp/fn/tn, *_answered, tokens_in/out mean/sum, latency_s mean/median, prompt_version, split_hash, model_reported.
## Решения и допущения
- LLM-строки фильтруются так: experiment=typeE, строки теста из e_split.json, текущий split_hash.
- Неответ (value не 0/1) в основных метриках считается ошибкой: при y=1 → fn, при y=0 → fp. Так у LLM и классических моделей n_eval = 200. Метрики только по ответам 0/1 — в `*_answered` + `coverage`.
- Техошибки исключены из метрик (`n_tech_error`). `complete` = нет техошибок и покрыты все строки теста.
- Порог формата пробного прогона — 90% parse_ok. Если формат ломается, это стоп и возврат к шагу 07; промпт здесь не правится.
- Проверка «гипотезы закоммичены» для E оставлена, хотя в задании шага сказано, что она не нужна: по CLAUDE.md гипотезы коммитятся до любых оценочных прогонов, а H12 строится на E. Решение вынесено в HUMAN_TODO G1.
- Реальный `results/typeE_metrics.csv` не создавал: без LLM-строк он выглядел бы как готовая таблица. Создаётся после прогона.
- Классические строки берутся из реального `results/typeE_classical.csv` (без API) и в demo тоже; LLM-строки в demo — mock.
## Проверка
- `.venv/bin/python -m pytest -q`: 429 passed (было 424). Тесты: неответы и техошибки, фильтры и complete, resample = среднее/sd CSV, коды check 0/4/2, сквозной demo.
- `scripts/run_experiments.sh --demo`: «=== Готово ===». Шаг 6: check «Формат в норме» (12 групп), `demo/results/typeE_metrics.csv`: LLM 12 строк, классических 11. Повтор — всё из кэша.
- `typeE_classical.csv` пересобран скриптом без изменений (git diff пуст).
## Что нужно для завершения
- G1: ключи в `.env`, smoke 4/4, гипотезы закоммичены (или решение человека снять это требование для E), шаг 07 done (`prompt_version`/`prompt_hash` в config). ПК gpu включён (`scripts/ensure_gpu_ollama.sh` → 0).
- Повтор шага 09:
  1. `scripts/ensure_gpu_ollama.sh`.
  2. `src.run_classifier --models all --shots 0,5,20 --limit 20`; при коде 3 повторить.
  3. `src.typeE_metrics check --limit 20`. Код 4 → шаг 09 blocked, вернуться к 07.
  4. `src.run_classifier --models all --shots 0,5,20` (код 3 → повтор), затем `src.typeE_metrics`, `src.manifest`.
  5. В handoff только количества (вызовы, ошибки, parse_ok, complete), без метрик и выводов. Коммит с logs/ и results/.
## Открытые вопросы и риски
- Бюджет: 2400 вызовов, GigaChat в 1 поток (~600 вызовов на модель). Бесплатных лимитов может не хватить (HUMAN_TODO G1, шаг 06).
- qwen3-8b на dev рассуждает перед ответом (~230 токенов, ~12 с на вызов): 600 вызовов ≈ 2 ч на gpu, сбои RAM переживаются кэшем.
- Если в группе один класс (маленький --limit), sklearn предупреждает в balanced_accuracy. На 200 строках теста этого нет.
- Правило «неответ = ошибка» спорное: и основные, и `*_answered` числа есть, выбор для отчёта за человеком.
## Для следующих шагов
- Шаг 11: в ноутбуке вызывать `typeE_metrics.build(results/typeE_llm.jsonl)` (числа считаются по таблице из журнала), а не вписывать CSV руками. H12: пары LLM k-shot ↔ LogReg fixed k-shot на тех же 200 строках (Макнемар по `typeE_llm.jsonl` row_idx + предсказаниям LogReg, их пересчитать через `typeE.make_models`).
- Шаг 12: проверка воспроизводимости — `src.typeE_metrics` должен давать тот же CSV.
