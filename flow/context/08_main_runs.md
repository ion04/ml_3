STATUS: deferred
## Сделано
- Оценочные прогоны НЕ запускались: `src.checks g1` = 1, не выполнено 5 из 6 условий: нет ключей GigaChat/Yandex/DeepSeek, smoke пропущен, банк не проходит `validate --final` (38 не verified, 20 заготовок TODO), гипотезы не выбраны и не закоммичены. Промпт тоже не заморожен (шаг 07 deferred). Журнал: 34 записи (smoke 6, dev 28), новых нет.
- Новый `src/manifest.py` пересобирает `results/run_manifest.json` (в demo: `demo/results/`) по журналу и таблицам. Содержимое:
  - `calls_by_experiment`: даты, записи/ok/ошибки, `retried_requests`, `unresolved_requests`;
  - `models` (без smoke/dev): `model_requested`, `model_reported`, params, температуры, max_tokens, токены, `finish_reason_length`, для Ollama VRAM и base_url;
  - `prompt`: версия и хэш из config и из кода; `params`; `hypotheses`: id, commit, время; `typeE_split_hash`;
  - `results`: по каждой таблице строки/ошибки/parse_ok/модели/версии промпта; `d_exec`; `environment`: Python, пакеты, sha256 requirements; `git_head`.
  Раздел `ollama_server` сохраняется. Реальный манифест собран сейчас (только smoke/dev, `models` пуст).
- Код выхода 3 (`run_bank.EXIT_TECH_ERRORS`) = остались техошибки. Возвращают его run_bank, run_extra и run_classifier (раньше было 1, как и для «не запущен»).
- `scripts/run_experiments.sh`:
  - `with_retries`: на коде 3 пауза 60 с и повтор, до 3 запусков; другие коды останавливают скрипт сразу;
  - exec_code вызывается с `--experiment main,paraphrase`;
  - новый шаг 8 `src.manifest`.
- `src/exec_code.py`: `--experiment` принимает список через запятую. Пары (эксперимент, язык) берутся по наличию таблиц.
- README п. 5 (коды выхода, manifest), HUMAN_TODO G1 «Прогоны» (шаг 08), `tests/test_manifest.py` (4 теста).
- `pip freeze` совпадает с `requirements.txt`, файл не менялся.
## Ключевые файлы и точки входа
- `.venv/bin/python -m src.manifest [--demo]`; `manifest.build(log_path, results_dir, demo)`, `call_stats(records)`, `by_model`, `table_stats`, `write(manifest, path)`.
- `scripts/run_experiments.sh`: 0 gpu-туннель → 1–2 g1 → 3 main → 4 run_extra selected → 5 exec_code → 6 тип E → 7 (шаг 10) → 8 manifest.
## Решения и допущения
- Основной прогон: `run_bank --lang ru`. EN (H4) делает `run_extra lang_en` (таблица main_en, experiment=main). Сейчас `languages` = ["ru"].
- Раздел `models` в манифесте считает только оценочные вызовы (без smoke/dev). Счётчики по экспериментам включают всё.
- Манифест производный: каждый раз пересобирается целиком, кроме `ollama_server`.
- `git_head` — HEAD на момент сборки, то есть коммит до коммита с результатами.
- Дозапрос: только вызовы без успешной записи (кэш). Содержательно плохие ответы уже в кэше и не перезапрашиваются.
## Проверка
- `.venv/bin/python -m pytest -q`: 424 passed (было 420). Новые тесты: манифест на demo, `call_stats`, сохранение ollama_server, код 3 и дозапрос только ошибок (журнал дописан, ошибки видны, третий запуск — всё из кэша).
- `scripts/run_experiments.sh --demo`: «=== Готово ===», шаг 8 печатает сводку. Демо: main 144 записи (main_ru 72 + main_en 72), parse_ok 72/72; ошибок 0.
- `scripts/run_experiments.sh` без demo: туннель gpu ок, затем остановка на g1 (5 из 6 ✗). run_bank/run_extra cot/run_classifier: «не запущен»; журнал 34 → 34.
## Что нужно для завершения
- G1 полностью: ключи в `.env` + smoke 4/4; банк `validate --final` = 0 и закоммичен; 3 гипотезы в hypotheses.md и config, коммит; затем шаг 07 (заморозка `prompt_version`/`prompt_hash`).
- ПК gpu включён (`scripts/ensure_gpu_ollama.sh` → код 0).
- Повтор шага 08:
  1. `.venv/bin/python -m src.checks g1` → 0;
  2. `with_retries`-эквивалент: `src.run_bank --experiment main --models all --lang ru`, `src.run_extra selected`, `src.exec_code --experiment main,paraphrase` (при коде 3 повторить);
  3. `src.server_info --write --note "<прежняя note>"` при необходимости, `src.manifest`, `pip freeze > requirements.txt`;
  4. в handoff записать только количества (вызовы, ошибки, parse_ok), затем коммит `step 08: ...` с logs/ и results/.
- Тип E — шаг 09, не здесь (скрипт делает его шагом 6; на шаге 08 можно вызывать команды по отдельности).
## Открытые вопросы и риски
- `with_retries` в bash не покрыт тестом (покрыт только код 3 в Python). Проверен `bash -n`; в demo ошибок нет, поэтому повтор не срабатывал.
- Если у D-задач появятся `paraphrase_ru`, их код пойдёт в d_exec с experiment=paraphrase, а имена файлов будут `D1__paraphrase_ru.py`. Ноутбук должен фильтровать по experiment.
- Лимиты openai_compat (3 потока) на реальных API не проверены. При частых 429 уменьшить `max_workers` в models.json.
- max_tokens не задан; `finish_reason_length` в манифесте покажет обрезанные ответы.
## Для следующих шагов
- Шаг 09: после прогона E запустить `src.manifest` снова: появятся `typeE` в `calls_by_experiment` и `typeE_llm.jsonl` в `results`.
- Шаг 11/12: таблица моделей для отчёта (model_reported, даты, параметры) и хэш коммита гипотез — из `results/run_manifest.json`. Железо локальной модели — из `ollama_server`.
- Шаг 10: d_exec.jsonl содержит experiment; листы D строить по main (и paraphrase, если H8).
