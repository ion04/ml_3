STATUS: done
## Сделано
- `src/run_bank.py`: задачи × модели → `ask` (meta task_id/type/variant/experiment/lang/prompt_version) → parse → grade → `results/<experiment>_<lang>.jsonl`. `--demo` (mock той же роли, всё в `demo/`), `--types`, `--limit`, `--bank`, `--experiment main|dev`, tqdm, сводка по моделям (всего/новых/из кэша/ошибок/parse_ok, без точности).
- `src/run_extra.py`: режимы cot (H6), selfconsistency (H7, + голосование), followup_doubt (H9), paraphrase (H8), lang_en (H4), verbal_formula (H2). Без аргументов — режимы выбранных гипотез; печатает, что H1/H3/H5/H10–H13 считаются из main и типа E. `all` — все режимы; `--dev` — dev-банк, experiment=dev, таблицы results/dev_<режим>_<lang>.jsonl.
- `src/exec_code.py`: код D без правок → `results/d_code/<model>/<task>.py`, запуск `.venv/bin/python` во временной папке, таймаут 300 с (killpg), env из белого списка + MPLBACKEND=Agg → `results/d_exec.jsonl` (rc, время, stdout/stderr, metric, metric_ok, pass_at_1).
- `scripts/run_experiments.sh [--demo]`: smoke → g1 → main → run_extra → exec_code → (E: шаг 06) → (проверка: шаг 10); `set -euo pipefail`.
- `src/llm.py`: `_log_lock` (RLock) вокруг записи журнала и индекса кэша (потоки); кэшируется любая запись с error=None (пустой текст тоже — это ответ модели); mock отвечает кодом на ```python, вердиктом на «верно или неверно / true or false», иначе числом (каждый 4-й код падает).
- `config/experiment.json` → `extra_runs` (типы по режимам, `cot.extra_models`, `selfconsistency.vote_rule`), новое поле `prompt_hash: null`.
- `src/checks.py`: проверка гипотез вынесена в `check_hypotheses_committed(cfg)` (G1 и прогонщики).
- `tests/test_runners.py` (27 тестов) + `tests/fixtures/bank_small.jsonl`; README п. 5; HUMAN_TODO (G1 «Прогоны», «Можно делать в любой момент»).
## Ключевые файлы и точки входа
- `run_bank.make_job(task, model, experiment, lang, variant, T, repeat, history, extra)`, `run_jobs(jobs, demo, desc) -> rows` (порядок = jobs), `write_results(path, rows)` (слияние по ключу experiment/lang/model/task/variant/repeat), `results_path(exp, lang, demo)`, `out_dir(demo)`, `read_jsonl`, `select_models(spec, demo, allow_disabled)`, `select_tasks`, `eval_problems(experiment, bank_path)`, `show(path)`.
- Строка результатов: experiment, lang, task_id, type, topic, source, difficulty, answer_kind, pair_id, task_variant (из банка), variant (промпт), model_key, model_reported, repeat, temperature (эффективная), prompt_version, call_id, error, finish_reason, tokens_in/out, latency_s, response_text, value, raw_answer_line, confidence, refused, parse_ok, parse_notes, correct, score, reason. followup: + first_call_id, first_value, first_correct.
- Таблицы: main_ru, main_en (lang_en), cot_<lang>, paraphrase_<lang>, verbal_formula_<lang>, selfconsistency_<lang> + selfconsistency_vote_<lang>, followup_doubt_<lang>, d_exec.jsonl; dev → results/dev_ru.jsonl.
## Решения и допущения
- Оценочный прогон (не demo/dev) не стартует без: `validate --final`, проверок гипотез как в G1 (`check_hypotheses_config` + `check_hypotheses_committed`), `prompt_version` и `prompt_hash` в config = `PROMPT_VERSION` и `prompts.prompt_hash()`. `experiment=dev` только на tasks_dev.jsonl и наоборот.
- Заготовки `todo` пропускаются с сообщением. Модели: выключенные, без ключей, Ollama не отвечает — пропуск с сообщением; mock только с --demo; `cot.extra_models` идут даже при enabled=false (вариант base, эксперимент cot).
- Потоки на сервис (provider+base_url): gigachat 1, openai_compat 3, ollama 1, mock 4; переопределение `max_workers` в models.json.
- `run_bank --lang` по умолчанию `ru`; EN-прогон основной — только `run_extra lang_en` (типы из `extra_runs.lang_en.types`). Остальные режимы — по `languages` из config.
- Ответ с техошибкой: строка с error, correct=None; повторный запуск дозапрашивает только такие (кэш хранит лишь успехи). Код выхода 1, если были ошибки.
- H7: голосуют повторы repeat < sc_repeats; группы по порядку repeat; числа совпадают в пределах tolerance задачи (grade.close), отказ = «нельзя определить» = один ответ, неразобранные не голосуют, ничья → раньше появившийся; correct большинства = correct его первого ответа; ни один не разобран → False, все техошибки → None. Таблица голосов обновляется по (модель, задача). Подтверждение — HUMAN_TODO G1.
- exec_code: stdin=DEVNULL, killpg группы и после нормального выхода; каждый ответ запускается один раз (журнал запусков дописывается, ключ включает call_id). Метрика: последняя строка stdout с числом, `parse.pick_number` (после последнего = / :). Сеть коду доступна (песочницы нет) — пункт HUMAN_TODO.
- Имя кода для не main/ru: `<task>__<experiment>_<lang>.py`. exec_code по умолчанию берёт все языки, для которых есть таблица.
## Проверка
- `.venv/bin/python -m pytest -q`: 345 passed (311 старых + 34 новых). Тесты подменяют demo/, logs/, results/ на tmp и проверяют, что реальные logs/ и results/ не изменились.
- `scripts/run_experiments.sh --demo`: «=== Готово ===» (main 72 вызова на 18 не-todo задачах, все режимы, 12+12 запусков D). Повторный запуск: «новых: 0».
- `scripts/run_experiments.sh` без --demo (версия до ревью, с отдельным smoke): остановился на smoke-тесте (ключей нет, код 1), дописал 1 запись experiment=smoke в logs/calls.jsonl (qwen3-8b ok). Теперь smoke запускается только внутри `checks g1`.
- `src.run_bank` без --demo: «Прогон не запущен» со списком (гипотезы, промпт), вызовов нет.
- Ревью субагентом: найдено 11 пунктов, исправлены (проверка гипотез как в G1, prompt_hash, голоса H7, дубли моделей, двойной smoke, stdin/killpg, кэш пустых ответов, docstring «без правок»).
## Что нужно для завершения
- Ничего: шаг done на mock. Реальные прогоны — шаг 08 после G1.
## Открытые вопросы и риски
- Параллельность openai_compat=3 не проверена на реальных лимитах Yandex/DeepSeek; при 429 уменьшить `max_workers` в models.json (ретраи есть).
- max_tokens не задаётся (умолчания провайдеров); длинный CoT/код может обрезаться — смотреть finish_reason=length на шаге 07/08.
- vote() сравнивает значения как есть: bool True и число 1.0 равны в Python (на одной задаче не встречается).
- write_results не удаляет устаревшие строки (например, после уменьшения sc_repeats или смены prompt_version) — при смене параметров удалить производную таблицу и пересобрать (журнал не трогать).
- Метрика D из stdout — подсказка (для classification_report/матрицы ошибок возьмётся не то число); решают люди по чек-листу.
## Для следующих шагов
- Шаг 07: `.venv/bin/python -m src.run_bank --experiment dev [--models ...]` → results/dev_ru.jsonl; отладка cot/followup: `src.run_extra cot followup_doubt --dev`. При заморозке записать в config `prompt_version` И `prompt_hash` (= `prompts.prompt_hash()`) — без этого main не стартует.
- Шаг 06: добавить команду E в `scripts/run_experiments.sh` (пункт 6), можно переиспользовать `run_bank.select_models`, `run_jobs` не подходит (у E нет задачи банка) — свой цикл через ask или make_job-подобный.
- Шаг 08: `scripts/run_experiments.sh` или по шагам; манифест — по results/*.jsonl (model_reported, call_id → журнал).
- Шаг 10/11: B-обоснование, C, D — `correct=None`/вердикт; pass@1 и metric_ok из d_exec.jsonl; голосование H7 — selfconsistency_vote_<lang>.jsonl; H9 flip — followup_doubt_<lang>.jsonl (correct второго хода при first_correct=True).
