STATUS: done
## Сделано
- `src/typeE.py`: `download` (ucimlrepo id=601, запасной вариант zip UCI) → `data/ai4i2020.csv` (все 14 колонок, имена без единиц, `UID`→`UDI`), `split` → `data/e_split.json`, `classical` → `results/typeE_classical.csv` (реальные данные, API не нужен), `show N`, `all`.
- `src/run_classifier.py`: `--models all --shots 0,5,20 [--limit N] [--lang ru] [--demo] [--dev] [--estimate]` → `results/typeE_llm.jsonl` (dev: `dev_typeE_llm.jsonl`). Внутри переиспользуются `run_bank.make_job/run_jobs/write_results`.
- Промпт E: `prompts.CLASSIFY` (intro, нумерованные примеры «Исход: отказ|нет отказа», шаблон строки `row`, `labels`), `FORMAT[*]["label"]` = «отказ или нет отказа». `parse_label` переписан. mock отвечает классом на промпт E.
- `run_bank`: `make_job(meta=...)` (доп. поля meta), `write_results(key=...)`, `prompt_problems(cfg)` вынесена из `eval_problems`.
- `config/experiment.json → typeE`: test_mode, n_test, n_test_failures, shots, shot_failures {"5":2,"20":5}, n_resamples 20, n_dev_rows 6.
- `data/README.md` (источник, дата 2026-10-10, утечки, разбиение), README п. 6, `scripts/run_experiments.sh` п. 6 (classical → пробный --limit 20 → полный), HUMAN_TODO (G1 тип E ×4, G2, G3).
- `tests/test_typeE.py` (+ случаи меток в `test_parse.py`).
## Ключевые файлы и точки входа
- `typeE.load_data()`, `features(df)` (6 признаков без утечек), `load_config()`, `make_split`, `check_split`, `split_hash`, `row_text(row, lang)`, `shots_list`, `classical(df, split, seed)`, `metrics(y, pred)`.
- `e_split.json`: `test` (200, перемешан seed), `shots` {"5","20"} (порядок = порядок в промпте), `dev_rows` (6), `resamples` (20 × {"5","20"}). Индекс = позиция строки в CSV с 0 (UDI = индекс+1).
- `typeE_classical.csv`: model, shots, train_set (fixed|resample|none), resample_id, eval_set (test|natural|full), n_train, n_eval(_failures), f1_failure, balanced_accuracy, accuracy, precision/recall_failure, tp/fp/fn/tn, seed, test_mode.
- Строка `typeE_llm.jsonl` = строка run_bank (task_id `E<row_idx>`, variant `classify_row`, value 1/0, correct, confidence…) + shots, row_idx, udi, y_true, split_hash. meta в журнале: experiment typeE|dev, shots, row_idx, task_id, variant, lang, prompt_version, repeat.
## Решения и допущения
- Удалены TWF/HDF/PWF/OSF/RNF (разложение цели) и UDI/Product ID; в CSV хранятся, удаляет `features()`.
- Тест обогащён (40/200, умолчание из HUMAN_NOTES), `natural` даёт 7/200. Пул = всё вне теста. Few-shot стратифицированные и вложенные: 20 = 5+15, 5 = 2+3 (первые из выбранных 20), порядок случайный (seed 42). Разброс: 20 вложенных наборов, rng seed+1, только LogReg/NB.
- Pipeline: ColumnTransformer(OneHot(handle_unknown=ignore) для Type + StandardScaler), LogReg(max_iter=1000, random_state=seed), GaussianNB (плотная матрица). Dummy(most_frequent) обучается на тех же примерах. `eval_set=natural` = все строки вне обучения; `full` = константа «нет отказа» на всех 10000 строк.
- Название датасета в промпт не пишем; в intro есть пояснение L/M/H (из описания UCI) — подтверждает человек (G1).
- parse_label: сначала точная метка («отказ», «нет отказа», failure, no failure, 1, 0, «Исход: …»); иначе шаблоны фраз (отрицания «отказа нет», «low risk of failure» маскируются), побеждает класс, стоящий раньше, заметка «метка не дословная»; эхо шаблона «отказ | нет отказа» → None.
- Оценочный прогон E заблокирован, пока: гипотезы не закоммичены, промпт не заморожен, `data/e_split.json` и `data/ai4i2020.csv` не закоммичены без правок (этот шаг их коммитит). `--dev` (experiment=dev, строки пула) разрешён всегда.
- PROMPT_VERSION оставлен v1: реальных вызовов с промптом ещё не было. prompt_hash изменился (новый CLASSIFY/FORMAT), заморозка — шаг 07.
## Проверка
- `.venv/bin/python -m pytest -q`: 409 passed (было 345).
- `src.typeE all`: CSV 10000×14, split 200/40, пул 9800; повторный `split` даёт тот же файл (тест сверяет). `classical`: 173 строки.
- `scripts/run_experiments.sh --demo`: «=== Готово ===»; E на mock 240 + 2400 вызовов, parse_ok 100%, повтор — всё из кэша. `run_classifier` без --demo: «не запущен» со списком причин, вызовов нет.
- Ревью субагентом: split, утечки, pipeline и метрики подтверждены (classical пересчитан независимо). Исправлено: `parse_label` неверно читал отрицания («Отказа нет», «отказ не случится», «not expected to fail» и т. п.); тест «реальный прогон заблокирован» больше не может дойти до API; в строки добавлен split_hash; гейт проверяет коммит разбиения.
## Что нужно для завершения
- Ничего: шаг done. LLM-прогон E — шаг 09 (после G1 и заморозки промпта на шаге 07).
## Открытые вопросы и риски
- Объём полного прогона (`run_classifier --estimate`): 2400 вызовов; вход ≈ 1,37–2,19 млн токенов (0/5/20-shot ≈ 760/1690/4400 символов на промпт), выход ≈ 48 тыс. (ответ двумя строками) – 720 тыс. (если модель рассуждает). Оценка грубая (1 токен ≈ 2,5–4 символа); после шага 07 уточнить по tokens_in в журнале. Пункт про бюджет есть в HUMAN_TODO G1.
- `prompt_hash` не покрывает форматирование чисел (`typeE.fmt_decimal/fmt_int`) и разбиение: после заморозки их не менять (разбиение страхует гейт и split_hash).
- value может быть `"cannot_determine"` (модель ответила «недостаточно данных») или None (не разобрано): ноутбук должен считать их отдельно, а не передавать в f1_score как класс.
- Если в ответе несколько «ОТВЕТ:», берётся последняя (общее поведение парсера).
- 20-shot промпт ≈ 1,8 тыс. токенов: в qwen num_ctx 8192 помещается.
## Для следующих шагов
- Шаг 07: `.venv/bin/python -m src.run_classifier --dev [--models ...]` → проверить формат ответа E на 6 строках пула (experiment=dev, тест не трогать). Заморозка промпта включает CLASSIFY.
- Шаг 09: `scripts/run_experiments.sh` п. 6 или вручную: `--limit 20`, затем полный. Для GigaChat 1 поток: ~600 вызовов на модель.
- Шаг 11: метрики LLM по `typeE_llm.jsonl` (группы model × shots, только текущий split_hash; F1 по классу 1, balanced accuracy; отказы/неразобранные отдельно), сравнение с `typeE_classical.csv` (train_set=fixed, eval_set=test), разброс по resample, константа eval_set=full. H12: LLM 5-shot vs LogReg 5-shot, LLM 20-shot vs LogReg 20-shot (одни и те же 200 строк → парный тест, например Макнемар).
