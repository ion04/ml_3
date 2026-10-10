STATUS: done
## Сделано
- Аудит на соответствие PDF (стр. 1–9) и CLAUDE.md. Итоги — `docs/qa_report.md`, карта «требование → код → статус» — `docs/requirements_traceability.md` (~70 строк, открытых пробелов нет).
- Исправления в коммите 6dcc7d9:
  - `grade.close`: граница ±1% входит в допуск при любой погрешности дробных чисел (`FLOAT_SLACK`). Раньше 0,16261 при эталоне 0,161 считалось неверным.
  - CI у каждой точности:
    - `analysis.compare_conditions` → `acc_a/b_ci_*`, `name_conditions`, усы в `plot_two_conditions`;
    - `typeD_table` → CI для pass@1, correct, method_error_any;
    - `typeE_ci_table` + `logreg_rows` → F1 и balanced accuracy LLM/LogReg с бутстрепом строк, ячейка в разделе 5 ноутбука, `summary["typeE_ci"]`, таблица в отчёте 3.6.
  - Состав банка и вызовы моделей отчёт берёт из ноутбука: ячейка раздела 0 → `an.bank_summary`, `an.runs_summary` → `summary["bank"]`, `summary["runs"]`. `build_report` больше не читает `data/tasks.jsonl`, из `run_manifest` берёт только промпт, окружение, железо и split_hash.
  - dev-задачи: `agent_draft`. `validate` допускает его, а `--final` (G1) требует `verified`. B: ровно половина ложных.
  - notes: A4 (точка (1; 0) на границе зазора, вопрос 12), D1 и D3 (диапазоны корректных решений).
  - Чистка кода: дубли JSONL и split_hash, мёртвый код, устаревшие комментарии; поведение и `prompt_hash` d946bcbce2dd не менялись.
- `tests/test_known_values.py`, 62 теста на известных значениях: бутстреп, Макнемар, ECE, каппа, F1/BA, ₽, Клоппер–Пирсон, парсер, допуск.
- `docs/audit/recompute_refs.py`: независимый пересчёт A1–A8, dev-A1 и вариантов D1–D3, dev-D1 — всё в допуске.
- HUMAN_TODO переписан: «С чего начать» с временем, разделы гейтов, проверка у каждого пункта, плюс 3 пункта шага 13. README: «С нуля до сдачи» (ПК gpu, gpu_ollama.sh).
## Ключевые файлы и точки входа
- `docs/qa_report.md`, `docs/requirements_traceability.md`, `docs/audit/recompute_refs.py` (`.venv/bin/python docs/audit/recompute_refs.py`, ~35 с).
- `an.bank_summary(tasks)`, `an.runs_summary(log, models)`, `an.typeE_ci_table(preds, n_boot, seed)`, `an.logreg_rows(logreg, preds)`, `an.name_conditions(table, a, b)`.
- `summary_metrics.json`: новые ключи `bank`, `runs`, `typeE_ci`. У таблиц гипотез появились столбцы `acc_<условие>_ci_low/high`; у H4 столбец `delta_ru_minus_en` переименован в `diff_ru_minus_en`.
## Решения и допущения
- dev-эталоны требуют `verified` только в `--final`. До G1 отладка промпта на dev не блокируется.
- Метрики надёжности (H7/H8/H9) по-прежнему считаются только для выбранных гипотез. Выбор вынесен в HUMAN_TODO G1.4 (шаг 13).
- Доли H3, H7-разброс, H8, H9 оставлены с CI Клоппера–Пирсона. Средний балл 0–2 без CI: это не точность.
- Тип E на тепловую карту не добавлен, на рисунке `typeE_shots` нет усов (CI есть в таблице 3.6). Описать в методике: HUMAN_TODO G3.1.
- Реальный `report/report.md` пересобран: без summary в разделе 2.5 и в таблице вызовов стоит «нет данных».
## Проверка
- `.venv/bin/python -m pytest -q`: 578 passed (было 515).
- `scripts/run_experiments.sh --demo` с пустого `demo/`: код 0, 27 с. После исправлений `scripts/run_analysis.sh --demo` выполнен дважды, `summary_metrics.json` совпал побайтно.
- sha1 файлов в `logs/`, `results/`, `grading/`, `report/` до и после demo не изменились.
- Копия ноутбука с `DEMO = False` выполняется без ошибок.
- `bash -n scripts/*.sh`: ок. `.env` не в истории git, ключей не найдено. Каждая версия `logs/calls.jsonl` — префикс следующей.
- `src.bank validate`: 0 ошибок. `src.bank refs`: расхождений 0. `prompt_hash` = d946bcbce2dd.
## Что нужно для завершения
- Ничего: шаг done. Остальное — работа человека и ключи: HUMAN_TODO G1, G2, G3. Затем `python3 flow/flow.py run` доделает отложенные шаги 07–12.
## Открытые вопросы и риски
- Не проверено на реальных ответах: вид новых CI-таблиц и усов на реальных данных, длина таблицы `runs.models` в отчёте.
- `same_answer` в run_extra и error_candidates и `close` в bank и grade называются одинаково, а работают по-разному. Оставлены как есть, отмечено в qa_report.
- Если человек выберет вариант (б) для метрик надёжности, это ≈ +700 вызовов.
## Для следующих шагов
- Шаг 11 (повтор): ячейка `bank`/`runs` в разделе 0 обязательна для отчёта. Без `summary_metrics.json` разделы 2.1 и 2.5 отчёта пусты.
- Шаг 12 (повтор): после `scripts/run_analysis.sh` проверить, что в разделах 2.1 и 2.5 есть таблицы, а в разделе 3.6 — таблица CI типа E.
- Шаг 07 (повтор): dev-задачи в статусе `agent_draft` прогоняются как обычно. `checks g1` не пройдёт, пока человек не поставит `verified`.
