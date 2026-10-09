STATUS: done
## Сделано
- `src/bank.py`: схема задачи (docstring), `validate [--final]`, `stats`, `refs [--write]`.
  - `refs` сверяет эталоны с `data/refs/<id>.py` (строка `REFERENCE: a; b`) и `data/d_reference/<id>.py` (строка `METRIC: x`).
  - `--write` записывает `reference_metric` для D.
- `data/tasks.jsonl`, 38 задач:
  - A 8 (все «учебник», это оригиналы пар);
  - B 8 (B1–B4 учебник, 3 ложных + 1 верное; B5–B8 заготовки);
  - C 6 (C1–C3 учебник + рубрики);
  - D 6 (D1–D3 учебник + эталоны);
  - F 10 (все заготовки «свои»).
  Итого 18 `agent_draft` + 20 `todo` (`source: свои`). Все 10 тем покрыты.
- `data/tasks_dev.jsonl`: dev-A1/B1/C1/D1/F1, статус `verified` (dev), эталоны проверены кодом.
- `data/refs/A1–A8.py`, `dev-A1.py`; `data/d_reference/D1–D3.py`, `dev-D1.py`, `_template.py`. Метрики: D1 0.979, D2 0.96, D3 0.8994 (ROC AUC), dev-D1 0.9778.
- `data/checklist_D.md`: 6 пунктов из PDF, оценка 1/0/н/п.
- `docs/hypotheses_guide.md`: H1–H13 (задачи, прогоны, вызовы, метрика, тест, риски), наборы H1+H6+H9 и H5+H10+H12.
- `hypotheses.md`: шаблон на 3 гипотезы, прогноз не заполнен.
- `config/experiment.json`: `hypotheses: []`, `custom_hypothesis_groups`, T/sc/seed/bootstrap, `prompt_version: null`, `typeE` (умолчания).
- `src/checks.py`: `g1` (6 условий, список ✓/✗), `g2` (заглушка, код 1), `hypcommit`.
- Тесты `tests/test_bank.py`, `tests/test_checks.py`; README (п. 3), HUMAN_TODO G1.
## Ключевые файлы и точки входа
- `bank.load_tasks(path)`, `bank.save_tasks`, `bank.validate(tasks, dev, final, hypotheses, languages)`, `bank.validate_files(final)`, `bank.stats_tables(tasks)`, `bank.complete_pairs(tasks, variant)`, `bank.pair_groups`.
- Поля задачи: id, type, topic, source, difficulty, answer_kind, variant, pair_id, text_ru/en, reference, tolerance, reference_status, rubric, reference_code, reference_metric, notes. Необязательные: paraphrase_ru/en (H8), formula_ru/en + verbal_ru/en (H2). Рубрика C: required_arguments (все нужны для 2 баллов) + необязательный bonus_arguments.
- Формат reference: number → float; numbers → список чисел в порядке из условия; bool → true/false; text (C) → кратко метод; code (D) → null + reference_metric; cannot_determine → строка "cannot_determine".
- `checks.HYPOTHESIS_GROUPS`, `checks.last_commit(path)`, `checks.early_eval_calls(log, commit_time)`.
## Решения и допущения
- Пары H1: оригинал в A (учебник) ↔ изменённая версия в F (`modified`): P1–P5 = A1–A5 ↔ F1–F5. Иначе F 10 = 4–5 пар + 3 missing + 2 distractor не помещается. missing_data F6–F8 в паре с A6–A8 (контроль для H3). Distractor F9, F10 без пары.
- В группе `pair_id` ровно один original и ≥1 другая версия, иначе ошибка даже без `--final`.
- H2 через поля `formula_ru/en` (P(D)=…) и `verbal_ru/en` (те же числа в %, словами; без натуральных частот, иначе эффект перевернётся). Черновик есть только у A1. Enum `verbal/formula` в `variant` оставлен для пар внутри банка.
- `--final` дополнительно проверяет:
  - покрытие всех 10 тем;
  - отсутствие «TODO» в любых полях;
  - баланс B: |ложных − n/2| ≤ 1;
  - требования выбранных гипотез: H1 ≥4 пар, H2 ≥5 formula+verbal, H3 ≥3 missing_data, H8 ≥10 paraphrase (+_en, если в languages есть en).
- Всегда (не только --final): id — непустая строка; у `modified` эталон не равен эталону оригинала; у todo-заготовок tolerance может быть null (D4–D6: null, подбирает человек).
- `refs --write` не перезаписывает `reference_metric` у verified-задач, если там уже есть число. `close()` при эталоне 0 использует абсолютный допуск 1e-9.
- B1 переформулирован относительно PDF («ошибка на тесте» вместо «переобучение»: исходная формулировка спорная).
- В условия D добавлены «25% в тест, random_state=42», чтобы метрику можно было сравнить. Стратификацию и скейлер не подсказываем: их ловит чек-лист.
  - Допуски D (отн.): 0.02 / 0.02 / 0.03, взяты по разбросу разумных альтернатив (в notes).
  - D3 использует ROC AUC, потому что F1 у корректных решений разбросан от 0.48 до 0.68.
- G1, пункт 6: оценочными считаются все записи журнала, кроме experiment `smoke` и `dev`. Время коммита — последний коммит, изменивший hypotheses.md (%cI), сравнивается с ts_utc с учётом зоны.
- G1: smoke-тест запускается, только если есть все ключи (иначе «пропущен»). hypotheses.md: обязательны все 7 строк полей, `**` в прогнозе допускается. Список hypotheses в config/experiment.json сверяется с версией из коммита гипотез (сам файл можно менять другим шагам).
## Проверка
- `.venv/bin/python -m pytest -q`: 53 passed (19 старых + 34 новых).
- Ревью субагентом: эталоны A1–A8, B1–B4 и dev пересчитаны независимо, расхождений нет. Исправлены: двусмысленности в dev-A1 (формула α с ½), dev-F1 (заменена), инструкциях F7/F9/F10/D6; рубрики C (score_2 = все обязательные аргументы; DBSCAN в C2 не выше 1 без обсуждения eps); H2-версии A1; проверки id, нулевого эталона и hypotheses.md.
- `src.bank validate`: 0 ошибок, 3 сводных предупреждения (todo, agent_draft, text_en).
- `src.bank validate --final`: код 1 (заготовки todo, нет verified) — ожидаемо.
- `src.bank refs`: «Расхождений: 0».
- `src.checks g1`: код 1, понятный список (нет ключей, банк не финальный, гипотезы не выбраны); пункт про журнал ✓.
## Что нужно для завершения
- Ничего, шаг done. Заполнение заготовок, проверка черновиков, выбор гипотез — пункты G1 в HUMAN_TODO.
## Открытые вопросы и риски
- У H1 мало наблюдений (5 пар × 4 модели). Лимит 40 задач: можно добавить только P9, остальное — заменой заготовок.
- Баланс B смешан с источником: учебные B1–B4 = 3 ложных + 1 верное, заготовки B5–B8 = 3 верных + 1 ложное (иначе не 4/4). Сравнение «учебник/свои» на B этим искажено — отметить в ограничениях или переставить вердикты (решает человек).
- Журнал только дописывается: любая запись с experiment ≠ smoke/dev до коммита гипотез навсегда закроет G1. Шаг 07 обязан писать experiment="dev", демо — только в demo/.
## Для следующих шагов
- Шаг 04: `answer_kind` задаёт формат ответа/парсер. Для numbers порядок значений задан в условии. A4 ожидает «0.4; 1». Эталоны-проценты хранятся долями (0.161).
- Шаг 05: run_extra берёт `paraphrase_ru/en` и `verbal_ru/en` у задач, где поле заполнено и нет TODO. Для H1/H3 пары брать через `bank.pair_groups`. exec_code сравнивает метрику с `reference_metric` по относительному `tolerance`.
- Шаг 08: хэш коммита гипотез — `checks.last_commit(checks.HYPOTHESES_MD)`. Прогонять только при `src.checks g1` = 0.
- Шаг 10: реализовать `checks.g2()` (сейчас заглушка). Рубрики B/C лежат в `rubric`, чек-лист D — в `data/checklist_D.md`.
