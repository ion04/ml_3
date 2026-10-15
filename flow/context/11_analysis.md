STATUS: deferred
## Сделано
- Попытка 2 (15.10): реальные данные шагов 08–10 есть, ручных оценок нет (`src.checks g2`: 4 из 4 ✗). Ноутбук впервые прогнан на **реальных** данных во временном git worktree (/tmp, удалён), чтобы `results/` и `report/` в репозитории не менялись: выполняется полностью.
  - Второй прогон в той же копии со случайными оценками в листах (через `demo_grades.fill`, только в /tmp): ветки каппы, C/D, баллов 0–2, H11 и выборки парсера работают на реальной структуре листов.
- Исправлено и добавлено:
  - ячейка 0.3 падала (ValueError), когда `parser_spotcheck.csv` есть, но `human_agrees` пуст. Теперь печатается «ещё не заполнена»;
  - **полнота прогонов**: `an.coverage_table`, `expected_cot`, `expected_followup`. Таблицы «полнота прогона» в разделах E, H6 и H9 и предупреждение «ВНИМАНИЕ: … неполный у: <модель>» → `summary["coverage"]` (typeE, cot, followup_doubt). Сейчас неполный только deepseek-v4-pro: cot 6/16, followup 0/26, E 0/600;
  - `an.add_full_input`: столбец `tokens_in_full`, для GigaChat это `tokens_in + tokens_cached` из журнала. Используется в H6 `tokens_total_ratio` и в таблице цены (`tokens_in_full_mean`). ₽ по-прежнему считаются по `tokens_in`;
  - в разделе D: предупреждение, если упавший код отмечен «метрика верна» (по `INSTRUCTIONS.md` там должен стоять 0);
  - в `inputs` (sha256 входов) добавлен журнал `logs/calls.jsonl`.
- 2 новых теста в `tests/test_analysis.py`. HUMAN_TODO G3.1: правило про токены H6 и проверка таблиц полноты.
## Ключевые файлы и точки входа
- `analysis.ipynb` (сохранён с `DEMO = False`, без выходов), `src/analysis.py`, `scripts/run_analysis.sh [--demo]`.
- Ключи `results/summary_metrics.json`: прежние (counts, kappa, accuracy_*, mcnemar_models, typeD, typeE, typeE_ci, hypotheses.H*, cost, cost_typeE, local_model, pricing, parser_spotcheck, inputs, bank, runs) + новый `coverage`.
## Решения и допущения
- Реальные выходы, рисунки и summary_metrics.json в репозиторий не записывал. Без C/D (ручные оценки) точность по моделям считается только по A/B/F, поэтому числа неокончательные. Каппа и тарифы тоже не проверены.
- Ожидаемое число ответов: cot = все задачи типов `extra_runs.cot.types` для моделей main и `extra_models`; followup = ответы main, верные **при прогоне** (как `run_extra.jobs_followup`); E = n_test × число вариантов shots.
- Правила подсчёта из попытки 1 не менялись (подтверждение — HUMAN_TODO G3.1).
## Проверка
- `.venv/bin/python -m pytest -q`: 594 passed.
- Реальные данные (копия): nbconvert без ошибок. Два запуска подряд дают побайтно одинаковый `summary_metrics.json`. Выборочно сверил: CI [0; 0] у H1 «все» верен (оба расхождения в одной паре P3), n в H6/H9 совпадают с манифестом.
- `scripts/run_analysis.sh --demo`: без ошибок.
- `git status` перед коммитом: изменены только `analysis.ipynb`, `src/analysis.py`, `tests/test_analysis.py`, `HUMAN_TODO.md` и этот файл.
## Что нужно для завершения
- G2: листы r1/r2, spotcheck, тарифы → `.venv/bin/python -m src.checks g2` = 0, коммит листов.
- Желательно до этого: дозапрос DeepSeek (HUMAN_TODO G1.8) → `src.autograde` → коммит. Либо человек решает обойтись тремя моделями в H6/H9/E (тогда предупреждения о полноте остаются и идут в ограничения).
- Затем повторить шаг 11:
  1. `scripts/run_analysis.sh`;
  2. раздел 1: если κ < 0,6 у B, C или D — STATUS: blocked (пункт уже есть в G2);
  3. проверить таблицы полноты и предупреждение раздела D;
  4. коммит `analysis.ipynb` (с выходами), `report/figures/`, `results/summary_metrics.json`.
## Открытые вопросы и риски
- `graded_auto.jsonl` нужно пересобрать (`src.autograde`) после дозапроса DeepSeek и после решения G2 о числах «0.32;0.1987;0.1213» у qwen. Ноутбук его не пересобирает.
- Таблица «Тип E: токены и стоимость» отсортирована по алфавиту, а не по ролям (косметика).
- В `flow/README.md` и др. могут быть чужие незакоммиченные правки — не трогал.
## Для следующих шагов
- Шаг 12: `build_report` пока не выводит `summary["coverage"]` и `tokens_in_full_mean`. Если данные DeepSeek останутся неполными, добавить таблицу полноты в отчёт (ограничения).
- Воспроизводимость: повторный `scripts/run_analysis.sh` даёт тот же JSON. Журнал входит в `inputs`, поэтому новый smoke-вызов меняет его хэш.
