STATUS: deferred
## Сделано
- Попытка 3 (15.10, 13:18 MSK). G2 не пройден (`src.checks g2`: 4 из 4 ✗), поэтому выходы ноутбука в репозиторий не записывались.
- **Дозапрос DeepSeek для E** (работа шага 09: его запуск и повтор на шаге 10 обрывались вместе с сессией, в журнале было 468/600). `src.run_classifier --models deepseek-v4-pro --shots 0,5,20`: 132 новых + 468 из кэша, ошибок 0, parse_ok 600/600. Вызовы 10:03–10:22 UTC. Затем `typeE_metrics check` → 0, `typeE_metrics` (12 строк LLM, все complete), `autograde` (2696 строк), `manifest`. Коммит 1ce1afb. HUMAN_TODO G1.8: пункт E закрыт, пункт про полноту (шаги 11, 12) поправлен.
- Ноутбук прогнан на реальных данных во временном worktree (/tmp, удалён): nbconvert без ошибок. `summary["coverage"]` полный у всех моделей: E 600/600, cot 16/16, followup 19–26 = ожидаемому. Предупреждения только про G2: листы, выборка парсера, тарифы.
- Косметика: таблица «Тип E: токены и стоимость» теперь в порядке ролей (`an.order_models`), а не по алфавиту.
## Ключевые файлы и точки входа
- `analysis.ipynb` (сохранён с `DEMO = False`, без выходов), `src/analysis.py`, `scripts/run_analysis.sh [--demo]`.
- Ключи `results/summary_metrics.json`: accuracy_by_model/type/model_type, bank, bootstrap_n, cost, cost_typeE, counts, coverage, demo, hypotheses, hypotheses_selected, inputs, kappa, kappa_below_min, local_model, mcnemar_models, parser_spotcheck, pricing, rubric_scores, runs, seed, typeD, typeE, typeE_ci.
- Рисунки (в report/figures/ после финального запуска): heatmap_model_type, accuracy_ci, cost_accuracy, typeE_shots, typeD_checklist, H1/H3/H5/H6/H9/H10/H11/H12.
## Решения и допущения
- Дозапрос E — это продолжение прерванного прогона: техошибок не было, ответы из кэша не перезапрашивались. Вызовы видны в журнале и манифесте. Это не перезапрос ради лучшего ответа.
- Ответы DeepSeek для E получены через ~7 ч после остальных моделей (03:19–03:51 UTC). Упомянуть в ограничениях (текст человека, пункт в HUMAN_TODO).
- Правила подсчёта не менялись (подтверждение — HUMAN_TODO G3.1, пункт шага 11).
- `docs/bank_overview.md` (не отслеживается, создан 13:05 не этим шагом) не трогал и не коммитил.
## Проверка
- `.venv/bin/python -m pytest -q`: 596 passed.
- Реальные данные (копия): nbconvert код 0, ещё два запуска подряд дают побайтно одинаковый `summary_metrics.json` (md5 совпал).
- `scripts/run_analysis.sh --demo`: код 0.
- `git diff --numstat logs/calls.jsonl`: 600 добавлено, 0 удалено.
## Что нужно для завершения
- G2 (человек): листы `grading/sheet_r1.csv`, `sheet_r2.csv`, `parser_spotcheck.csv`, тарифы → `.venv/bin/python -m src.checks g2` = 0, коммит листов. Если G2 изменит правило A2/F2 (несколько чисел), сначала выполнить `src.autograde`.
- Затем повторить шаг 11:
  1. `scripts/run_analysis.sh`;
  2. раздел 1: если κ < 0,6 у B, C или D — STATUS: blocked (пункт есть в G2);
  3. проверить предупреждение раздела D (упавший код с «метрика верна»);
  4. коммит `analysis.ipynb` (с выходами), `report/figures/`, `results/summary_metrics.json`.
## Открытые вопросы и риски
- Spotcheck парсера пересобран после дозапроса (человек ещё ничего не отметил). Если отметки появятся, `autograde` их не перезапишет.
- Списание DeepSeek за зависшие запросы 15.10 — пункт шага 08 в «Добавлено агентами».
- Handoff шага 09 всё ещё `deferred`: при повторном запуске флоу всё возьмётся из кэша (новых вызовов 0), и шаг закроется.
## Для следующих шагов
- Шаг 09/10 при повторе: данные E и graded_auto уже полные, пересобирать нечего.
- Шаг 12: таблицы полноты сейчас без пометок «неполный». Журнал входит в `inputs`, поэтому новый вызов (smoke) меняет его хэш.
