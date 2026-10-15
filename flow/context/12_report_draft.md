STATUS: deferred
## Сделано
- Попытка 2 (15.10). Каркас отчёта (`src/build_report.py`, `src/error_candidates.py`, `report/selfcheck_questions.md`, `report/slides_outline.md`, `scripts/check_reproducibility.sh`) был готов с попытки 1 (подробности: `git show 1804d89:flow/context/12_report_draft.md`). Теперь он впервые проверен на **реальных** данных шагов 08–11, без ручных оценок (G2 не пройден).
- Проверка шла во временном git worktree в /tmp (сейчас удалён): `scripts/run_analysis.sh` → report.md на 687 строк, все разделы и 13 рисунков; `error_candidates --force` → 15 кандидатов с цитатами. В репозиторий **не** записаны `summary_metrics.json`, рисунки и error_candidates.md: без G2 числа неокончательные, а кандидаты раскрыли бы модели до слепой проверки.
- Правки `build_report` по реальному виду отчёта:
  - 3.1 — таблица «Полнота прогонов» (`summary["coverage"]`: получено / ожидалось, пометка **неполный**); неполные прогоны попадают в справку к ограничениям;
  - тип D разбит на две таблицы (pass@1, корректность, методологическая ошибка с CI; чек-лист отдельно) вместо одной на 20 столбцов;
  - в таблицу цены добавлен `tokens_in_full_mean` с пояснением; таблица «Тип E: токены» упорядочена по ролям;
  - в критерии гипотезы убрана двойная точка;
  - ID каталога Yandex в `gpt://…/yandexgpt-5.1` заменяется на `<YANDEX_FOLDER_ID>`.
- `check_reproducibility.sh`: `--exclude '.venv'` (без «/»), иначе symlink .venv копировался и `uv venv` падал.
- **Воспроизводимость на реальных данных (предварительная)**: новое .venv из requirements.txt → Restart & Run All → `summary_metrics.json` совпал побайтно (sha256 5b5518a54b59), report.md копии тоже. Записано в `docs/reproducibility.md`.
- `report/report.md` в репозитории пересобран: методика из манифеста, результаты — «нет данных». README (строка 7a «дозапрос недоступной модели», критерии 7/9, описание 3.1), HUMAN_TODO (G3.1, G3.5, «Можно в любой момент», новый пункт про ID каталога Yandex), slides_outline (слайд 8). +2 теста.
## Ключевые файлы и точки входа
- `.venv/bin/python -m src.build_report [--demo] | check`; новые функции `br.coverage_table`, `incomplete_runs`, `sorted_by_role`, `YANDEX_FOLDER_RE`.
- `scripts/run_analysis.sh` (ноутбук → error_candidates → build_report), `scripts/check_reproducibility.sh` (коды 0/1/2/3).
## Решения и допущения
- Реальные выходы ноутбука до G2 не коммитятся (так же решил шаг 11). Проверка воспроизводимости записана как «предварительная», а финальная строка в журнале оставлена пустой.
- ID каталога Yandex скрыт только в отчёте. В журнале и results он остаётся: журнал не переписывается, и это не ключ. Решение о публичности репозитория принимает человек (HUMAN_TODO, «Добавлено агентами»).
- Маскировка применяется ко всему тексту отчёта, включая блоки человека: если там есть `gpt://<каталог>/`, он тоже скроется.
## Проверка
- `.venv/bin/python -m pytest -q`: 596 passed.
- Реальные данные в /tmp: ноутбук и build_report без ошибок; в 3.1 у deepseek-v4-pro E 0/600, cot 6/16, followup 0/26 «неполный», остальные модели полные.
- `scripts/check_reproducibility.sh` (worktree): «Итог: совпадает», код 0, ≈2 мин. `scripts/run_analysis.sh --demo`: без ошибок.
- `grep -ril mock report/` пусто; `build_report check` → код 1 (14 пустых блоков человека, ожидаемо). Журнал не трогался.
## Что нужно для завершения
- G2: листы r1/r2, spotcheck, тарифы → `.venv/bin/python -m src.checks g2` = 0; κ ≥ 0,6 (иначе blocked).
- Желательно: дозапрос DeepSeek (HUMAN_TODO G1.8) → `src.autograde`. Либо решение человека обойтись тремя моделями.
- Затем повтор шагов 11 и 12:
  1. `scripts/run_analysis.sh`;
  2. проверить report.md: нет «нет данных» в 2–3, 3.1 без «неполный» (или неполнота есть в справке к ограничениям);
  3. `scripts/check_reproducibility.sh` и строка «реальные данные, финально» в `docs/reproducibility.md`;
  4. коммит `analysis.ipynb`, `report/`, `results/summary_metrics.json`, `docs/`.
## Открытые вопросы и риски
- Hypotheses-таблицы — сырые имена колонок summary (`diff_ci_low` и т. п.); читаемо, но не по-русски. Переименовывать не стал: имена совпадают с ноутбуком и summary.
- 3.9 показывает невыбранные H3, H5, H10–H13 (данные есть); убрать можно только через ноутбук (G3.1).
- qwen A2/F2 «0.32;0.1987;0.1213» → в кандидатах ошибок как «уверенность 100, неверно»; зависит от решения G2 о правиле парсера.
- Демо: qwen3-8b-think в cot «0/12 неполный» (extra_models не мокаются) — только demo/, на отчёт не влияет.
## Для следующих шагов
- Шаг 13 / повтор: сверить report.md с PDF, `grep -ri mock report/` пусто, `grep b1g report/` пусто, `build_report check`.
- Перед финальным `run_analysis.sh` проверить, что `graded_auto.jsonl` собран после последнего прогона (handoff шага 10).
