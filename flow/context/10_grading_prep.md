STATUS: deferred
## Сделано
- Попытка 2 (15.10): код шага уже был (попытка 1, demo). Теперь собрано по реальным данным шагов 08–09.
  - `src.autograde` → `results/graded_auto.jsonl`: 2060 строк, с автооценкой 2060. Состав: main A/B/F 104, cot 70, followup 62, typeE 1800, d_exec 24. Пропущены: техошибка 1 (deepseek cot), C 24 (идут в листы).
  - `grading/parser_spotcheck.csv`: 20 строк (10 «подозрительных», 10 случайных).
- `src.blind` → `grading/sheet_r1.csv`, `sheet_r2.csv` (80 строк: B 32, C 24, D 24; r1 = r2), `key_DO_NOT_OPEN.csv`.
  - Корректность D вручную нужна в 6 строках: D4 у всех 4 моделей (метрика вне допуска) и 2 упавших кода gigachat (D1, D6). В остальных стоит «—» (авто-верно).
- Анонимизация проверена grep по брендам, `<think>`, «as an AI», «я — языковая модель» в answer_text и exec_result: 0 совпадений, замен «[название скрыто]» тоже 0.
- `config/pricing.json`: в примечание DeepSeek добавлено, что с 14.10 в журнал пишется `tokens_cached`. Цены не менялись (сверены 10.10, курс ЦБ 84,9048), `verified_by_human: false`.
- HUMAN_TODO:
  - G2: «листы готовы»; новое решение «несколько чисел в числовом ответе» (qwen A2/F2);
  - G1.8: к ручному дозапросу DeepSeek добавлен `src.autograde`.
## Ключевые файлы и точки входа
- `.venv/bin/python -m src.autograde [--demo] [--force-spotcheck]`, `-m src.blind [--demo] [--force]`, `-m src.checks g2`; `scripts/run_experiments.sh`, шаг 7.
- `results/graded_auto.jsonl`, `grading/{sheet_r1,sheet_r2,key_DO_NOT_OPEN,parser_spotcheck}.csv`, `grading/INSTRUCTIONS.md`, `config/pricing.json`.
- API модулей и поля описаны в handoff попытки 1 (git show 894b92d:flow/context/10_grading_prep.md) и в docstring модулей.
## Решения и допущения
- Статус deferred: нет ответов DeepSeek для cot (10), followup и E (600). Когда шаги 08/09 их дозапросят, нужно снова запустить `src.autograde`, иначе `graded_auto` устареет: `run_analysis.sh` его не пересобирает.
- Листы от DeepSeek не зависят (в них только main, который полный: 152/152). Поэтому G2 можно начинать уже сейчас.
- Spotcheck при новом запуске пересобирается, только если человек ещё ничего не отметил. Уже начатый лист сохраняется.
- qwen3-8b A2/F2: «ОТВЕТ: 0.32;0.1987;0.1213» → парсер берёт первое число → неверно (правило шага 04). Парсер не менял: правило оценки решает человек (HUMAN_TODO G2).
- D4 (kNN без масштабирования) у всех моделей вне допуска 5%. Это задумано (notes задачи), чек-лист оценивают люди.
## Проверка
- `.venv/bin/python -m pytest -q`: 592 passed.
- Повторный запуск `autograde` + `blind`: файлы побайтно те же (md5), код 0.
- `src.checks g2` → код 1, 4 из 4 ✗ (листы, spotcheck и тарифы не заполнены) — ожидаемо до G2.
- Журнал не трогался. `.env`, `demo/` не в коммите.
## Что нужно для завершения
- DeepSeek отвечает (`.venv/bin/python -m src.smoke_test` → deepseek ok) → шаги 08/09 дозапрашивают → `.venv/bin/python -m src.autograde` → коммит `results/graded_auto.jsonl` (и `grading/parser_spotcheck.csv`, если он пересобран).
- Либо человек решает обойтись без DeepSeek в H6/H9/E (HUMAN_TODO G1.8). Тогда текущий graded_auto окончательный, шаг можно закрыть как done.
- G2 (человек): листы r1/r2, spotcheck, тарифы → `src.checks g2` = 0.
## Открытые вопросы и риски
- Стиль ответа может выдать модель: LaTeX `\frac` у gigachat, markdown-заголовки и длина ответа. Убрать это нельзя без правки содержания.
- Ключ лежит в grading/ рядом с листами: проверяющим выдавать только их файл.
- Если человек изменит правило парсера (A2/F2) или эталоны, нужно перезапустить `src.autograde`. Вызовы не нужны.
## Для следующих шагов
- Шаг 11: правила сопоставления листов и ключа — в handoff попытки 1 (item_id → ключ; D: pass@1 из graded_auto, корректность из `d_result_correct`, «—» = авто True; «н/п» по всем пунктам исключать).
- Шаг 11: перед анализом убедиться, что graded_auto собран после последнего прогона: mtime или число строк typeE = строкам `typeE_llm.jsonl`.
- ₽ = tokens_in·rub_per_1m_in/1e6 + tokens_out·rub_per_1m_out/1e6. Для GigaChat tokens_in уже без кэша.
