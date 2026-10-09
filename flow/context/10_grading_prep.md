STATUS: deferred
## Сделано
- Реальных прогонов нет (шаги 07–09 deferred: нет ключей, G1 не пройден), поэтому всё проверено на demo. Реальные `graded_auto.jsonl` и листы не создавались: `autograde`/`blind` без --demo дают код 2 и «проверять нечего».
- `config/pricing.json`: ₽ за 1 млн токенов, вход и выход (`rub_per_1m_in/out`), с `source_url`, `date_checked` 2026-10-10, `original`, `note`; `verified_by_human: false`. Курс ЦБ 84,9048 ₽/$ на 10.10.2026 (`usd_rub`). Цены:
  GigaChat-2-Max 650/650, YandexGPT Pro 5.1 800/800, deepseek-v4-pro 112,07/336,22 (пик, без кэша), qwen3-8b 0 (время и память); выключенные модели тоже (GigaChat-3-Ultra null).
- `src/autograde.py` → `results/graded_auto.jsonl` + `grading/parser_spotcheck.csv`; `src/blind.py` → `grading/sheet_r1.csv`, `sheet_r2.csv`, `key_DO_NOT_OPEN.csv`; `grading/INSTRUCTIONS.md`; `checks.g2()`.
- `scripts/run_experiments.sh`, шаг 7: autograde → blind; код 2 не ошибка, код 1 у blind — остановка.
- README п. 7, HUMAN_TODO G2 (переписан), `tests/test_grading.py` (38 тестов), память `llm-provider-pricing`.
## Ключевые файлы и точки входа
- `.venv/bin/python -m src.autograde [--demo] [--force-spotcheck]`, `-m src.blind [--experiments main,paraphrase] [--demo] [--force]`, `-m src.checks g2`.
- `autograde.build(results_dir, tasks)`, `d_exec_records`, `grade_d`, `spotcheck_sample`, `read_csv`/`write_csv` (UTF-8 BOM), `human_started`, `BLIND_EXPERIMENTS`. `blind.build(...)→(sheet, key)`, `anonymize`, `name_regex`, `answer_sha`, константы `GRADE_COLUMNS`, `CHECKLIST`, `SCORE_COL`, `CORRECT_COL`, `NA="—"`.
- graded_auto: check (answer|d_exec), experiment, lang, model_key, task_id, type, variant, repeat, call_id, correct, score, reason, correct_at_run, needs_manual, manual_part; E: shots, row_idx, split_hash; answer: value, parse_ok, parse_notes, reference; d_exec: pass_at_1, metric, reference_metric, tolerance.
- Ключ: item_id, type, task_id, model_key, model_reported, experiment, lang, variant, call_id, required (обязательные столбцы через «;»), answer_sha.
## Решения и допущения
- Автопроверка заново разбирает `response_text` и сравнивает с текущим банком: правка эталона учитывается без новых вызовов. Пропускаются: техошибки, задачи с reference=null, C, ответы E с чужим split_hash; таблицы dev_* и selfconsistency_vote_* не читаются.
- D: correct=True/False автоматически, если метрика извлечена (False также при «нет кода», «упал», «таймаут»). Если метрики или эталона нет — None.
- В листах корректность D ставят люди везде, кроме авто-«верно»: метрика из stdout — только подсказка. Итог для D в анализе брать у людей (`d_result_correct`), «—» = авто-верно.
- В листы идут B, C, D из main и paraphrase (все языки). Строки сгруппированы по типам, внутри перемешаны с seed 42, r1 = r2.
- Анонимизация: вырезается `<think>` (и незакрытый); бренды, ключи models.json и model_reported → «[название скрыто]»; ответ с finish_reason blacklist/content_filter → пометка. Побочно маскируются «Алиса», «Яндекс» в обычном тексте.
- Если кода D нет, в чек-листе ставится «н/п». Это правило подтверждает человек (G2).
- Выборка парсера: до 5 E «метка не дословная», до 10 подозрительных всего, остальные случайные по кругу по типам. Seed из config. human_agrees принимает только «да»/«нет».
- Защита от перезаписи: лист или spotcheck, где человек что-то заполнил (или нечитаемый формат), не перезаписывается. Если ответы те же — код 0, если набор изменился — код 1. CSV читается и с «;», и в cp1251.
- g2: обязательные клетки из ключа («—» в них = пусто); answer_sha ловит лист от другого набора ответов; pricing: verified_by_human и цены включённых моделей; полностью совпавшие r1 и r2 — предупреждение.
- DeepSeek: цена пика без кэша — верхняя оценка; USD без НДС, рубли с НДС. С 01.09.2026 новые клиенты GigaChat платят через cloud.ru (569,34 ₽ за 1 млн, указано в note). Выбор цены — за человеком.
## Проверка
- `.venv/bin/python -m pytest -q`: 467 passed (было 429).
- `scripts/run_experiments.sh --demo`: «=== Готово ===»; graded_auto 2874 строки; листы 96 (B 48, C 24, D 24; D с ручной корректностью 19); spotcheck 20. Повторный `blind --demo` даёт те же файлы.
- Без demo: autograde и blind → код 2, файлов нет; `checks g2` → код 1 (3 из 3 ✗).
- Ревью субагентом: исправлены перезапись листов после Excel («;», cp1251), «—» в обязательной клетке, авто-«неверно» D, пробелы и ложные срабатывания анонимизации, расхождение d_exec, старый split E, перекос выборки к E, падения на пустых полях.
## Что нужно для завершения
- G1 → шаги 07, 08, 09 done (реальные results/*.jsonl, d_exec.jsonl).
- Повтор шага 10: `src.exec_code --experiment main,paraphrase` → `src.autograde` → `src.blind`; проверить глазами/`grep -i` по брендам, что в `answer_text` нет названий моделей; коммит `results/graded_auto.jsonl`, `grading/*.csv`.
- G2 (человек): листы, spotcheck, тарифы → `src.checks g2` = 0.
## Открытые вопросы и риски
- Анонимизация по словарю: новые самоназвания или стиль ответа могут выдать модель. На реальных ответах просмотреть глазами.
- Цены не сверены человеком. Страница Yandex отдаёт капчу скриптам, числа прочитаны WebFetch.
- Ключ лежит в grading/ рядом с листами: выдавать проверяющим только их файл.
## Для следующих шагов
- Шаг 11: лист → ключ по item_id; B: вердикт из graded_auto + балл людей; D: pass@1 из graded_auto, корректность — `d_result_correct` людей (при «—» — авто True); d1..d6 (1/0/н/п), «н/п» по всем пунктам (нет кода) исключать из доли ошибок; значения через `checks.norm`; каппа по score_0_2 и d*.
- ₽: tokens_in·rub_per_1m_in/1e6 + tokens_out·rub_per_1m_out/1e6; в demo mock-модели брать по роли (mock-ru1 → gigachat-2-max и т. д.).
- Spotcheck: доля «да» по selection=«случайный» отдельно от «подозрительных».
