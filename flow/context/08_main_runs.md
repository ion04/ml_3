STATUS: deferred
## Сделано
- 15.10: `src.checks g1` → 0 (6/6), туннель к gpu отвечает (Ollama 0.40.2 :11435). Гипотезы H1, H6, H9 закоммичены 67d7108 (14.10 22:00 +03), все оценочные вызовы позже.
- main (`run_bank --experiment main --models all --lang ru`): 152 вызова (38 × 4), ошибок 0, parse_ok 149/152 (gigachat 35/38, остальные 38/38), finish_reason=length 0. Время 00:39–00:43 UTC.
- `run_extra selected` (H1 считается из main; H6 → cot A+B; H9 → followup_doubt A/B/F):
  - cot: 70 из 80 ответов (gigachat, yandex, qwen3-8b, qwen3-8b-think по 16/16, deepseek 6/16), parse_ok 70/70;
  - followup_doubt: 62 вызова (gigachat 21, yandex 22, qwen 19), ошибок 0, parse_ok 62/62. DeepSeek ещё не прогнан.
- `exec_code --experiment main,paraphrase`: 24 запуска (6 D × 4 модели), код найден 24, rc=0 у 22, таймаутов 0 → `results/d_exec.jsonl`, `results/d_code/`.
- **DeepSeek API перестал отвечать с 00:43 UTC**: запросы не начинали обрабатываться, сервер слал keep-alive. Даже smoke «2+2» в 02:01 не получил ответа за 300 с, остальные модели отвечали за 1–2 с. Три пробы cot A7 (01:40–03:07) по 5 попыток — все TimeoutError. В журнале 31 запись с ошибкой, все видны (attempt 1–5).
- `src/llm.py::_with_deadline`: общий срок ответа openai_compat = `timeout_s`. По таймауту закрывается клиент, возникает TimeoutError (её ask повторяет). Раньше запрос висел 14 мин при timeout 300 с. Тест `test_with_deadline`.
- `tests/test_bank.py::test_cli_validate_exit_codes` больше не зависит от состояния реального банка: он стал финальным, и тест падал.
- `src.server_info --write` (железо gpu на 15.10, прежняя note), `src.manifest` пересобран, `pip freeze` = requirements.txt (не менялся). README п. 5: абзац о сроке ответа. HUMAN_TODO: G1.8 (дозапрос DeepSeek), «Добавлено агентами» (сверить списание DeepSeek). Память: факт о зависании DeepSeek.
## Ключевые файлы и точки входа
- `results/main_ru.jsonl`, `cot_ru.jsonl` (71 строка: 1 строка deepseek A7 с error), `followup_doubt_ru.jsonl`, `d_exec.jsonl`, `run_manifest.json`.
- Логи запусков (не в git): `flow/logs/step08_*.log`.
## Решения и допущения
- Прогоны с зависшим DeepSeek дважды останавливал (pkill). Записи журнала целые (проверено json-разбором). Прерванные попытки без ответа в журнал не попали: это не ответы модели. Готовые ответы при повторе брались из кэша.
- cot и followup остальных моделей прогнаны отдельно (`--models gigachat-2-max,yandexgpt-pro-5.1,qwen3-8b`). `write_results` сливает строки, поэтому DeepSeek допишется в те же таблицы.
- Вызовы DeepSeek для H6/H9 будут получены в другое время, чем остальные (main у всех получен 15.10 00:39–00:43). Это стоит упомянуть в ограничениях. Даты есть в журнале и манифесте.
- Тип E (шаг 09) здесь не запускался.
## Проверка
- `.venv/bin/python -m pytest -q`: 592 passed.
- `src.manifest`: main 152/152 ok; followup 62/62; cot ок 70, «без ответа 3» (A7, A8, B1 deepseek). Ещё не запрашивались deepseek cot B2–B8 и весь deepseek followup.
- Журнал только дописывался: 147 → 471 строка, все строки — валидный JSON. `.env` и `demo/` не в коммите.
## Что нужно для завершения
- DeepSeek снова отвечает: `.venv/bin/python -m src.smoke_test` → строка deepseek `ok`.
- Затем:
  1. `.venv/bin/python -m src.run_extra selected --models deepseek-v4-pro` (код 3 → повторить позже). Нужно 10 cot (A7, A8, B1–B8) и followup на всех верных ответах deepseek в main;
  2. `.venv/bin/python -m src.manifest` → «без ответа 0»;
  3. коммит `step 08: DeepSeek cot/followup` с logs/ и results/.
- main, exec_code и прогоны остальных моделей повторять не нужно: всё возьмётся из кэша.
## Открытые вопросы и риски
- Если DeepSeek долго недоступен, вариант на решение человека: H6/H9 без DeepSeek (3 модели) с пометкой в отчёте. Замена модели после коммита гипотез нежелательна.
- Зависшие запросы DeepSeek могли быть списаны, хотя ответа нет (пункт в HUMAN_TODO).
- gigachat parse_ok 35/38 в main: для шага 10 (parser_spotcheck) такие ответы смотреть в первую очередь.
- Повтор запроса после таймаута 300 с — новый запрос с тем же промптом. Это не «перезапрос ради лучшего ответа»: ответа не было, и это видно в журнале (error, attempt).
## Для следующих шагов
- Шаг 09 (тип E): DeepSeek — 600 вызовов. Сначала smoke. При зависаниях `_with_deadline` даст TimeoutError через 300 с, `with_retries` в run_experiments.sh повторит. Можно запустить E для остальных моделей отдельно через `--models`.
- Шаг 10/11: d_exec содержит только experiment=main (paraphrase нет: H8 не выбрана). В cot у qwen3-8b-think вариант base (режим рассуждений), `reasoning_text` в журнале.
- Шаг 11: H1 — пары из main_ru (`bank.pair_groups`); H6 — main vs cot (A, B), токены cot у GigaChat = tokens_in + tokens_cached; H9 — followup_doubt_ru (first_correct=True, correct второго хода).
