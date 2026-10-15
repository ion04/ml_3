STATUS: done
## Сделано
- Попытка 1 (15.10, ночь): `src.checks g1` → 0; гипотезы H1, H6, H9 закоммичены 67d7108 (14.10 22:00 +03), все оценочные вызовы позже. main, cot/followup трёх моделей, exec_code получены; DeepSeek с 00:43 UTC не отвечал (31 запись cot с TimeoutError, все видны в журнале, attempt 1–5). Добавлен `llm._with_deadline` (общий срок ответа openai_compat).
- Попытка 2 (15.10 09:51–09:53 UTC): smoke DeepSeek ok (запись 09:51) → `run_extra selected --models deepseek-v4-pro`: cot 10 новых + 6 из кэша (qwen3-8b-think 16 из кэша), followup_doubt 26 новых. Техошибок 0, parse_ok 36/36, finish_reason=stop 36/36. Код выхода 0.
- Итог по таблицам (только количества):
  - main_ru: 152 строки (38 × 4), ошибок 0, parse_ok 149 (gigachat 35/38, остальные 38/38);
  - cot_ru: 80 строк (5 моделей × 16, A+B), ошибок 0, parse_ok 80;
  - followup_doubt_ru: 88 строк (gigachat 21, yandex 22, qwen3-8b 19, deepseek 26 = число верных A/B/F в main у каждой), ошибок 0, parse_ok 88;
  - d_exec: 24 запуска (6 D × 4 модели), повторный `exec_code` — «к запуску 0».
- `src.manifest` пересобран: main/cot/followup «без ответа 0». `pip freeze` = requirements.txt. HUMAN_TODO G1.8 (шаг 08) закрыт.
## Ключевые файлы и точки входа
- `results/main_ru.jsonl`, `cot_ru.jsonl`, `followup_doubt_ru.jsonl`, `d_exec.jsonl`, `d_code/`, `run_manifest.json`; `logs/calls.jsonl` (2322 строки).
- Логи запусков (не в git): `flow/logs/step08_*.log`, последний — `step08_deepseek_retry.log`.
## Решения и допущения
- Только H6 (cot) и H9 (followup_doubt) требуют отдельных прогонов; H1 считается из main (пары `bank.pair_groups`).
- Повторы DeepSeek — после таймаутов (ответа не было, ошибки в журнале), это не перезапрос ради лучшего ответа.
- Ответы DeepSeek для H6/H9 получены 15.10 ~09:52 UTC, у остальных моделей — 00:43–03:00 UTC; main у всех 00:39–00:43. Даты — в журнале и манифесте; упомянуть в ограничениях (текст человека).
- Тип E для DeepSeek здесь не запускался — это шаг 09.
## Проверка
- `.venv/bin/python -m pytest -q`: 596 passed.
- Сверка: followup по каждой модели = числу верных A/B/F в main; cot 16 строк на модель, все без ошибок.
- `git diff logs/calls.jsonl`: 0 удалённых строк, только дописано (2282 → 2322; 4 smoke + 36 deepseek).
- `.env`, `demo/` не в коммите.
## Что нужно для завершения
- Ничего: шаг выполнен.
## Открытые вопросы и риски
- Возможное списание DeepSeek за 31 зависший запрос — пункт «Добавлено агентами» (шаг 08) в HUMAN_TODO остаётся.
- gigachat parse_ok 35/38 в main — в parser_spotcheck (шаг 10) в первую очередь.
- `results/graded_auto.jsonl` (шаг 10) собран до дозапроса DeepSeek: cot/followup DeepSeek в нём нет — шаг 10 при повторном запуске пересоберёт (`src.autograde`), затем шаги 11–12.
## Для следующих шагов
- Шаг 09: DeepSeek сейчас отвечает (09:51–09:53 UTC) — дозапросить E: `src.run_classifier --models deepseek-v4-pro --shots 0,5,20 --limit 20` → `typeE_metrics check --limit 20` → полный прогон.
- Шаг 10: пересобрать `src.autograde` (новые cot/followup DeepSeek); слепые листы содержат только main и не меняются.
- Шаг 11: H6 — main vs cot (A, B), у GigaChat токены = tokens_in + tokens_cached; у qwen3-8b-think в cot вариант base (режим рассуждений), `reasoning_text` в журнале. H9 — followup_doubt_ru (first_correct=True, correct второго хода).
