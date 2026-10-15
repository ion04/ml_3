STATUS: deferred
## Сделано
- 15.10: G1 пройден, промпт заморожен (v1, d946bcbce2dd), гипотезы закоммичены 14.10 (67d7108) — до первого вызова E. Туннель gpu: `scripts/ensure_gpu_ollama.sh` → 0.
- Smoke: GigaChat и Yandex ok; **DeepSeek снова не отвечает** (03:08, 03:13, 03:51 UTC — TimeoutError 300 с, записи smoke в журнале). Пробы останавливал вручную, чтобы не ждать 5×300 с.
- Пробный прогон E (`--limit 20`, gigachat-2-max, yandexgpt-pro-5.1, qwen3-8b; 0/5/20): 180 вызовов, ошибок 0; `typeE_metrics check --limit 20` → код 0 (parse_ok ≥ 95% во всех группах). Формат не ломается — к шагу 07 не возвращаемся.
- Полный прогон 200 строк × 0/5/20 для тех же 3 моделей: 1800 строк (1620 новых + 180 из кэша), техошибок 0, parse_ok 1790/1800 (gigachat 5-shot 190/200, остальные 200/200), finish_reason=length 0, «нельзя определить» 1 (gigachat 0-shot). Вызовы 15.10 03:19–03:51 UTC.
- `results/typeE_metrics.csv`: LLM 9 строк (все complete=True), классических 11. `src.manifest` пересобран.
- HUMAN_TODO: G1.7 пункт «коммит гипотез перед E» закрыт (вопрос снят); G1.8 новый пункт «Дозапросить DeepSeek для типа E».
## Ключевые файлы и точки входа
- `results/typeE_llm.jsonl` (1800 строк), `results/typeE_metrics.csv`, `results/run_manifest.json`, `logs/calls.jsonl` (2282 строки).
- Логи запусков (не в git): `flow/logs/step09_trial.log`, `step09_full.log`, `step09_deepseek_probe.log`.
- Команды: `src.run_classifier --models <ключи> --shots 0,5,20 [--limit 20]`, `src.typeE_metrics [check --limit N]`.
## Решения и допущения
- E запущен для трёх отвечающих моделей через `--models`, без DeepSeek (иначе 600 вызовов × до 5 попыток по 300 с). `write_results` сливает строки: DeepSeek допишется в те же таблицы, остальное возьмётся из кэша.
- Мой ошибочный запуск `src.smoke_test --help` (флага нет, тест запустился) пересёкся с GigaChat-прогоном E в 1 потоке: 3 записи smoke gigachat с HTTP 429 (ретраи, видно в журнале), затем ok. На прогон E 429 не повлияли (ошибок 0).
- qwen3-8b рассуждает в видимом тексте перед «ОТВЕТ:» (вых. токенов в среднем 190–550, все finish=stop) — это не режим think; при сравнении токенов/задержки учитывать.
- Метрики и выводы в handoff не пишу (правило CLAUDE.md); числа — только из CSV/ноутбука.
## Проверка
- `.venv/bin/python -m pytest -q`: 592 passed.
- `src.typeE_metrics check` (полный): «Формат в норме», 9 групп × 200 строк.
- `src.manifest`: typeE 1800 записей, ок 1800, ошибок 0. «Без ответа 3» — это старые deepseek cot шага 08.
- Журнал только дописывался: `git diff` по logs/calls.jsonl — 0 удалённых строк, все строки JSON. `.env`, `demo/` не в коммите.
## Что нужно для завершения
- DeepSeek отвечает: smoke-строка deepseek `ok` (`.venv/bin/python -m src.smoke_test`; у скрипта нет `--help`).
- Затем:
  1. `.venv/bin/python -m src.run_classifier --models deepseek-v4-pro --shots 0,5,20 --limit 20` → `.venv/bin/python -m src.typeE_metrics check --limit 20` (код 4 → blocked, к шагу 07);
  2. то же без `--limit` (код 3 → повтор);
  3. `.venv/bin/python -m src.typeE_metrics`, `.venv/bin/python -m src.manifest`; ожидается 12 строк LLM, все complete=True;
  4. коммит `step 09: DeepSeek тип E` с logs/ и results/.
- Остальные модели повторять не нужно (кэш). Альтернатива при долгой недоступности — решение человека (HUMAN_TODO G1.8): E на трёх моделях с пометкой в ограничениях.
## Открытые вопросы и риски
- Если DeepSeek вернётся позже, его ответы E будут получены в другую дату — указать в ограничениях (даты в журнале/манифесте).
- Возможное списание DeepSeek за зависшие запросы (пункт шага 08 в «Добавлено агентами»).
- 10 ответов gigachat 5-shot с parse_ok=False (метка взята эвристикой) — в parser_spotcheck шага 10 в первую очередь.
- У GigaChat tokens_in занижен кэшем префикса (полный вход = tokens_in + tokens_cached), особенно в 5/20-shot.
## Для следующих шагов
- Шаг 10: parser_spotcheck — включить строки typeE с parse_ok=False (gigachat) и одну «нельзя определить».
- Шаг 11: `typeE_metrics.build(results/typeE_llm.jsonl)`; пока группы только для 3 моделей — ноутбук не должен падать без deepseek. H12 (если понадобится): LLM k-shot vs LogReg fixed k-shot, Макнемар по row_idx.
