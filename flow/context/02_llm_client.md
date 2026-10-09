STATUS: done
## Сделано
- `src/llm.py`: `ask()` для провайдеров gigachat (SDK), openai_compat (Yandex, DeepSeek), ollama (нативный `/api/chat`) и mock. Каждая попытка сразу дописывается в журнал (flush + fsync). Есть кэш по журналу, ретраи и замер памяти Ollama.
- `config/models.json`: 4 включённые модели, отключённые альтернативы и 4 mock-модели (`mock-ru1/ru2/big/local`, `enabled:false`).
- `src/smoke_test.py`, `tests/test_llm.py` (19 тестов, без сети), команда в README, .env.example (раздел про Ollama).
- Сверка документации на 2026-10-09 (сводка в памяти `llm-provider-api-facts`). Отличия от задания шага:
  - `deepseek-chat` снят с поддержки 2026-07-24. Вместо него взят `deepseek-v4-pro`, рассуждения выключены через `extra_body.thinking=disabled`, иначе temperature игнорируется.
  - Yandex: endpoint `https://ai.api.cloud.yandex.net/v1`, `project=folder_id`, явный URI `gpt://${YANDEX_FOLDER_ID}/yandexgpt-5.1`. `yandexgpt/latest` теперь указывает на Pro 5 и будет отключён.
  - GigaChat: temperature обязана быть > 0, поэтому `min_temperature=0.0001` (при T < 0.001 включается строгий режим). Появилась модель GigaChat-3-Ultra.
## Ключевые файлы и точки входа
- `ask(model_key, prompt, temperature=0.0, *, system, history, max_tokens, meta, repeat, use_cache, log_path) -> LLMResult`.
  - Поля результата: text, tokens_in, tokens_out, latency_s, model_reported, effective_temperature, cached, call_id, error, finish_reason.
  - Ошибки не выбрасываются: при неудаче всех попыток `text=None`, а `error` заполнен.
- Вспомогательные функции: `llm.load_models()`, `llm.get_model(key)`, `llm.missing_env(cfg)` (каких переменных не хватает), `llm.ollama_ps(base_url, model)`.
- Журнал: `logs/calls.jsonl`; для mock по умолчанию `demo/calls.jsonl`. mock с путём внутри `logs/` → ValueError.
- `.venv/bin/python -m src.smoke_test`: код 0, только если ответили все включённые модели, кроме mock.
## Решения и допущения
- Поля записи журнала = контракт + `params`, `reasoning_text`, `finish_reason`, `local_vram_mb` (из `/api/ps`). В `meta` всегда добавляется `repeat`.
- Ключ кэша = sha256(model_key, model_requested, system, history, prompt, **effective** temperature, max_tokens, repeat). meta в ключ не входит.
  - Кэш отдаёт первую успешную запись с ключом. Попадание в кэш в журнал не пишется.
  - Индекс журнала держится в памяти и перечитывается, если файл вырос извне.
- Ретраи: до 5 попыток с паузами 2/4/8/16 с. Повторяются только сеть, таймаут, 429 и 5xx. SSL CERTIFICATE_VERIFY_FAILED не повторяется. Встроенные ретраи SDK выключены (`max_retries=0`).
- `effective_temperature = max(T, min_temperature)`.
- `latency_s` меряется вокруг самого HTTP-запроса. OAuth GigaChat делается до замера, клиент GigaChat кэшируется на модель.
- Ollama: `options.seed = 42 + repeat`, `num_ctx = 8192`. Без явного значения было бы 4096, так Ollama делает при памяти < 24 ГБ.
- `local_mem_mb`: максимум RSS `ollama` вместе с дочерними процессами. Веса держит дочерний `llama-server`.
- Модели без ключа smoke-тест не вызывает и в журнал не пишет: это не вызов модели.
- Выбор ru1 = GigaChat-2-Max и big = deepseek-v4-pro (а не flash) — допущение. Подтверждение вынесено в HUMAN_TODO G1.
## Проверка
- `.venv/bin/python -m pytest -q`: 19 passed. Тесты покрывают журнал, кэш (в том числе после перезапуска процесса), ретраи 429 и отказ от ретраев на 401, history, min_temperature, отсутствие секретов, запрет mock→logs/, разбор ответов openai_compat и ollama на подменённых клиентах.
- Smoke 2026-10-09: запусков два, в `logs/calls.jsonl` 2 записи с experiment=smoke. Последний: qwen3-8b `ok «4»`, tok 33/2, 4.38 с (включая перезагрузку модели), RSS 5326 МБ, 100% GPU (5360 МБ, `ollama ps`: RUNNER llamacpp, CONTEXT 8192). `<think>` в ответе нет.
- Остальные 3 модели: «нет переменных», ключей нет. Код выхода 1, это ожидаемо.
## Что нужно для завершения
- Ничего, шаг done. Ключи и smoke-тест всех моделей — пункты G1 в HUMAN_TODO.
## Открытые вопросы и риски
- Не проверены на реальном API, ключей нет:
  - формат `model` в ответе GigaChat;
  - URI `yandexgpt-5.1`, `project=folder_id` и наличие `usage` в ответе Yandex;
  - `deepseek-v4-pro` с `thinking=disabled`.
  Проверятся первым smoke-тестом с ключами.
- В апстриме бывает, что qwen3 при `think=false` всё равно выдаёт `<think>`. Шаг 04 должен вырезать `<think>…</think>` при разборе (сырой ответ в журнале не трогать).
- GigaChat для физлиц обрабатывает запросы в один поток: прогоны не распараллеливать.
- Тарифы DeepSeek V4-Pro не проверены. Возможно, для экономии на типе E (~2400 вызовов) лучше `deepseek-flash`.
## Для следующих шагов
- Вызывать только `llm.ask`. В прогонах передавать `meta={"task_id", "variant", "experiment", "lang"}` и `repeat`. Для H7 разные `repeat` дают разные ключи кэша.
- Демо-прогоны: модели `mock-ru1/ru2/big/local`. Ответ детерминированный, содержит `ОТВЕТ: <0..99>` и `УВЕРЕННОСТЬ: <50..100>`. Шаблон меняется через `params.template` (поля {model}, {n_messages}, {answer}, {confidence}).
- Реальные модели для прогонов: `[m for m in llm.load_models().values() if m["enabled"] and m["provider"] != "mock"]`.
- H6: включить `qwen3-8b-think` (`think=true`), рассуждения пишутся в `reasoning_text`. H9: передавать `history`.
- В тестах подменять `llm.MODELS_PATH`, `llm._cache_index`, `llm.time.sleep` и передавать `log_path=tmp_path/...` (см. фикстуру в `tests/test_llm.py`).
