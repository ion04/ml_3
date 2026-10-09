# Цель
Единый клиент к четырём LLM с полным логированием (шаги 1–2 хода работы из PDF). Все последующие шаги вызывают модели только через него.

# Контракты (на них опираются следующие шаги и проверки гейтов)
- `config/models.json` — список моделей:
  `{"key", "role" (ru1|ru2|big|local), "provider" (gigachat|openai_compat|ollama|mock), "model" (точный id), "base_url", "api_key_env", "min_temperature", "params", "enabled"}`.
  Кандидаты по умолчанию (роли из таблицы в PDF):
  - ru1: GigaChat-2-Max, при желании ещё и GigaChat-2 (Lite);
  - ru2: YandexGPT Pro через OpenAI-совместимый endpoint Yandex AI Studio, модель `gpt://<folder_id>/yandexgpt/latest`;
  - big: DeepSeek-V3 (`deepseek-chat`);
  - local: `qwen3:8b` в Ollama, режим рассуждений выключен.
  **Сверь актуальные endpoints, имена моделей и ограничения по температуре с официальной документацией на сегодня** (WebSearch/WebFetch). Найденные отличия опиши в handoff, устойчивые особенности сохрани в память.
- `src/llm.py`:
  `ask(model_key, prompt, temperature=0.0, *, system=None, history=None, max_tokens=None, meta=None, repeat=0, use_cache=True, log_path=None) -> LLMResult`.
  - Поля `LLMResult`: `text, tokens_in, tokens_out, latency_s, model_reported, effective_temperature, cached, call_id, error`.
  - `history` — список `{"role", "content"}` для многоходовых диалогов (нужен для гипотезы H9).
  - Функция `ask` — та самая `ask(model, prompt, temperature) → (ответ, токены, время)` из задания.
- Логирование. Каждый вызов, включая ошибки, **сразу** дописывается в `logs/calls.jsonl` (append, flush + fsync).
  - Поля записи: `call_id, ts_utc, model_key, provider, model_requested, model_reported, temperature_requested, effective_temperature, max_tokens, system, history, prompt, response_text, tokens_in, tokens_out, latency_s, error, attempt, meta (task_id, variant, experiment, repeat, lang), cache_key, local_mem_mb`.
  - Для провайдера mock лог по умолчанию пишется в `demo/calls.jsonl`.
- Кэш. Ключ = хэш от `model_key, model_requested, system, history, prompt, temperature, max_tokens, repeat`. Если в логе уже есть успешная запись с таким ключом, она возвращается без вызова API. Так повторный запуск не тратит лимит и не меняет ответы.
- Ретраи с backoff только для технических ошибок: сеть, 429, 5xx, таймаут. Каждая попытка видна в логе.
- Температура. Если провайдер не принимает 0, используется `min_temperature` из конфига. Фактическое значение пишется в `effective_temperature`.
- Локальная модель: в `local_mem_mb` записывается память процесса Ollama (psutil, максимум RSS за время вызова).
- Провайдер `mock` — детерминированные ответы по шаблону. Нужен для тестов всех следующих шагов.
- `src/smoke_test.py` (`.venv/bin/python -m src.smoke_test`):
  - отправляет один короткий запрос каждой включённой модели, кроме mock;
  - печатает таблицу (модель, model_reported, токены, задержка, ok/ошибка без секретов);
  - код выхода 0, только если ответили все;
  - meta: `experiment="smoke"`.

# Проверка
- `tests/test_llm.py` на mock: запись в лог, кэш, ретраи (сымитировать 429), многоходовый history, отсутствие ключей в логе.
- Если ключи в `.env` уже есть, запусти smoke-тест и приложи результат к handoff. Если ключей нет, шаг всё равно `done`: код готов и протестирован на mock. Пункты про ключи уже есть в HUMAN_TODO G1.
- Ни один тест не должен ходить в реальные API.
