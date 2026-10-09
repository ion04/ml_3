STATUS: deferred
## Сделано
- Smoke 2026-10-10: ответила только qwen3-8b (на ноутбуке); у GigaChat, Yandex и DeepSeek нет ключей в `.env` (`.env` нет).
- Dev-прогоны на qwen3-8b, промпт v1, experiment=dev, всего 28 вызовов:
  - `run_bank --experiment dev`: 5 вызовов;
  - `run_extra cot followup_doubt --dev`: 2 + 3;
  - `run_classifier --dev`: 6 строк × 0/5/20 = 18.
  Таблицы `results/dev_ru.jsonl`, `dev_cot_ru`, `dev_followup_doubt_ru`, `dev_typeE_llm`.
- `src/prompt_debug.py`: по журналу (experiment=dev) заново разбирает ответы и строит таблицы «модель × версия» и «все режимы»: parse_ok, формат, хвост после УВЕРЕННОСТЬ, обрезка, токены; отдельно токены E по числу примеров. `--write` обновляет авто-раздел `docs/prompt_debug.md`. `freeze` записывает prompt_version + prompt_hash в config, но только если у всех включённых моделей есть base-ответы текущей версии на 5 dev-задач (`--force` — обход).
- `docs/prompt_debug.md`: журнал итераций (v1, только qwen), наблюдения, авто-таблицы.
- По заметке человека (появилась в HUMAN_NOTES во время шага) добавлена поддержка удалённого Ollama:
  - `llm.load_models` берёт `OLLAMA_BASE_URL` для моделей ollama (хвост /v1 отрезается);
  - при заданной переменной psutil не используется (`local_mem_mb=None`);
  - в журнал пишутся `ollama_base_url` и `local_model_size_mb` (из /api/ps);
  - `scripts/setup_home_server.sh`, `src/server_info.py` (железо по ssh → `results/run_manifest.json` → `ollama_server`), `.env.example`, README.
- Тесты: `tests/test_prompt_debug.py`, `tests/test_server_info.py`, удалённый Ollama в `tests/test_llm.py`.
## Ключевые файлы и точки входа
- `.venv/bin/python -m src.prompt_debug [--write]`, `... -m src.prompt_debug freeze [--force]`.
- `prompt_debug.build(log) -> (checked, markdown)`, `freeze_problems(checked)`, `write_doc(text)` (между маркерами AUTO).
- `llm.ollama_base_url(default)`, `llm.ollama_is_remote()`; `server_info.collect(host)`, `write_manifest(info)` (обновляет только ключ `ollama_server`).
## Решения и допущения
- Промпт не менял и не замораживал: доступна одна модель, а подгонка под одну запрещена. `PROMPT_VERSION=v1`, `prompt_version`/`prompt_hash` в config = null.
- Метрика «формат»: parse_ok + есть УВЕРЕННОСТЬ + нет finish_reason=length + у кода ровно один блок python (так требует FORMAT["code"]).
- qwen3-8b, base: parse_ok 5/5, формат 4/5. На dev-D1 код стоит после «ОТВЕТ:» и повторён двумя блоками; `extract_code` взял длинный. cot, followup и E — всё в формате. Кандидат на v2 — уточнить FORMAT["code"], но только если так же ведут себя другие модели.
- Удалённый Ollama считается удалённым, если задана `OLLAMA_BASE_URL` (при SSH-туннеле адрес — localhost, но модель на сервере).
- Сервер не настраивал: для sudo нужен пароль. Скрипт и пункт G1 оставлены человеку.
## Проверка
- `.venv/bin/python -m pytest -q`: 418 passed (было 409).
- `src.prompt_debug freeze`: код 1, перечислены 3 модели без dev-ответов. Config не изменён.
- `src.server_info` на реальном `home`: i5-6500, 4 потока, 11,6 ГБ, GPU нет, Ollama 0.1.41, llama3. В manifest не записано: сервер ещё не готов.
## Что нужно для завершения
- Ключи GigaChat, Yandex (key + folder_id) и DeepSeek в `.env`, smoke-тест проходит (G1 в HUMAN_TODO).
- Решение по серверу для qwen (пункт G1 «Домашний сервер»). Dev-ответы qwen уже в кэше журнала: повторно не запрашиваются, если не меняется промпт.
- Повторный шаг 07: `run_bank --experiment dev`, `run_extra cot followup_doubt --dev` (если в гипотезах H6/H9), `run_classifier --dev`, `prompt_debug --write`; при сбоях формата до 3 итераций (v2, v3) для всех моделей с записью в журнал итераций; затем `prompt_debug freeze` и коммит `step 07: freeze prompt <v>`.
## Открытые вопросы и риски
- Заметка человека («на ноутбуке 1–5 запросов») появилась в 22:29 UTC, во время dev-прогонов (22:22–22:31). На ноутбуке сделано 28 dev-вызовов qwen. Полных прогонов не было.
- Сервер слабее ноутбука для LLM: только CPU. На ноутбуке dev ≈ 19,5 ток/с. Тип E (2400 вызовов, у qwen выход 200–670 токенов) на CPU может идти очень долго. Решает человек.
- Dev-ответы qwen получены на ноутбуке (Metal), основной прогон, возможно, будет на сервере (CPU): числа могут отличаться при том же seed. Для отладки формата это не важно; в отчёте указать, где шёл основной прогон.
- Гипотезы не выбраны: cot/followup проверены на всякий случай (по 2–3 задачи).
## Для следующих шагов
- Шаг 08/09: локальную модель не гонять полностью на ноутбуке (HUMAN_NOTES). Перед прогоном `src.server_info --write`; манифест дополнять, не перезаписывая ключ `ollama_server`.
- Записи журнала ollama теперь содержат `ollama_base_url` и `local_model_size_mb`; при удалённом сервере `local_mem_mb=None`. Для памяти брать `local_model_size_mb`/`local_vram_mb`.
- Токены E на вызов (qwen, вход/выход по 0/5/20 примеров) — таблица в `docs/prompt_debug.md`, для `--estimate`/бюджета.
