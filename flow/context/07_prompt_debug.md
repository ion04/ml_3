STATUS: deferred
## Сделано
- Попытка 2 (2026-10-10): ключей по-прежнему нет. В `.env` заполнены только GIGACHAT_SCOPE, GIGACHAT_VERIFY_SSL_CERTS и OLLAMA_BASE_URL. Smoke: ответила только qwen3-8b, теперь на ПК gpu (6004 МБ на GPU, сервер localhost:11435), код 1.
- Попытка 1: dev-прогоны qwen3-8b на ноутбуке, промпт v1, experiment=dev, 28 вызовов. Таблицы: `results/dev_ru`, `dev_cot_ru`, `dev_followup_doubt_ru`, `dev_typeE_llm`. Кроме того, `src/prompt_debug.py`, `docs/prompt_debug.md`, поддержка удалённого Ollama (`OLLAMA_BASE_URL`), `src/server_info.py`.
- Попытка 2:
  - `run_bank --experiment dev --models qwen3-8b` берёт все 5 ответов из кэша, новых вызовов нет;
  - `src/server_info.py` поддерживает Windows-ПК: определяет ОС по `echo %OS%`, запускает PowerShell через `-EncodedCommand` с выводом в UTF-8, версию Ollama и модели берёт по API `OLLAMA_BASE_URL`, умеет `--note`, хост по умолчанию `gpu`;
  - в `results/run_manifest.json` → `ollama_server` записаны железо gpu (Threadripper 3970X, 64 потока, 31,9 ГБ = 4×8 ГБ DDR4-2133, RTX 3090 24 ГБ, драйвер 591.74), Ollama 0.40.2, qwen3:8b 500a1f067a9f и note о сбоящей RAM;
  - новый `scripts/ensure_gpu_ollama.sh`: проверяет Ollama по OLLAMA_BASE_URL, поднимает туннель `gpu_ollama.sh` (ждёт до 30 с), при недоступном ПК выходит с кодом 1. Вызывается шагом 0 в `scripts/run_experiments.sh` (не в demo);
  - README, `docs/prompt_debug.md` (где считались dev-ответы qwen), HUMAN_TODO (пункты про домашний сервер закрыты решением «gpu»).
- Промпт не менял и не замораживал.
## Ключевые файлы и точки входа
- `.venv/bin/python -m src.prompt_debug [--write]`, `... -m src.prompt_debug freeze [--force]`.
- `scripts/ensure_gpu_ollama.sh` (код 0 = Ollama на gpu отвечает).
- `.venv/bin/python -m src.server_info [--host gpu|home] [--write] [--note ...]`.
- `prompt_debug.build(log)`, `freeze_problems(checked)`, `write_doc(text)`. `server_info.collect/ollama_api_info/write_manifest`.
## Решения и допущения
- Промпт v1 не подгоняется под одну модель, поэтому `prompt_version`/`prompt_hash` в config = null.
- Метрика «формат» = parse_ok + есть УВЕРЕННОСТЬ + нет finish_reason=length + у кода ровно один блок python.
- qwen3-8b base: parse_ok 5/5, формат 4/5 (на dev-D1 код стоит после «ОТВЕТ:» и повторён двумя блоками). Кандидат на v2 — правка FORMAT["code"], но только если так же ведут себя другие модели.
- Ключ кэша не зависит от сервера Ollama: dev-ответы qwen, полученные на ноутбуке, переиспользуются. Для отладки формата это допустимо, записано в `docs/prompt_debug.md`.
- Если `OLLAMA_BASE_URL` пуст, `ensure_gpu_ollama.sh` останавливает реальный прогон: на ноутбуке не считаем (HUMAN_NOTES).
## Проверка
- `.venv/bin/python -m pytest -q`: 420 passed (+2 теста server_info: Windows и Linux через подменённый ssh).
- `src.server_info` на реальном gpu: все поля заполнены. `scripts/ensure_gpu_ollama.sh`: «Ollama отвечает … 0.40.2», код 0.
- `src.smoke_test`: 1 из 4 (нет GIGACHAT_CREDENTIALS, YANDEX_API_KEY/FOLDER_ID, DEEPSEEK_API_KEY). `prompt_debug freeze` по-прежнему код 1.
## Что нужно для завершения
- В `.env`: GIGACHAT_CREDENTIALS (и решение по TLS), YANDEX_API_KEY + YANDEX_FOLDER_ID, DEEPSEEK_API_KEY. Smoke 4/4 (HUMAN_TODO G1).
- ПК gpu включён и доступен по ssh.
- Затем повторить шаг 07:
  1. `scripts/ensure_gpu_ollama.sh`, `src.smoke_test`;
  2. `src.run_bank --experiment dev`, `src.run_extra cot followup_doubt --dev` (если H6/H9 выбраны, иначе тоже можно), `src.run_classifier --dev`, `src.prompt_debug --write`;
  3. если формат не соблюдается у нескольких моделей, до 3 итераций v2/v3 на всех моделях с записью в журнал итераций;
  4. `src.prompt_debug freeze`, коммит `step 07: freeze prompt <v>`.
## Открытые вопросы и риски
- RAM на gpu сбоит. Падение ssh-сессии останавливает Ollama, `gpu_ollama.sh` переподключается. Ответы с ошибкой соединения ретраятся или дозапрашиваются, в журнале они видны как error.
- Туннель живёт, пока жив процесс `gpu_ollama.sh` на ноутбуке (nohup). После перезагрузки ноутбука его поднимает `ensure_gpu_ollama.sh`.
- Гипотезы не выбраны: cot/followup проверены на всякий случай (2–3 задачи, только qwen).
- В `logs/calls.jsonl` дописана запись smoke от человека (22:58 UTC, gpu) и моя (smoke, gpu). Обе закоммичены как есть.
## Для следующих шагов
- Шаги 08/09: в начале `scripts/ensure_gpu_ollama.sh`. Перед прогоном можно обновить `src.server_info --write --note "..."` (ключ `ollama_server` перезаписывается целиком, note передать снова).
- Память локальной модели берётся из `local_model_size_mb`/`local_vram_mb` (на удалённом сервере `local_mem_mb=None`).
- Токены E на вызов (qwen, по 0/5/20 примеров) — в таблице `docs/prompt_debug.md`, для `--estimate`.
