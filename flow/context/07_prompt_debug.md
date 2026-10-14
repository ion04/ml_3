STATUS: done
## Сделано
- Попытка 3 (2026-10-14): ключи есть. `scripts/ensure_gpu_ollama.sh` → код 0 (gpu, :11435), `src.smoke_test` → 4/4, код 0.
- Dev-прогоны, промпт v1, experiment=dev, только `data/tasks_dev.jsonl` и строки пула E (банк `data/tasks.jsonl` не трогали):
  - `run_bank --experiment dev`: 4 модели × 5 задач (у qwen3-8b ответы из кэша, получены 10.10);
  - `run_extra cot followup_doubt --dev`: cot 2 задачи × 4 модели, followup 2–3 задачи на модель;
  - `run_classifier --dev`: 6 строк × 0/5/20 примеров × 4 модели.
  Новых вызовов в журнале 86, техошибок 0.
- Итог (таблицы в `docs/prompt_debug.md`, собраны `src.prompt_debug`):
  - base: parse_ok 5/5 у всех моделей; формат 5/5 у gigachat, yandex и deepseek, у qwen 4/5 (dev-D1: два блока кода);
  - cot и followup: всё в формате;
  - E: 71 из 72 ответов в формате; у gigachat 1 ответ без «ОТВЕТ:», значение взято эвристикой с parse_ok=False.
- **Промпт v1 заморожен без итераций**: `config/experiment.json` → `prompt_version: "v1"`, `prompt_hash: "d946bcbce2dd"`.
- `src/llm.py`: в журнал добавлено поле `tokens_cached` (GigaChat `precached_prompt_tokens`, DeepSeek `prompt_cache_hit_tokens` / OpenAI `cached_tokens`), функция `_openai_cached_tokens`. Тест `test_cached_tokens_logged`.
- `tests/test_runners.py::test_eval_problems_require_prompt_hash` явно обнуляет prompt_hash: тест полагался на null в config.
- `docs/prompt_debug.md`: журнал итераций (v1 заморожен, на каких моделях проверен), наблюдения, автотаблицы. HUMAN_TODO: G1.6 «принять заморозку v1», G1.7 бюджет E, G2 тарифы (кэш GigaChat). Память: факт о кэше GigaChat.
## Ключевые файлы и точки входа
- `.venv/bin/python -m src.prompt_debug [--write | freeze [--force]]`; `docs/prompt_debug.md` (раздел AUTO руками не править).
- `config/experiment.json` → `prompt_version`/`prompt_hash`; `run_bank.prompt_problems(cfg)` сверяет их с `prompts.prompt_hash()`.
- Таблицы dev: `results/dev_ru.jsonl`, `dev_cot_ru.jsonl`, `dev_followup_doubt_ru.jsonl`, `dev_typeE_llm.jsonl`.
## Решения и допущения
- Оба сбоя формата встречаются каждый у одной модели (qwen D1: два блока кода; gigachat E: нет метки ОТВЕТ). Правка промпта под них была бы подгонкой под одну модель (вопрос № 10), поэтому v2 не делали. Парсер эти случаи уже обрабатывает (заметка «блоков кода 2»; эвристика + parse_ok=False).
- Ответы qwen на dev получены 10.10 на ноутбуке и взяты из кэша (ключ кэша от сервера не зависит). Для отладки формата это допустимо, записано в docs.
- Гипотезы не выбраны: cot и followup проверили на всех моделях на всякий случай (это дёшево).
- Заморозка обратима: оценочных прогонов ещё нет. Если человек поправит промпт (G1.6), нужно поднять PROMPT_VERSION, снова прогнать dev на всех моделях и выполнить `prompt_debug freeze`.
- GigaChat: `prompt_tokens` не включает токены из кэша префикса, и они не тарифицируются (developers.sber.ru, «Подсчет токенов»). Поэтому у повторных few-shot вызовов tokens_in ≈ 80 при промпте 1300+. tokens_in оставлен как есть: это тарифицируемый вход. Полный вход = tokens_in + tokens_cached. У DeepSeek кэш уже входит в tokens_in.
## Проверка
- `.venv/bin/python -m pytest -q`: 579 passed.
- `src.prompt_debug freeze`: «Заморожено: prompt_version=v1, prompt_hash=d946bcbce2dd».
- `src.run_bank` (main) без гипотез: «не запущен», в причинах только гипотезы, на промпт жалоб нет. Вызовов нет.
- В `logs/calls.jsonl` новые записи только experiment=smoke и dev. Журнал только дописывался.
## Что нужно для завершения
- Ничего: шаг выполнен. Подтверждение заморозки — HUMAN_TODO G1.6 (пункт шага 07).
## Открытые вопросы и риски
- Dev-задач 5: редкие сбои формата на оценочном банке возможны (код D несколькими блоками, E без метки). Их ловят parse_ok=False и заметки. Шаг 10 (parser_spotcheck) берёт их в первую очередь.
- Записи до 14.10 (в том числе dev qwen и gigachat E) поля `tokens_cached` не имеют.
- Тип E без примеров: GigaChat и DeepSeek рассуждают (выход 200–380 токенов), это влияет на бюджет шага 09.
## Для следующих шагов
- Шаги 08/09: промпт заморожен, менять `src/prompts.py`, `typeE.fmt_*` и разбиение нельзя (иначе гейт prompt_hash). Оценочные прогоны по-прежнему ждут гипотез (G1.4) и `validate --final` (G1.5).
- Шаг 10/11: `analysis.cost_rub` по tokens_in корректен для GigaChat (кэш не тарифицируется), для DeepSeek даёт верхнюю оценку. Для сравнения **размера** промпта (например, токены CoT/base в H6) у GigaChat брать tokens_in + tokens_cached.
- Токены E по моделям и числу примеров (для `run_classifier --estimate`) — таблица в `docs/prompt_debug.md`. Для GigaChat вход там занижен кэшем.
