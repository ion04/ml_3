# ЛР3. Можно ли доверять LLM в задачах классического ML

Учебный проект по курсу «Методы машинного обучения». Задание: `ЛР LLM и AutoML.pdf`, стр. 1–9 (ЛР4 не выполняется).

- План работы: [PLAN.md](PLAN.md)
- Что осталось сделать человеку: [HUMAN_TODO.md](HUMAN_TODO.md)
- Агентский флоу: [flow/README.md](flow/README.md)
- Правила проекта: [CLAUDE.md](CLAUDE.md)

## Установка окружения

Версия Python: **3.12.14**. Системный Python 3.14 слишком новый для части зависимостей (gigachat, statsmodels, ucimlrepo), поэтому окружение создаётся на 3.12.

```bash
# Вариант 1: через uv (так создано исходное окружение)
uv venv --seed --python 3.12 .venv
.venv/bin/python -m pip install -r requirements.txt

# Вариант 2: без uv (нужен python3.12, например brew install python@3.12)
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Проверка
.venv/bin/python -c "import sklearn, statsmodels, openai, gigachat, ucimlrepo; print('ok')"
```

Ключи API:
```bash
cp .env.example .env   # затем вписать значения; комментарии в файле объясняют, где их взять
```
Файл `.env` не коммитится.

Локальная модель:
```bash
brew install ollama
brew services start ollama
ollama pull qwen3:8b
```

Модули запускаются из корня проекта так: `.venv/bin/python -m src.<module>`. Тесты запускаются командой `.venv/bin/python -m pytest`.

## Структура каталогов

| Путь | Что лежит |
|------|-----------|
| `data/` | банк задач (`tasks.jsonl`, `tasks_dev.jsonl`), датасет AI4I 2020 для типа E, разбиение |
| `src/` | код: клиент LLM, промпты, парсер, прогонщики, подготовка проверки |
| `config/` | модели (`models.json`), настройки эксперимента и гипотезы (`experiment.json`), тарифы |
| `logs/` | `calls.jsonl`, все вызовы LLM с сырыми ответами. Файл только дописывается |
| `results/` | производные результаты: автопроверка, метрики, таблицы |
| `grading/` | листы слепой ручной проверки и ключи деанонимизации |
| `report/`, `report/figures/` | отчёт и рисунки |
| `tests/` | тесты pytest (на mock-провайдере, без ключей) |
| `docs/` | вспомогательные материалы (например, справка по гипотезам) |
| `scripts/` | shell-скрипты запуска прогонов |
| `demo/` | результаты на mock и синтетике. **Не коммитится** и не является результатами эксперимента |
| `flow/` | агентский флоу: задания шагов, handoff, оркестратор |

## Как запустить эксперимент

> Раздел дополняется по шагам флоу; пункты с пометкой _(шаг NN)_ ещё не готовы.

1. Установить окружение и заполнить `.env` (см. выше).
2. Smoke-тест моделей (по одному короткому запросу к каждой включённой модели из `config/models.json`; код выхода 0, только если ответили все):
   ```bash
   .venv/bin/python -m src.smoke_test
   ```
   Локальная модель считается на ПК `gpu` (Windows, RTX 3090): `scripts/gpu_ollama.sh` запускает там Ollama и туннель `localhost:11435`, в `.env` задаётся `OLLAMA_BASE_URL=http://localhost:11435`. Проверка или запуск туннеля — `scripts/ensure_gpu_ollama.sh` (его вызывает `scripts/run_experiments.sh`). Железо сервера для отчёта — `.venv/bin/python -m src.server_info --write [--host gpu|home] [--note "..."]` (→ `results/run_manifest.json` → `ollama_server`). Запасной вариант — домашний сервер (`scripts/setup_home_server.sh`, `--host home`).
   Все вызовы LLM идут через `src/llm.py::ask` и сразу дописываются в `logs/calls.jsonl` (вызовы mock-моделей пишутся в `demo/calls.jsonl`). Повторный запуск с теми же входами берёт ответ из журнала (кэш) и API не вызывает.
3. Банк задач и гипотезы (`data/tasks.jsonl`, 38 задач; `data/tasks_dev.jsonl`, 5 задач для отладки промпта; схема полей описана в docstring `src/bank.py`):
   ```bash
   .venv/bin/python -m src.bank validate          # схема + список незаполненного
   .venv/bin/python -m src.bank validate --final  # строгая проверка перед прогонами
   .venv/bin/python -m src.bank stats             # таблица «тип × тема» для отчёта
   .venv/bin/python -m src.bank refs [--write]    # пересчитать эталоны кодом (data/refs/, data/d_reference/);
                                                  # --write записывает reference_metric задач D
   .venv/bin/python -m src.checks g1              # гейт G1: ключи, smoke, банк, гипотезы закоммичены до прогонов
   .venv/bin/python -m src.checks hypcommit       # хэш коммита с гипотезами (для отчёта)
   ```
   Гипотезы: справка `docs/hypotheses_guide.md`, шаблон `hypotheses.md`, выбор в `config/experiment.json` → `hypotheses`. Чек-лист методологических ошибок для типа D: `data/checklist_D.md`.
4. Промпт, парсер и автопроверка (`src/prompts.py`, `src/parse.py`, `src/grade.py`; проверяются офлайн):
   ```bash
   .venv/bin/python -m src.prompts dev-A1 --variant cot --lang en   # показать system и prompt задачи
   .venv/bin/python -m pytest tests/test_parse.py tests/test_grade.py tests/test_prompts.py -q
   ```
   Формат ответа: последние строки `ОТВЕТ: …` и `УВЕРЕННОСТЬ: 0–100` (`ANSWER:` / `CONFIDENCE:` для EN). Варианты промпта: `base`, `cot` (H6), `paraphrase` (H8), `formula`/`verbal` (H2), `followup_doubt` (H9), `classify_row` (тип E). Автоматически проверяются A, B (вердикт), E и F; C, D и обоснование B оценивают люди вслепую.

   Отладка и заморозка промпта (шаг 07; только dev-задачи, на всех включённых моделях, журнал итераций — `docs/prompt_debug.md`):
   ```bash
   .venv/bin/python -m src.run_bank --experiment dev && .venv/bin/python -m src.run_extra cot followup_doubt --dev
   .venv/bin/python -m src.run_classifier --dev
   .venv/bin/python -m src.prompt_debug --write   # parse_ok и соблюдение формата модель × версия (по журналу)
   .venv/bin/python -m src.prompt_debug freeze    # prompt_version + prompt_hash → config/experiment.json
   git add -A && git commit -m "step 07: freeze prompt <версия>"
   ```
   `freeze` не сработает, пока у каждой включённой модели нет ответов base-промпта текущей версии на все 5 dev-задач.
5. Основной прогон и прогоны под гипотезы (шаг 05; реальные прогоны — шаг 08, только после `src.checks g1` = 0):
   ```bash
   scripts/run_experiments.sh            # всё одной командой: smoke → g1 → main → доп. прогоны → код D → ...
   scripts/run_experiments.sh --demo     # то же на mock-моделях, всё пишется в demo/

   .venv/bin/python -m src.run_bank --experiment main --models all --lang ru [--types A,B] [--limit N] [--demo]
   .venv/bin/python -m src.run_bank --experiment dev                 # отладка промпта на data/tasks_dev.jsonl
   .venv/bin/python -m src.run_extra [selected | all | cot selfconsistency followup_doubt paraphrase lang_en verbal_formula] [--demo]
   .venv/bin/python -m src.run_extra cot followup_doubt --dev        # проверка режимов на dev-задачах (experiment=dev)
   .venv/bin/python -m src.exec_code [--experiment main[,paraphrase]] [--demo]   # запуск кода из ответов типа D
   .venv/bin/python -m src.manifest [--demo]                         # паспорт прогонов → results/run_manifest.json
   ```
   - `run_bank` → `results/<experiment>_<lang>.jsonl`: одна строка на ответ (разбор парсером, автооценка, `call_id` для связи с журналом). Оценочный прогон не стартует, пока банк не проходит `validate --final`, гипотезы не закоммичены (те же проверки, что в G1) и промпт не заморожен (`prompt_version` и `prompt_hash` в `config/experiment.json`).
   - `run_extra` без аргументов запускает режимы для гипотез из `config/experiment.json`: H2 `verbal_formula`, H4 `lang_en`, H6 `cot`, H7 `selfconsistency` (+ `selfconsistency_vote_<lang>.jsonl`), H8 `paraphrase`, H9 `followup_doubt` (только задачи, верно решённые в main). H1, H3, H5, H10–H13 считаются из main и типа E. Типы задач для режимов: `config/experiment.json` → `extra_runs`.
   - `exec_code` → код (не исправляется, снимается только общий отступ блока) в `results/d_code/<model>/<task>.py`, запуск во временной папке (таймаут 300 с, окружение без ключей), итоги и pass@1 в `results/d_exec.jsonl`.
   - Повторный запуск не делает новых вызовов (кэш журнала); дозапрашиваются только вызовы с технической ошибкой. Параллельность: GigaChat и Ollama по 1 потоку, OpenAI-совместимые по 3.
   - Код выхода прогонщиков: 0 — всё получено; 1 — прогон не запущен (не пройдены проверки); 2 — нет моделей или неверные аргументы; 3 — остались вызовы с технической ошибкой (сеть, 429, 5xx). На коде 3 `run_experiments.sh` ждёт 60 с и повторяет команду (до 3 запусков); содержательно плохие ответы не перезапрашиваются — они уже в кэше.
   - `src.manifest` пересобирает `results/run_manifest.json` по журналу и таблицам: даты и число вызовов/ошибок по экспериментам, по моделям `model_requested`/`model_reported`/параметры/температуры/токены, версия и отпечаток промпта, параметры эксперимента, гипотезы и хэш их коммита, строки/ошибки/parse_ok каждой таблицы, запуски кода D, версии пакетов и sha256 `requirements.txt`. Раздел `ollama_server` (`src.server_info`) сохраняется.
6. Тип E (AI4I 2020; данные и решения по разбиению — `data/README.md`):
   ```bash
   .venv/bin/python -m src.typeE all            # download → split → classical (вызовов LLM нет)
   .venv/bin/python -m src.typeE show 0         # строка датасета текстом (ru/en), как её видит модель
   .venv/bin/python -m src.run_classifier --estimate                         # объём прогона, без вызовов
   .venv/bin/python -m src.run_classifier --dev                              # отладка промпта на 6 строках пула (шаг 07)
   .venv/bin/python -m src.run_classifier --models all --shots 0,5,20 --limit 20 [--demo]   # пробный прогон
   .venv/bin/python -m src.run_classifier --models all --shots 0,5,20 [--demo]              # полный (шаг 09)
   ```
   - `src.typeE classical` → `results/typeE_classical.csv`: Dummy(most_frequent), LogisticRegression и GaussianNB
     (Pipeline: one-hot для `Type` + StandardScaler) на тех же 5 и 20 примерах, что видит LLM, плюс 20 случайных
     стратифицированных наборов для разброса LogReg/NB. Метрики: F1 класса «отказ», balanced accuracy, accuracy,
     precision/recall, матрица ошибок; `eval_set`: `test` (200 строк), `natural` (все строки вне обучения),
     `full` (константа «нет отказа» на всём датасете — accuracy при естественной доле отказов).
   - `src.run_classifier` → `results/typeE_llm.jsonl` (dev: `results/dev_typeE_llm.jsonl`): одна строка на ответ
     (`shots`, `row_idx`, `y_true`, `value` 1/0, `correct`, `confidence`, `call_id`). Ответ модели: `ОТВЕТ: отказ | нет отказа`.
     Оценочный прогон не стартует, пока гипотезы не закоммичены и промпт не заморожен.
7. Автопроверка и листы слепой проверки: _(шаг 10)_.
8. Анализ: `analysis.ipynb`, Restart & Run All _(шаг 11)_.
