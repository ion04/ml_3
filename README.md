# ЛР3. Можно ли доверять LLM в задачах классического ML

Учебный проект по курсу «Методы машинного обучения». Задание: `ЛР LLM и AutoML.pdf`, стр. 1–9 (ЛР4 не выполняется).

- План работы: [PLAN.md](PLAN.md)
- Что осталось сделать человеку: [HUMAN_TODO.md](HUMAN_TODO.md)
- Агентский флоу: [flow/README.md](flow/README.md)
- Правила проекта: [CLAUDE.md](CLAUDE.md)

## С нуля до сдачи

Подробности по каждой команде — в разделе «Как запустить эксперимент» ниже. Что делает только человек (с оценкой времени) — в [HUMAN_TODO.md](HUMAN_TODO.md).
Шаги 5–7 и 9 может выполнить агентский флоу: `python3 flow/flow.py run` перезапускает отложенные шаги, `python3 flow/flow.py status` показывает, где работа.
Порядок важен: **гипотезы и прогноз коммитятся до любых оценочных прогонов** (это проверяют `src.checks g1` и сами прогонщики).

| # | Что сделать | Команда | Готово, если |
|---|-------------|---------|--------------|
| 1 | Окружение | см. «Установка окружения» | `.venv/bin/python -m pytest -q` зелёный |
| 2 | Ключи API | `cp .env.example .env`, вписать ключи (где взять — в комментариях файла) | `git status` не показывает `.env` |
| 3 | Локальная модель на ПК `gpu` | `scripts/ensure_gpu_ollama.sh` (сам поднимет `scripts/gpu_ollama.sh`: Ollama на ПК + туннель `localhost:11435`); в `.env` `OLLAMA_BASE_URL=http://localhost:11435` | «Ollama отвечает», код 0 |
| 4 | Smoke-тест 4 моделей | `.venv/bin/python -m src.smoke_test` | все строки `ok`, код 0 |
| 5 | G1: гипотезы и прогноз (**коммит**), банк задач, решения по промпту и типу E | пункты G1 в HUMAN_TODO | `.venv/bin/python -m src.checks g1` → код 0 |
| 6 | Отладка и заморозка промпта на 5 dev-задачах | п. 4 ниже, `src.prompt_debug freeze` | в `config/experiment.json` заполнены `prompt_version`, `prompt_hash` |
| 7 | Все прогоны: основной, под гипотезы, код D, тип E, автопроверка, слепые листы, паспорт | `scripts/run_experiments.sh` | «=== Готово ===», в `results/` таблицы и `run_manifest.json`, в `grading/` листы |
| 8 | G2: слепая проверка B/C/D двумя людьми, выборка парсера, тарифы | листы в `grading/` (п. 7), `grading/INSTRUCTIONS.md` | `.venv/bin/python -m src.checks g2` → код 0 |
| 9 | Анализ, кандидаты ошибок, каркас отчёта | `scripts/run_analysis.sh` | `results/summary_metrics.json` с `"demo": false`, `report/report.md` без «нет данных» |
| 10 | G3: выводы по гипотезам, 5 ошибок, ограничения, рекомендации | блоки `<!-- ЧЕЛОВЕК:… -->` в `report/report.md`, ячейки «Вывод» в `analysis.ipynb` | `.venv/bin/python -m src.build_report check` → код 0 |
| 11 | Вопросы самоконтроля и доклад на 5 минут | `report/selfcheck_questions.md`, `report/slides_outline.md` | ответы под всеми 12 вопросами, доклад укладывается в 5 мин |
| 12 | Проверка воспроизводимости в чистом окружении | `scripts/check_reproducibility.sh` | «Итог: совпадает», запись в `docs/reproducibility.md` |
| 13 | Сдача | `git add -A && git commit`; если нужен docx/PDF — `brew install pandoc`, затем `pandoc report/report.md -o report.docx --resource-path=report` | `git status` чист, `.env` и `demo/` не в коммите, в сданном файле видны рисунки |

Весь конвейер без ключей на mock-моделях (всё пишется в `demo/`, это не результаты):
`scripts/run_experiments.sh --demo && scripts/run_analysis.sh --demo`.

Для защиты: карта «требование задания → код» — [docs/requirements_traceability.md](docs/requirements_traceability.md); итоги аудита — [docs/qa_report.md](docs/qa_report.md); независимый пересчёт эталонов — `.venv/bin/python docs/audit/recompute_refs.py`.

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

Локальная модель (`qwen3:8b`) для прогонов считается на ПК `gpu` (Windows 10, RTX 3090, доступ `ssh gpu`), а не на ноутбуке:
```bash
# на ПК: портативный Ollama 0.40.2 в C:\Users\616\ollama, модель скачана (ollama pull qwen3:8b)
scripts/ensure_gpu_ollama.sh                       # проверка; если туннеля нет — сам запускает scripts/gpu_ollama.sh
nohup scripts/gpu_ollama.sh >> flow/logs/gpu_ollama.log 2>&1 &   # то же вручную: Ollama на ПК + туннель localhost:11435
curl -s localhost:11435/api/version                # должен ответить версией
.venv/bin/python -m src.server_info --write --note "..."         # железо ПК → results/run_manifest.json (ollama_server)
```
Если ПК недоступен, прогоны локальной модели откладываются (на ноутбуке допустимы только короткие проверки на 1–5 запросов). Ollama на ноутбуке (`brew install ollama && brew services start ollama && ollama pull qwen3:8b`) нужен только для таких проверок; запасной сервер — `scripts/setup_home_server.sh` (`src.server_info --host home`).

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
| `docs/` | справка по гипотезам, журнал отладки промпта, воспроизводимость, трассировка требований, отчёт аудита (`docs/audit/` — независимый пересчёт эталонов) |
| `scripts/` | shell-скрипты: прогоны, анализ, воспроизводимость, туннель к ПК `gpu` |
| `demo/` | результаты на mock и синтетике. **Не коммитится** и не является результатами эксперимента |
| `flow/` | агентский флоу: задания шагов, handoff, оркестратор |

## Как запустить эксперимент

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
   .venv/bin/python -m src.typeE_metrics check --limit 20 [--demo]   # формат пробного прогона (код 4 → вернуться к шагу 07)
   .venv/bin/python -m src.run_classifier --models all --shots 0,5,20 [--demo]              # полный (шаг 09)
   .venv/bin/python -m src.typeE_metrics [--demo]                    # → results/typeE_metrics.csv
   ```
   - `src.typeE classical` → `results/typeE_classical.csv`: Dummy(most_frequent), LogisticRegression и GaussianNB
     (Pipeline: one-hot для `Type` + StandardScaler) на тех же 5 и 20 примерах, что видит LLM, плюс 20 случайных
     стратифицированных наборов для разброса LogReg/NB. Метрики: F1 класса «отказ», balanced accuracy, accuracy,
     precision/recall, матрица ошибок; `eval_set`: `test` (200 строк), `natural` (все строки вне обучения),
     `full` (константа «нет отказа» на всём датасете — accuracy при естественной доле отказов).
   - `src.run_classifier` → `results/typeE_llm.jsonl` (dev: `results/dev_typeE_llm.jsonl`): одна строка на ответ
     (`shots`, `row_idx`, `y_true`, `value` 1/0, `correct`, `confidence`, `call_id`). Ответ модели: `ОТВЕТ: отказ | нет отказа`.
     Оценочный прогон не стартует, пока гипотезы не закоммичены и промпт не заморожен.
   - `src.typeE_metrics` → `results/typeE_metrics.csv`: одна строка на LLM × число примеров и на классическую модель
     (fixed, среднее и sd по resample, константа на test и на всём датасете). Колонки: F1 «отказа», balanced accuracy,
     accuracy, precision/recall, матрица ошибок, `coverage` и метрики `*_answered` (только ответы 0/1), неответы,
     техошибки, `complete`, токены, задержка. Неответ в основных метриках = ошибка. Только числа, без выводов.
7. Автопроверка, листы слепой проверки, тарифы:
   ```bash
   .venv/bin/python -m src.autograde [--demo]      # → results/graded_auto.jsonl + grading/parser_spotcheck.csv
   .venv/bin/python -m src.blind [--demo]          # → grading/sheet_r1.csv, sheet_r2.csv, key_DO_NOT_OPEN.csv
   .venv/bin/python -m src.checks g2               # ручная проверка завершена? (код 0 = да)
   ```
   - `src.autograde` заново разбирает каждый ответ таблиц `results/*.jsonl` (без `dev_*`) и сравнивает с эталоном текущего банка: A, F, вердикт B, E. Для D — pass@1 и корректность метрики из `d_exec.jsonl`; кроме автоматически верных, корректность D ставят люди в листах (метрика из stdout — подсказка). Пропускаются техошибки, задачи без эталона и ответы E на старом разбиении. `correct_at_run` — оценка на момент прогона. Выборка парсера: 20 ответов (до 10 «подозрительных»: `parse_ok=False`, пометки парсера, до 5 ответов E с недословной меткой; остальные случайные поровну по типам, seed из config).
   - `src.blind`: ответы B (обоснование), C, D из `main` и `paraphrase`, без названий моделей и `<think>`; два одинаковых листа, строки по типам, внутри типа перемешаны (seed). Уже заполняемые листы не перезаписываются (читаются и после Excel: «;», cp1251). В ключе — обязательные клетки каждой строки и отпечаток ответа, по ним `checks g2` проверяет полноту и актуальность листов. Правила для проверяющих — `grading/INSTRUCTIONS.md`.
   - `config/pricing.json`: ₽ за 1 млн входных/выходных токенов по официальным страницам (`source_url`, `date_checked`), USD → ₽ по курсу ЦБ (`usd_rub`), локальная модель 0 ₽ (вместо цены — время и память). Сверяет человек: `verified_by_human`.
8. Анализ: `analysis.ipynb` (каждое число отчёта считается в нём из `results/`, `grading/`, `logs/`, `config/`):
   ```bash
   scripts/run_analysis.sh            # выполнить ноутбук на месте (после G2): report/figures/*.png, results/summary_metrics.json
   scripts/run_analysis.sh --demo     # проверка на mock: случайные оценки в demo/grading/ (src.demo_grades),
                                      # копия demo/analysis_demo.ipynb с DEMO = True, всё пишется в demo/
   ```
   - Логика — `src/analysis.py` (тесты `tests/test_analysis.py`), ноутбук только вызывает функции и рисует.
   - «Верно»: A, F, вердикт B, E — автопроверка (`graded_auto.jsonl`); C — доля проверяющих с баллом ≥ 1; D — доля проверяющих,
     отметивших верную итоговую метрику (авто «верно» = 1). Итог = среднее двух проверяющих (0, 0,5, 1), расхождения — списком.
   - Разделы: каппа Коэна (κ < 0,6 — предупреждение), точность по моделям / типам / «модель × тип» с бутстреп-CI (1000 ресэмплов задач,
     seed 42), отказы и две версии точности (вопрос 9), Макнемар `exact=True` для 6 пар + Холм, тип D (pass@1, корректность,
     чек-лист), тип E (F1 «отказа», balanced accuracy по 0/5/20 примерам, LogReg/NB, константа), блоки H1–H13 (метрика, тест,
     рисунок, место под вывод человека), цена и скорость (₽ за правильный ответ, память локальной модели).
   - Рисунки: `heatmap_model_type`, `accuracy_ci`, `cost_accuracy`, `typeD_checklist`, `typeE_shots`, `H*_…`.
9. Отчёт (`report/`; числа в нём только из файлов, текст пишет человек):
   ```bash
   .venv/bin/python -m src.error_candidates [--demo]   # → report/error_candidates.md: 10–15 кандидатов на разбор ошибок
   .venv/bin/python -m src.build_report [--demo]       # → report/report.md: каркас по структуре из PDF
   .venv/bin/python -m src.build_report check          # какие места для текста человека ещё пусты (код 1, если есть)
   ```
   - `src.build_report` берёт числа из `results/summary_metrics.json`, модели и даты — из `results/run_manifest.json`, параметры — из `config/*.json`, таблицу «тип × тема» — из `data/tasks.jsonl`, гипотезы и прогноз — из `hypotheses.md` в версии коммита, полный текст промпта — из `src/prompts.py`. Чего нет — помечено «нет данных». Свой текст человек пишет между строками `<!-- ЧЕЛОВЕК:id -->` и `<!-- /ЧЕЛОВЕК:id -->`: при пересборке он сохраняется (блок, которому нет места в новой структуре, переносится в конец отчёта; если блок испорчен — нет закрывающей строки или опечатка в id, — отчёт не перезаписывается; прошлая версия — `report/report.md.bak`). Текст вне блоков при пересборке пропадает.
   - `src.error_candidates` отбирает по правилам (docstring модуля): неверный ответ при уверенности ≥ 90, переворот после «ты ошибся», число в задаче с недостающими данными, ответ оригинала в изменённой задаче, код D с методологической ошибкой или упавший, C на 0 баллов, пропущенный отказ в E. Для каждого — задача, модель, эталон, точная цитата ответа из журнала и `call_id`. В реальном режиме работает только после G2: до конца слепой проверки ответы с названиями моделей не показываются.
   - `scripts/run_analysis.sh` после ноутбука сам вызывает обе команды.
   - Вопросы самоконтроля — `report/selfcheck_questions.md`, план доклада — `report/slides_outline.md` (пишет человек).
10. Воспроизводимость:
    ```bash
    scripts/check_reproducibility.sh          # копия проекта во временной папке, новое .venv из requirements.txt,
                                              # Restart & Run All analysis.ipynb, сравнение results/summary_metrics.json
    scripts/check_reproducibility.sh --demo   # то же на demo/ (проверка процедуры)
    ```
    Код 0 — JSON совпал побайтно; 1 — различается (печатаются ключи); 2 — нечего проверять; 3 — ноутбук в копии упал. Файлы проекта не меняются. Результаты проверок — `docs/reproducibility.md`.
