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

> Раздел-заглушка. Следующие шаги флоу дополнят его конкретными командами.

1. Установить окружение и заполнить `.env` (см. выше).
2. Smoke-тест моделей (по одному короткому запросу к каждой включённой модели из `config/models.json`; код выхода 0, только если ответили все):
   ```bash
   .venv/bin/python -m src.smoke_test
   ```
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
5. Основной прогон и прогоны под гипотезы: _(шаги 05, 08)_.
6. Тип E (AI4I 2020): _(шаги 06, 09)_.
7. Автопроверка и листы слепой проверки: _(шаг 10)_.
8. Анализ: `analysis.ipynb`, Restart & Run All _(шаг 11)_.
