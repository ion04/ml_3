STATUS: done
## Сделано
- `git init -b main`, `.gitignore` по заданию (`logs/`, `results/` отслеживаются). `core.quotepath=false`, чтобы git показывал кириллицу в путях.
- `.venv` на **Python 3.12.14** (uv-managed; системный 3.14 не используется): `uv venv --seed --python 3.12 .venv`. Флаг `--seed` нужен, чтобы внутри был pip.
- Установлены все пакеты из задания; `requirements.txt` = `pip freeze` (137 строк). Ключевые версии: openai 3.27.0, gigachat 0.2.3, scikit-learn 1.9.1, statsmodels 0.15.0, pandas 3.0.6, numpy 2.5.3, ucimlrepo 0.0.7.
- Каркас каталогов: `data/ src/ config/ logs/ results/ grading/ report/figures/ tests/ docs/ scripts/ demo/`; `src/__init__.py`, `tests/__init__.py`, в пустых каталогах `.gitkeep` (в `demo/` нет: он игнорируется).
- `.env.example` с комментариями, где взять каждый ключ. `README.md`: установка, структура, ссылки, заглушка «Как запустить эксперимент».
- Ollama 0.40.2 (`brew install ollama`), сервис запущен через `brew services start ollama`, модель `qwen3:8b` скачана (ID `500a1f067a9f`, 5.2 ГБ).
## Ключевые файлы и точки входа
- `.env.example`, `README.md`, `requirements.txt`, `.gitignore`, `src/__init__.py`.
- Python: `.venv/bin/python`; тесты: `.venv/bin/python -m pytest` (пока тестов нет, код выхода 5).
## Решения и допущения
- Переменные GigaChat названы так, как их читает SDK (pydantic-settings, префикс `GIGACHAT_`): `GIGACHAT_CREDENTIALS`, `GIGACHAT_SCOPE`, `GIGACHAT_VERIFY_SSL_CERTS`, `GIGACHAT_CA_BUNDLE_FILE`, а не `GIGACHAT_VERIFY_SSL` / `GIGACHAT_CA_BUNDLE` из задания. SDK подхватывает их из окружения сам.
- В gigachat 0.2.3 по умолчанию base_url = `https://api.giga.chat/v1`, auth_url = `https://ngw.devices.sberbank.ru:9443/api/v2/oauth`, timeout 30 с (для длинных ответов может не хватить, лучше задать явно).
- Остальные переменные: `YANDEX_API_KEY`, `YANDEX_FOLDER_ID`, `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY` (альтернатива для Qwen), `OLLAMA_BASE_URL=http://localhost:11434/v1`.
## Проверка
- `.venv/bin/python -c "import sklearn, statsmodels, openai, gigachat, ucimlrepo"`: ок (а также pandas, scipy, seaborn, nbformat, tenacity, dotenv, tqdm, psutil, pytest).
- `curl localhost:11434/api/version` → 0.40.2; `ollama list` показывает `qwen3:8b`.
- `git ls-files` не содержит `.env`, `demo/`, `.venv/`, `flow/state/`, `flow/logs/`.
## Что нужно для завершения
- Ничего: шаг выполнен. Ключи API вписывает человек (G1 в HUMAN_TODO).
## Открытые вопросы и риски
- Модель Qwen3 по умолчанию «думает». На шаге 02 режим рассуждений нужно выключить (`think: false` в нативном API Ollama или `/no_think`) и проверить это на реальном ответе.
- Сертификат НУЦ Минцифры для GigaChat не скачан. Решение (CA bundle или `VERIFY_SSL_CERTS=False`) за человеком, пункт есть в G1.
- Ollama-сервис запущен через launchd (brew services), после перезагрузки он поднимется сам.
## Для следующих шагов
- Железо для отчёта по локальной модели: MacBook Pro (MacBookPro18,3), Apple M1 Pro (arm64), 10 ядер CPU (8P+2E), GPU 16 ядер (Metal 4), 16 ГБ unified memory; macOS 26.6.2. Вызовов модели на шаге 01 не было (все вызовы LLM идут только через `src/llm.py::ask`), поэтому размещение на GPU проверить командой `ollama ps` во время первого вызова (колонка PROCESSOR).
- Новые пакеты ставить так: `.venv/bin/python -m pip install X`, затем снова `pip freeze > requirements.txt`.
- Раздел README «Как запустить эксперимент» сейчас заглушка с пометками «(шаг NN)», каждый шаг дописывает в него свои команды.
- Пустые каталоги отслеживаются через `.gitkeep`; при появлении файлов `.gitkeep` можно не удалять.
