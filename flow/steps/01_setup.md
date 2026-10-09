# Цель
Подготовить репозиторий и окружение, чтобы следующие шаги могли сразу писать код.

# Что сделать
1. `git init -b main`, если репозитория ещё нет. В `.gitignore` внести:
   `.venv/`, `.env`, `demo/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `.DS_Store`, `flow/state/`, `flow/logs/`, `.pytest_cache/`.
   Каталоги `logs/` и `results/` **не** игнорировать: там данные эксперимента.
2. Создать виртуальное окружение `.venv`.
   - Системный Python 3.14 может быть слишком новым для части пакетов (gigachat, statsmodels, ucimlrepo). В этом случае поставь Python 3.12 через `uv`, если он есть, иначе через `brew install python@3.12`, и создай venv на нём.
   - Выбранную версию Python запиши в README и в handoff.
3. Установить зависимости:
   - работа с API: `openai`, `gigachat`, `python-dotenv`, `tenacity`;
   - данные и модели: `pandas`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `ucimlrepo`;
   - графики: `matplotlib`, `seaborn`;
   - ноутбуки: `jupyter`, `nbconvert`, `ipykernel`, `nbformat`;
   - разное: `pytest`, `tqdm`, `psutil`.
   После установки: `pip freeze > requirements.txt`.
4. Создать каркас каталогов: `data/`, `src/` (пакет с `__init__.py`), `config/`, `logs/`, `results/`, `grading/`, `report/figures/`, `tests/`, `docs/`, `scripts/`, `demo/`. В пустые отслеживаемые каталоги положить `.gitkeep`.
5. Создать `.env.example` без значений, с комментарием к каждой переменной: где получить ключ и как.
   - `GIGACHAT_CREDENTIALS`, `GIGACHAT_SCOPE=GIGACHAT_API_PERS`, `GIGACHAT_VERIFY_SSL` / `GIGACHAT_CA_BUNDLE`;
   - `YANDEX_API_KEY`, `YANDEX_FOLDER_ID`;
   - `DEEPSEEK_API_KEY` (вариант: `DASHSCOPE_API_KEY` для Qwen);
   - `OLLAMA_BASE_URL=http://localhost:11434/v1`.
6. Создать `README.md` проекта:
   - установка окружения;
   - структура каталогов;
   - ссылки на `PLAN.md`, `HUMAN_TODO.md`, `flow/README.md`;
   - раздел «Как запустить эксперимент». В нём пока заглушка, следующие шаги будут его дополнять.
7. Ollama:
   - проверь `which ollama` и `ollama list`;
   - если Ollama установлен, но локальной модели нет, скачай `qwen3:8b` (`ollama pull qwen3:8b`, около 5 ГБ; сначала проверь, что места на диске больше 20 ГБ);
   - если Ollama не установлен, попробуй `brew install ollama` и запусти сервис (`brew services start ollama`);
   - если не получилось, добавь пункт в HUMAN_TODO G1.
   Запиши в handoff параметры железа: CPU, RAM, наличие GPU / Apple Silicon. Они нужны для отчёта по локальной модели.
8. Проверка: `.venv/bin/python -c "import sklearn, statsmodels, openai, gigachat, ucimlrepo"` проходит без ошибок.
9. Первый коммит включает уже существующие файлы (`PLAN.md`, `CLAUDE.md`, `HUMAN_TODO.md`, `flow/`, PDF задания).
