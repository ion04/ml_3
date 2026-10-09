"""Единый клиент LLM для ЛР3.

ask(model_key, prompt, temperature) -> LLMResult (ответ, токены, время и служебные поля).

Что делает модуль:
- берёт параметры модели из config/models.json;
- вызывает провайдера: gigachat (SDK), openai_compat (YandexGPT, DeepSeek), ollama (нативный API), mock;
- КАЖДУЮ попытку вызова (и успешную, и с ошибкой) сразу дописывает в журнал JSONL
  (logs/calls.jsonl, для mock — demo/calls.jsonl) с flush + fsync;
- кэш: если в журнале уже есть успешная запись с тем же ключом, API не вызывается;
- ретраи с экспоненциальной паузой только для технических ошибок (сеть, таймаут, 429, 5xx).
"""

import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import psutil
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
MODELS_PATH = ROOT / "config" / "models.json"
REAL_LOG = ROOT / "logs" / "calls.jsonl"
DEMO_LOG = ROOT / "demo" / "calls.jsonl"

MAX_ATTEMPTS = 5          # всего попыток на один вызов (1 основная + 4 повтора)
BACKOFF_BASE_S = 2.0      # пауза перед повтором: 2, 4, 8, 16 с
BACKOFF_MAX_S = 60.0
DEFAULT_TIMEOUT_S = 180   # таймаут одного HTTP-запроса, если в конфиге не задан timeout_s

# Ключи читаются из .env в переменные окружения. Уже заданные переменные не перезаписываются.
load_dotenv(ROOT / ".env")


@dataclass
class LLMResult:
    text: str | None                 # ответ модели (None, если все попытки неудачны)
    tokens_in: int | None
    tokens_out: int | None
    latency_s: float | None          # время самого запроса к API, с
    model_reported: str | None       # имя модели, которое вернул API
    effective_temperature: float     # температура, которая реально ушла в API
    cached: bool                     # True: ответ взят из журнала, API не вызывался
    call_id: str | None              # id записи в журнале
    error: str | None                # текст ошибки последней попытки или None
    finish_reason: str | None = None  # stop / length / ...: length = ответ обрезан по max_tokens


# ---------- Конфиг моделей ----------

def ollama_base_url(default: str) -> str:
    """Адрес Ollama: OLLAMA_BASE_URL из .env (удалённый сервер), иначе base_url из конфига.
    Хвост /v1 (адрес OpenAI-совместимого API) отрезается: мы ходим в нативный /api/chat."""
    url = (os.getenv("OLLAMA_BASE_URL") or default).rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def ollama_is_remote() -> bool:
    """Ollama на другом компьютере (задан OLLAMA_BASE_URL)? Тогда память через psutil не меряем:
    даже при SSH-туннеле адрес localhost, а модель работает на сервере."""
    return bool(os.getenv("OLLAMA_BASE_URL"))


def load_models(path=None) -> dict:
    """Словарь key -> описание модели из config/models.json.
    У моделей Ollama base_url заменяется на OLLAMA_BASE_URL, если переменная задана."""
    with open(path or MODELS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    models = {m["key"]: m for m in data["models"]}
    for m in models.values():
        if m["provider"] == "ollama":
            m["base_url"] = ollama_base_url(m["base_url"])
    return models


def get_model(model_key: str) -> dict:
    models = load_models()
    if model_key not in models:
        raise KeyError(f"Модель {model_key!r} не найдена в {MODELS_PATH}")
    return models[model_key]


def _env_vars_in(text: str | None) -> list[str]:
    """Имена переменных вида ${VAR} внутри строки."""
    return re.findall(r"\$\{(\w+)\}", text or "")


def missing_env(cfg: dict) -> list[str]:
    """Какие переменные окружения нужны модели, но не заданы (пустой список = всё есть)."""
    needed = (_env_vars_in(cfg.get("model")) + _env_vars_in(cfg.get("base_url"))
              + _env_vars_in(cfg.get("project")))
    if cfg.get("api_key_env"):
        needed.append(cfg["api_key_env"])
    return [name for name in dict.fromkeys(needed) if not os.getenv(name)]  # без повторов, порядок сохранён


def _expand_env(text: str | None) -> str | None:
    """Подставляет ${VAR} из окружения, например gpt://${YANDEX_FOLDER_ID}/yandexgpt/latest."""
    if text is None:
        return None
    for name in _env_vars_in(text):
        value = os.getenv(name)
        if not value:
            raise ValueError(f"Не задана переменная окружения {name} (нужна для {text!r})")
        text = text.replace("${" + name + "}", value)
    return text


def _api_key(cfg: dict) -> str:
    name = cfg.get("api_key_env")
    value = os.getenv(name) if name else None
    if not value:
        raise ValueError(f"Не задан ключ {name} для модели {cfg['key']} (впишите его в .env)")
    return value


def _scrub(text: str) -> str:
    """Заменяет на *** значения всех секретов из окружения: их нельзя писать в журнал и печатать."""
    for name, value in os.environ.items():
        if value and len(value) >= 8 and re.search(r"(KEY|CREDENTIALS|TOKEN|SECRET|PASSWORD)$", name):
            text = text.replace(value, "***")
    return text


# ---------- Журнал и кэш ----------

def make_cache_key(model_key, model_requested, system, history, prompt,
                   temperature, max_tokens, repeat) -> str:
    """Ключ кэша: sha256 от всех входов, которые влияют на ответ."""
    payload = json.dumps(
        [model_key, model_requested, system, history, prompt, temperature, max_tokens, repeat],
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Индекс успешных записей журнала: путь -> {"size": размер файла, "records": {cache_key: запись}}.
# Если размер файла изменился не через этот процесс, индекс перечитывается целиком.
_cache_index: dict[str, dict] = {}
# Прогонщики вызывают ask() из нескольких потоков: запись в журнал и индекс кэша — по одному потоку.
_log_lock = threading.RLock()


def _read_successes(log_path: Path) -> dict:
    records = {}
    if not log_path.exists():
        return records
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # недописанная строка (например, процесс убили при записи)
            # Успешная запись = вызов без ошибки. Пустой текст (например, сработал фильтр контента) —
            # тоже ответ модели: его нельзя перезапрашивать, поэтому он кэшируется.
            if rec.get("error") is None:
                # Берём самую первую успешную запись: именно этот ответ и есть «тот самый».
                records.setdefault(rec.get("cache_key"), rec)
    return records


def _cache_lookup(log_path: Path, key: str) -> dict | None:
    with _log_lock:
        size = log_path.stat().st_size if log_path.exists() else 0
        entry = _cache_index.get(str(log_path))
        if entry is None or entry["size"] != size:
            entry = {"size": size, "records": _read_successes(log_path)}
            _cache_index[str(log_path)] = entry
        return entry["records"].get(key)


def _append_log(log_path: Path, record: dict) -> None:
    """Дописывает одну запись в конец журнала и сразу сбрасывает её на диск."""
    line = _scrub(json.dumps(record, ensure_ascii=False))  # страховка: секретов в журнале нет
    with _log_lock:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        size_before = log_path.stat().st_size if log_path.exists() else 0
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        # Обновляем индекс кэша без перечитывания файла (если файл никто другой не менял).
        entry = _cache_index.get(str(log_path))
        if entry is not None:
            if entry["size"] == size_before:
                entry["size"] = log_path.stat().st_size
                if record["error"] is None:
                    entry["records"].setdefault(record["cache_key"], record)
            else:
                del _cache_index[str(log_path)]


def _resolve_log_path(provider: str, log_path) -> Path:
    """mock пишет только в demo/: демо-ответы не должны попасть в журнал эксперимента."""
    if log_path is None:
        return DEMO_LOG if provider == "mock" else REAL_LOG
    log_path = Path(log_path).resolve()
    if provider == "mock" and log_path.is_relative_to(ROOT / "logs"):
        raise ValueError("Ответы mock нельзя писать в logs/: используйте demo/")
    return log_path


# ---------- Ретраи ----------

def _status_code(exc: Exception) -> int | None:
    """HTTP-код ошибки, если он есть (openai, gigachat и httpx хранят его по-разному)."""
    code = getattr(exc, "status_code", None)
    if code is None and getattr(exc, "response", None) is not None:
        code = getattr(exc.response, "status_code", None)
    return code if isinstance(code, int) else None


def is_retryable(exc: Exception) -> bool:
    """Повторяем только технические ошибки: сеть, таймаут, 429, 5xx."""
    import openai  # импорт здесь, чтобы модуль не зависел от порядка импорта SDK

    if "CERTIFICATE_VERIFY_FAILED" in str(exc):
        return False  # ошибка TLS-сертификата (GigaChat без сертификата НУЦ) повтором не лечится
    if isinstance(exc, (httpx.TransportError, openai.APIConnectionError, ConnectionError, TimeoutError)):
        return True
    code = _status_code(exc)
    return code is not None and (code == 429 or code >= 500)


def _backoff_s(attempt: int) -> float:
    return min(BACKOFF_BASE_S * 2 ** (attempt - 1), BACKOFF_MAX_S)


# ---------- Замер памяти локальной модели ----------

def _ollama_rss_mb() -> float:
    """Суммарная резидентная память (RSS) процессов ollama и их дочерних процессов, МБ.

    Веса модели держит дочерний процесс-раннер (в Ollama 0.40 это llama-server), поэтому
    считаем не только сам `ollama serve`, но и всех его потомков.
    """
    procs = {}
    for proc in psutil.process_iter(["name"]):
        if "ollama" in (proc.info["name"] or "").lower():
            procs[proc.pid] = proc
            try:
                for child in proc.children(recursive=True):
                    procs[child.pid] = child
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    total = 0
    for proc in procs.values():
        try:
            total += proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / 2**20


class _MemWatcher:
    """Фоновый поток: пока идёт вызов, раз в 0,2 с замеряет RSS ollama и хранит максимум."""

    def __init__(self, interval_s=0.2):
        self.interval_s = interval_s
        self.max_mb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while True:
            self.max_mb = max(self.max_mb, _ollama_rss_mb())
            if self._stop.wait(self.interval_s):
                break

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.max_mb = max(self.max_mb, _ollama_rss_mb())  # последний замер после ответа


def ollama_ps(base_url: str, model_id: str) -> dict | None:
    """Что Ollama сообщает о загруженной модели (/api/ps): размер в памяти и сколько из него на GPU."""
    try:
        data = httpx.get(base_url.rstrip("/") + "/api/ps", timeout=5).json()
    except (httpx.HTTPError, ValueError):
        return None
    for m in data.get("models", []):
        if m.get("name") == model_id or m.get("model") == model_id:
            return {"size_mb": round(m.get("size", 0) / 2**20, 1),
                    "size_vram_mb": round(m.get("size_vram", 0) / 2**20, 1)}
    return None


# ---------- Провайдеры ----------
# Каждая функция возвращает dict: text, tokens_in, tokens_out, model_reported,
# finish_reason, reasoning_text, latency_s (время самого запроса).

def _call_openai_compat(cfg, model_id, messages, temperature, max_tokens, repeat):
    from openai import OpenAI

    # max_retries=0: повторы делает ask(), чтобы каждая попытка попала в журнал.
    # project: для Yandex AI Studio сюда передаётся folder_id (необязательное поле конфига).
    client = OpenAI(api_key=_api_key(cfg), base_url=_expand_env(cfg["base_url"]),
                    project=_expand_env(cfg.get("project")),
                    max_retries=0, timeout=cfg.get("timeout_s", DEFAULT_TIMEOUT_S))
    extra = dict(cfg.get("params") or {})
    if max_tokens is not None:
        extra["max_tokens"] = max_tokens
    t0 = time.perf_counter()
    resp = client.chat.completions.create(model=model_id, messages=messages,
                                          temperature=temperature, **extra)
    latency = time.perf_counter() - t0
    choice = resp.choices[0]
    return {
        "text": choice.message.content,
        "tokens_in": resp.usage.prompt_tokens if resp.usage else None,
        "tokens_out": resp.usage.completion_tokens if resp.usage else None,
        "model_reported": resp.model,
        "finish_reason": choice.finish_reason,
        # DeepSeek в режиме рассуждений отдаёт их отдельным полем
        "reasoning_text": getattr(choice.message, "reasoning_content", None),
        "latency_s": latency,
    }


_gigachat_clients: dict = {}  # клиент на модель: OAuth-токен живёт 30 мин и переиспользуется


def _call_gigachat(cfg, model_id, messages, temperature, max_tokens, repeat):
    from gigachat import GigaChat
    from gigachat.models import Chat, Messages

    client = _gigachat_clients.get(cfg["key"])
    if client is None:
        # Scope, проверка TLS (GIGACHAT_VERIFY_SSL_CERTS, GIGACHAT_CA_BUNDLE_FILE) SDK читает из окружения сам.
        client = GigaChat(credentials=_api_key(cfg), model=model_id,
                          base_url=_expand_env(cfg.get("base_url")),
                          timeout=cfg.get("timeout_s", DEFAULT_TIMEOUT_S), max_retries=0)
        _gigachat_clients[cfg["key"]] = client
    client.get_token()  # получить/обновить OAuth-токен до замера времени
    chat = Chat(model=model_id, temperature=temperature, max_tokens=max_tokens,
                messages=[Messages(role=m["role"], content=m["content"]) for m in messages],
                **(cfg.get("params") or {}))
    t0 = time.perf_counter()
    resp = client.chat(chat)
    latency = time.perf_counter() - t0
    choice = resp.choices[0]
    return {
        "text": choice.message.content,
        "tokens_in": resp.usage.prompt_tokens,
        "tokens_out": resp.usage.completion_tokens,
        "model_reported": resp.model,
        "finish_reason": choice.finish_reason,
        "reasoning_text": choice.message.reasoning_content,
        "latency_s": latency,
    }


def _call_ollama(cfg, model_id, messages, temperature, max_tokens, repeat):
    # Нативный API Ollama (/api/chat): только в нём режим рассуждений Qwen3 выключается
    # параметром think=false (он задаётся в params конфига).
    params = dict(cfg.get("params") or {})
    options = dict(params.pop("options", {}))
    options["temperature"] = temperature
    options["seed"] = 42 + repeat  # воспроизводимость; у повторов (H7) разные seed
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    body = {"model": model_id, "messages": messages, "stream": False, "options": options, **params}
    t0 = time.perf_counter()
    r = httpx.post(cfg["base_url"].rstrip("/") + "/api/chat", json=body,
                   timeout=cfg.get("timeout_s", DEFAULT_TIMEOUT_S))
    latency = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    return {
        "text": data["message"]["content"],
        "tokens_in": data.get("prompt_eval_count"),
        "tokens_out": data.get("eval_count"),
        "model_reported": data.get("model"),
        "finish_reason": data.get("done_reason"),
        "reasoning_text": data["message"].get("thinking"),
        "latency_s": latency,
    }


MOCK_TEMPLATE = "Демо-ответ mock-модели {model} (сообщений в диалоге: {n_messages}).\nОТВЕТ: {answer}\nУВЕРЕННОСТЬ: {confidence}"
MOCK_BOOL_TEMPLATE = "Демо-ответ mock-модели {model} (сообщений в диалоге: {n_messages}).\nОТВЕТ: {verdict}\nУВЕРЕННОСТЬ: {confidence}"
MOCK_CODE_TEMPLATE = "Демо-ответ mock-модели {model}.\n```python\n{code}\n```\nОТВЕТ: код выше\nУВЕРЕННОСТЬ: {confidence}"
MOCK_LABEL_TEMPLATE = "Демо-ответ mock-модели {model} (сообщений в диалоге: {n_messages}).\nОТВЕТ: {label}\nУВЕРЕННОСТЬ: {confidence}"


def _call_mock(cfg, model_id, messages, temperature, max_tokens, repeat):
    """Детерминированный ответ без сети: зависит только от входа (модель, сообщения, T, repeat).

    Чтобы демо-прогон проходил все ветки разбора, mock смотрит, что просили в сообщениях пользователя:
    код (```python) → маленькая программа (каждая 4-я падает), «верно или неверно» → вердикт,
    классификация типа E («нет отказа» / «no failure») → класс (примерно каждый 5-й — отказ), иначе число.
    params.template в конфиге задаёт свой шаблон и отключает этот выбор.
    """
    seed_text = json.dumps([model_id, messages, temperature, repeat], ensure_ascii=False)
    n = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
    template = (cfg.get("params") or {}).get("template")
    if template is None:
        if "```python" in user_text:
            template = MOCK_CODE_TEMPLATE
        elif "верно или неверно" in user_text or "true or false" in user_text:
            template = MOCK_BOOL_TEMPLATE
        elif "нет отказа" in user_text or "no failure" in user_text:
            template = MOCK_LABEL_TEMPLATE
        else:
            template = MOCK_TEMPLATE
    verdict = ("верно" if n % 2 else "неверно") if "верно или неверно" in user_text else ("true" if n % 2 else "false")
    if n % 4 == 0:
        code = 'raise RuntimeError("демо: код mock-модели упал")'
    else:
        code = f'print("accuracy: {0.9 + (n % 10) / 100:.4f}")'
    if "нет отказа" in user_text:
        label = "отказ" if n % 5 == 0 else "нет отказа"
    else:
        label = "failure" if n % 5 == 0 else "no failure"
    text = template.format(model=model_id, n_messages=len(messages), answer=n % 100,
                           confidence=50 + n % 51, verdict=verdict, code=code, label=label)
    return {
        "text": text,
        "tokens_in": sum(len(m["content"].split()) for m in messages),
        "tokens_out": len(text.split()),
        "model_reported": f"{model_id}-mock",
        "finish_reason": "stop",
        "reasoning_text": None,
        "latency_s": 0.0,
    }


def _call_provider(provider, cfg, model_id, messages, temperature, max_tokens, repeat):
    if provider == "openai_compat":
        return _call_openai_compat(cfg, model_id, messages, temperature, max_tokens, repeat)
    if provider == "gigachat":
        return _call_gigachat(cfg, model_id, messages, temperature, max_tokens, repeat)
    if provider == "ollama":
        return _call_ollama(cfg, model_id, messages, temperature, max_tokens, repeat)
    if provider == "mock":
        return _call_mock(cfg, model_id, messages, temperature, max_tokens, repeat)
    raise ValueError(f"Неизвестный провайдер {provider!r}")


# ---------- Главная функция ----------

def _build_messages(system, history, prompt) -> list[dict]:
    """system + предыдущие ходы диалога + новый вопрос пользователя."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    for turn in history:
        if turn.get("role") not in ("user", "assistant") or not isinstance(turn.get("content"), str):
            raise ValueError(f"Неверный элемент history: {turn!r}")
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": prompt})
    return messages


def _result_from_record(rec: dict, cached: bool) -> LLMResult:
    return LLMResult(
        text=rec["response_text"], tokens_in=rec["tokens_in"], tokens_out=rec["tokens_out"],
        latency_s=rec["latency_s"], model_reported=rec["model_reported"],
        effective_temperature=rec["effective_temperature"], cached=cached,
        call_id=rec["call_id"], error=rec["error"], finish_reason=rec.get("finish_reason"),
    )


def ask(model_key, prompt, temperature=0.0, *, system=None, history=None, max_tokens=None,
        meta=None, repeat=0, use_cache=True, log_path=None) -> LLMResult:
    """Один запрос к модели model_key (ключ из config/models.json).

    history: [{"role": "user"|"assistant", "content": ...}, ...] — предыдущие ходы диалога.
    meta: task_id, variant, experiment, lang и т. п.; пишется в журнал как есть (+ repeat).
    repeat: номер повтора одного и того же запроса (H7); входит в ключ кэша.
    Ошибки не выбрасываются: при неудаче всех попыток возвращается LLMResult с text=None и error.
    """
    cfg = get_model(model_key)
    provider = cfg["provider"]
    model_id = _expand_env(cfg["model"])
    # Если провайдер не принимает 0, берём минимально допустимую температуру из конфига.
    eff_temp = max(float(temperature), float(cfg.get("min_temperature") or 0.0))
    history = list(history or [])
    meta = {**(meta or {}), "repeat": repeat}
    log_path = _resolve_log_path(provider, log_path)
    key = make_cache_key(model_key, model_id, system, history, prompt, eff_temp, max_tokens, repeat)

    if use_cache:
        rec = _cache_lookup(log_path, key)
        if rec is not None:
            return _result_from_record(rec, cached=True)

    messages = _build_messages(system, history, prompt)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        ts_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        t0 = time.perf_counter()
        out, error, exc, local_mem_mb = None, None, None, None
        try:
            # Память через psutil меряется, только если Ollama на этом компьютере.
            # У удалённого сервера (OLLAMA_BASE_URL) размер модели берётся из /api/ps ниже.
            if provider == "ollama" and not ollama_is_remote():
                with _MemWatcher() as mem:
                    out = _call_provider(provider, cfg, model_id, messages, eff_temp, max_tokens, repeat)
                local_mem_mb = round(mem.max_mb, 1)
            else:
                out = _call_provider(provider, cfg, model_id, messages, eff_temp, max_tokens, repeat)
        except Exception as e:  # любая ошибка пишется в журнал; повторяем только технические
            exc = e
            error = f"{type(e).__name__}: {e}"
            if _status_code(e) is not None:
                error = f"HTTP {_status_code(e)} {error}"
            error = _scrub(error)[:2000]
        elapsed = time.perf_counter() - t0
        out = out or {}

        record = {
            "call_id": uuid.uuid4().hex,
            "ts_utc": ts_utc,
            "model_key": model_key,
            "provider": provider,
            "model_requested": model_id,
            "model_reported": out.get("model_reported"),
            "temperature_requested": temperature,
            "effective_temperature": eff_temp,
            "max_tokens": max_tokens,
            "params": cfg.get("params") or {},
            "system": system,
            "history": history,
            "prompt": prompt,
            "response_text": out.get("text"),
            "reasoning_text": out.get("reasoning_text"),
            "finish_reason": out.get("finish_reason"),
            "tokens_in": out.get("tokens_in"),
            "tokens_out": out.get("tokens_out"),
            "latency_s": round(out.get("latency_s", elapsed), 3),
            "error": error,
            "attempt": attempt,
            "meta": meta,
            "cache_key": key,
            "local_mem_mb": local_mem_mb,
            "local_vram_mb": None,
        }
        if provider == "ollama":
            record["ollama_base_url"] = cfg["base_url"]    # где считалась локальная модель (ноутбук / сервер)
            if error is None:
                ps = ollama_ps(cfg["base_url"], model_id)
                record["local_vram_mb"] = ps["size_vram_mb"] if ps else None
                record["local_model_size_mb"] = ps["size_mb"] if ps else None   # вся модель в памяти (/api/ps)
        _append_log(log_path, record)

        if error is None or not is_retryable(exc) or attempt == MAX_ATTEMPTS:
            return _result_from_record(record, cached=False)
        time.sleep(_backoff_s(attempt))
