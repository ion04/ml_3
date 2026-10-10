"""Железо сервера, на котором считается локальная модель (Ollama), — для отчёта.

Запуск (сервер должен быть доступен по ssh):
  .venv/bin/python -m src.server_info                       # показать (по умолчанию ПК gpu)
  .venv/bin/python -m src.server_info --write --note "..."  # дописать в results/run_manifest.json → "ollama_server"

Сведения собираются одной командой по ssh (hostname, ОС, CPU, RAM, GPU, версия Ollama, ID модели).
Основной вариант — Windows-ПК gpu (Ollama там запускает scripts/gpu_ollama.sh): PowerShell-скрипт WINDOWS_SCRIPT,
а версия Ollama и список моделей берутся из API Ollama по OLLAMA_BASE_URL (ollama.exe там не в PATH).
Для Linux-сервера (запасной вариант) — bash-скрипт REMOTE_SCRIPT.
Остальные поля run_manifest.json (шаг 08) не трогаются: файл читается и обновляется по ключу.
"""

import argparse
import base64
import json
import subprocess
import sys
from datetime import datetime, timezone

import httpx

from src import llm

MANIFEST = llm.ROOT / "results" / "run_manifest.json"

# Linux-сервер. Каждая строка вывода — «ключ=значение». Команды, которых нет на сервере, дают пустое значение.
REMOTE_SCRIPT = r"""
export LC_ALL=C   # иначе lscpu и free печатают заголовки на языке системы (например, по-русски)
echo "hostname=$(hostname)"
echo "os=$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
echo "kernel=$(uname -sr)"
echo "cpu=$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
echo "cpu_threads=$(nproc 2>/dev/null)"
echo "ram_bytes=$(free -b 2>/dev/null | awk '/^Mem:/ {print $2}')"
echo "gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd ';' -)"
echo "ollama_version=$(ollama --version 2>/dev/null | grep -o '[0-9][0-9.]*' | tail -1)"
echo "models=$(ollama list 2>/dev/null | awk 'NR>1 {print $1 " " $2}' | paste -sd ';' -)"
"""


# То же для Windows. Вывод в UTF-8 (иначе русское название ОС приходит в кодировке консоли),
# индикатор прогресса выключен (иначе PowerShell печатает служебный CLIXML).
WINDOWS_SCRIPT = r"""
$ProgressPreference = 'SilentlyContinue'
$ErrorActionPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$os = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$mem = Get-CimInstance Win32_PhysicalMemory
"hostname=$env:COMPUTERNAME"
"os=$($os.Caption) $($os.Version)"
"cpu=$($cpu.Name.Trim())"
"cpu_threads=$($cpu.NumberOfLogicalProcessors)"
"ram_bytes=$($os.TotalVisibleMemorySize * 1024)"
"ram_modules=$(($mem | ForEach-Object { '{0} GB {1} MHz' -f ($_.Capacity / 1GB), $_.Speed }) -join ';')"
"gpu=$((& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader) -join ';')"
"""


def parse_output(text: str) -> dict:
    """Строки «ключ=значение» → словарь; объём памяти переводится в ГБ, списки разбиваются по «;»."""
    info = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            info[key.strip()] = value.strip() or None
    if info.get("ram_bytes"):
        info["ram_gb"] = round(int(info.pop("ram_bytes")) / 2**30, 1)
    else:
        info.pop("ram_bytes", None)
    for key in ("gpu", "models"):
        info[key] = info[key].split(";") if info.get(key) else []
    if "ram_modules" in info:   # есть только у Windows-скрипта
        info["ram_modules"] = info["ram_modules"].split(";") if info["ram_modules"] else []
    if info.get("cpu_threads"):
        info["cpu_threads"] = int(info["cpu_threads"])
    return info


def ssh(ssh_host: str, command: str, timeout_s: int, stdin: str | None = None) -> str:
    out = subprocess.run(["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh_host, command],
                         input=stdin, capture_output=True, text=True, timeout=timeout_s)
    if out.returncode != 0:
        raise RuntimeError(f"ssh {ssh_host}: код {out.returncode}: {out.stderr.strip()[:300]}")
    return out.stdout


def ollama_api_info(base_url: str) -> dict:
    """Версия Ollama и скачанные модели («имя id12») по HTTP API; пустые поля, если сервер не ответил."""
    try:
        version = httpx.get(base_url + "/api/version", timeout=5).json().get("version")
        tags = httpx.get(base_url + "/api/tags", timeout=5).json().get("models", [])
    except (httpx.HTTPError, ValueError):
        return {"ollama_version": None, "models": []}
    return {"ollama_version": version, "models": [f"{m['name']} {m.get('digest', '')[:12]}" for m in tags]}


def collect(ssh_host: str = "gpu", timeout_s: int = 60) -> dict:
    # cmd.exe (оболочка ssh на Windows) печатает «Windows_NT», bash — строку «%OS%» как есть
    windows = ssh(ssh_host, "echo %OS%", timeout_s).strip() == "Windows_NT"
    if windows:
        # PowerShell принимает скрипт в -EncodedCommand: base64 от UTF-16LE, без проблем с кавычками
        encoded = base64.b64encode(WINDOWS_SCRIPT.encode("utf-16-le")).decode()
        info = parse_output(ssh(ssh_host, f"powershell -NoProfile -NonInteractive -EncodedCommand {encoded}", timeout_s))
        base_url = llm.ollama_base_url("http://localhost:11434")
        info.update(ollama_api_info(base_url))
        info["ollama_base_url"] = base_url
    else:
        info = parse_output(ssh(ssh_host, "bash -s", timeout_s, stdin=REMOTE_SCRIPT))
    info["ssh_host"] = ssh_host
    info["collected_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return info


def write_manifest(info: dict, path=MANIFEST) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest["ollama_server"] = info
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Железо сервера локальной модели")
    ap.add_argument("--host", default="gpu", help="ssh-хост сервера (по умолчанию gpu)")
    ap.add_argument("--write", action="store_true", help="дописать в results/run_manifest.json")
    ap.add_argument("--note", help="примечание о сервере для отчёта (например, известные сбои железа)")
    args = ap.parse_args(argv)
    try:
        info = collect(args.host)
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"Сервер недоступен: {e}")
        return 1
    if args.note:
        info["note"] = args.note
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if args.write:
        write_manifest(info)
        print("Записано: results/run_manifest.json → ollama_server")
    return 0


if __name__ == "__main__":
    sys.exit(main())
