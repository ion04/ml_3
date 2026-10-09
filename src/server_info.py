"""Железо сервера, на котором считается локальная модель (Ollama), — для отчёта.

Запуск (сервер должен быть доступен по ssh):
  .venv/bin/python -m src.server_info              # показать
  .venv/bin/python -m src.server_info --write      # дописать в results/run_manifest.json → "ollama_server"

Сведения собираются одной командой по ssh (hostname, ОС, CPU, RAM, GPU, версия Ollama, ID модели).
Остальные поля run_manifest.json (шаг 08) не трогаются: файл читается и обновляется по ключу.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

from src import llm

MANIFEST = llm.ROOT / "results" / "run_manifest.json"

# Каждая строка вывода — «ключ=значение». Команды, которых нет на сервере, дают пустое значение.
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
    if info.get("cpu_threads"):
        info["cpu_threads"] = int(info["cpu_threads"])
    return info


def collect(ssh_host: str = "home", timeout_s: int = 30) -> dict:
    out = subprocess.run(["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh_host, "bash -s"],
                         input=REMOTE_SCRIPT, capture_output=True, text=True, timeout=timeout_s)
    if out.returncode != 0:
        raise RuntimeError(f"ssh {ssh_host}: код {out.returncode}: {out.stderr.strip()[:300]}")
    info = parse_output(out.stdout)
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
    ap.add_argument("--host", default="home", help="ssh-хост сервера (по умолчанию home)")
    ap.add_argument("--write", action="store_true", help="дописать в results/run_manifest.json")
    args = ap.parse_args(argv)
    try:
        info = collect(args.host)
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"Сервер недоступен: {e}")
        return 1
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if args.write:
        write_manifest(info)
        print(f"Записано: results/run_manifest.json → ollama_server")
    return 0


if __name__ == "__main__":
    sys.exit(main())
