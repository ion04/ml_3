"""Тесты src/server_info.py: разбор вывода и запись в манифест (без ssh)."""

import json
import subprocess

from src import server_info

SAMPLE = """hostname=home
os=Ubuntu 24.04.1 LTS
kernel=Linux 6.8.0-45-generic
cpu=AMD Ryzen 7 5800X 8-Core Processor
cpu_threads=16
ram_bytes=34359738368
gpu=NVIDIA GeForce RTX 3060, 12288 MiB
ollama_version=0.40.2
models=qwen3:8b 500a1f067a9f
"""


def test_parse_output():
    info = server_info.parse_output(SAMPLE)
    assert info["ram_gb"] == 32.0 and info["cpu_threads"] == 16
    assert info["gpu"] == ["NVIDIA GeForce RTX 3060, 12288 MiB"]
    assert info["models"] == ["qwen3:8b 500a1f067a9f"] and info["ollama_version"] == "0.40.2"


def test_parse_output_missing_commands():
    """На сервере без GPU и без Ollama поля пустые, а не ошибка."""
    info = server_info.parse_output("hostname=x\ncpu=\ngpu=\nmodels=\nram_bytes=\n")
    assert info["cpu"] is None and info["gpu"] == [] and info["models"] == [] and "ram_gb" not in info


def test_remote_script_runs_locally():
    """Скрипт синтаксически корректен и печатает все ключи (запуск в локальном bash, без ssh)."""
    out = subprocess.run(["bash", "-s"], input=server_info.REMOTE_SCRIPT, capture_output=True, text=True)
    keys = {line.split("=", 1)[0] for line in out.stdout.splitlines() if "=" in line}
    assert {"hostname", "os", "cpu", "gpu", "ollama_version", "models"} <= keys


def test_write_manifest_keeps_other_fields(tmp_path):
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps({"runs": [1, 2]}), encoding="utf-8")
    server_info.write_manifest({"hostname": "home"}, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {"runs": [1, 2], "ollama_server": {"hostname": "home"}}


WIN_SAMPLE = """hostname=DESKTOP-X
os=Windows 10 Pro 10.0.19045
cpu=AMD Ryzen Threadripper 3970X 32-Core Processor
cpu_threads=64
ram_bytes=34224627712
ram_modules=8 GB 2133 MHz;8 GB 2133 MHz
gpu=NVIDIA GeForce RTX 3090, 24576 MiB, 591.74
"""


def test_collect_windows(monkeypatch):
    """Windows-ПК: ОС определяется по `echo %OS%`, скрипт уходит в PowerShell, Ollama — через API."""
    sent = []

    def fake_ssh(host, command, timeout_s, stdin=None):
        sent.append(command)
        return "Windows_NT\r\n" if command == "echo %OS%" else WIN_SAMPLE

    monkeypatch.setattr(server_info, "ssh", fake_ssh)
    monkeypatch.setattr(server_info, "ollama_api_info",
                        lambda url: {"ollama_version": "0.40.2", "models": ["qwen3:8b 500a1f067a9f"]})
    info = server_info.collect("gpu")
    assert sent[1].startswith("powershell -NoProfile -NonInteractive -EncodedCommand ")
    assert info["ram_modules"] == ["8 GB 2133 MHz", "8 GB 2133 MHz"] and info["cpu_threads"] == 64
    assert info["ollama_version"] == "0.40.2" and info["models"] == ["qwen3:8b 500a1f067a9f"]
    assert info["ssh_host"] == "gpu"


def test_collect_linux_uses_bash(monkeypatch):
    sent = []

    def fake_ssh(host, command, timeout_s, stdin=None):
        sent.append((command, stdin))
        return "%OS%\n" if command == "echo %OS%" else SAMPLE

    monkeypatch.setattr(server_info, "ssh", fake_ssh)
    info = server_info.collect("home")
    assert sent[1] == ("bash -s", server_info.REMOTE_SCRIPT)
    assert info["models"] == ["qwen3:8b 500a1f067a9f"] and "ram_modules" not in info
