#!/usr/bin/env bash
# Подготовка домашнего сервера (Linux) для локальной модели: Ollama + qwen3:8b.
# Полные прогоны локальной модели идут на сервере, ноутбук (M1 Pro, 16 ГБ) — только для коротких проверок.
#
# Запуск с ноутбука (нужен sudo на сервере, поэтому ssh -t):
#   scp scripts/setup_home_server.sh home:/tmp/ && ssh -t home 'bash /tmp/setup_home_server.sh'
#   ... 'LISTEN_ALL=1 bash /tmp/setup_home_server.sh'   # чтобы ходить по Tailscale напрямую (см. ниже)
#
# Доступ с ноутбука, вариант 1 (по умолчанию, ничего не открывает в сеть) — SSH-туннель:
#   brew services stop ollama                    # иначе порт 11434 на ноутбуке занят своим Ollama
#   ssh -N -L 11434:localhost:11434 home         # держать открытым во время прогона
#   в .env: OLLAMA_BASE_URL=http://localhost:11434
# Вариант 2 — Tailscale напрямую: LISTEN_ALL=1 (Ollama слушает 0.0.0.0:11434; у Ollama нет паролей,
#   поэтому порт должен быть закрыт файрволом для всего, кроме Tailscale), в .env: OLLAMA_BASE_URL=http://100.65.17.12:11434
#
# После установки: .venv/bin/python -m src.server_info --write   (железо сервера → results/run_manifest.json)
set -euo pipefail

MODEL="qwen3:8b"
EXPECTED_ID="500a1f067a9f"   # ID qwen3:8b на ноутбуке (шаг 01): на сервере должны быть те же веса

# На 2026-10-10 на сервере стоял Ollama 0.1.41: он не знает qwen3 и параметр think, поэтому обновляем.
# Официальный скрипт ставит свежую версию поверх старой и создаёт или обновляет systemd-сервис ollama.
OLD_VERSION="$(ollama --version 2>/dev/null | grep -o '[0-9][0-9.]*' | tail -1 || true)"
echo "== Ollama сейчас: ${OLD_VERSION:-не установлен}; ставлю актуальную версию"
curl -fsSL https://ollama.com/install.sh | sh

# qwen3:8b занимает около 5,2 ГБ.
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 8 ]; then
    echo "ВНИМАНИЕ: свободно ${FREE_GB} ГБ; старые модели можно удалить так: ollama list, затем ollama rm <имя>"
fi

if [ "${LISTEN_ALL:-0}" = "1" ]; then
    echo "== Ollama будет слушать 0.0.0.0:11434"
    sudo mkdir -p /etc/systemd/system/ollama.service.d
    printf '[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0:11434"\n' \
        | sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null
    sudo systemctl daemon-reload
fi
sudo systemctl enable ollama
sudo systemctl restart ollama

# Ждём, пока сервис ответит (до 30 с).
for _ in $(seq 1 30); do
    curl -fs http://localhost:11434/api/version >/dev/null && break
    sleep 1
done
curl -fs http://localhost:11434/api/version || { echo "Ollama не отвечает: journalctl -u ollama"; exit 1; }
echo

echo "== Скачивание $MODEL"
ollama pull "$MODEL"

echo "== Проверка"
ollama --version
ollama list
if ollama list | grep -q "$EXPECTED_ID"; then
    echo "ID модели $EXPECTED_ID совпадает с ноутбуком"
else
    echo "ВНИМАНИЕ: ID $MODEL отличается от ноутбука ($EXPECTED_ID): записать в отчёт, какие веса использовались"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "GPU NVIDIA не найден: модель пойдёт на CPU (обычно заметно медленнее, чем GPU ноутбука)"
echo
# Модель здесь не вызываем: все вызовы LLM идут только через src/llm.py::ask и пишутся в журнал.
echo "Дальше на ноутбуке: OLLAMA_BASE_URL в .env, затем .venv/bin/python -m src.smoke_test"
echo "(tok_out и latency_s строки qwen3-8b показывают скорость сервера) и .venv/bin/python -m src.server_info --write"
