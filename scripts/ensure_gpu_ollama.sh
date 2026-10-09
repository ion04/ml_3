#!/usr/bin/env bash
# Проверка перед реальными прогонами: локальная модель должна считаться на ПК gpu, а не на ноутбуке.
#
#   scripts/ensure_gpu_ollama.sh     # код 0 — Ollama на ПК отвечает через туннель; иначе код 1
#
# Если туннель не поднят, но ПК доступен по ssh, запускает scripts/gpu_ollama.sh в фоне
# (лог: flow/logs/gpu_ollama.log) и ждёт ответа до 30 с.
set -uo pipefail
cd "$(dirname "$0")/.."

# OLLAMA_BASE_URL из .env (без него ask() пошёл бы в Ollama на ноутбуке)
URL=$(grep -E '^OLLAMA_BASE_URL=' .env 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'"' ')
URL="${URL%/}"; URL="${URL%/v1}"
if [[ -z "$URL" ]]; then
  echo "OLLAMA_BASE_URL не задан в .env: локальная модель пошла бы на ноутбуке. Нужно http://localhost:11435 (туннель к gpu)."
  exit 1
fi

if curl -s -m 3 "$URL/api/version" >/dev/null; then
  echo "Ollama отвечает: $URL ($(curl -s -m 3 "$URL/api/version"))"
  exit 0
fi

if ! ssh -o ConnectTimeout=5 -o BatchMode=yes gpu exit 2>/dev/null; then
  echo "ПК gpu недоступен по ssh, а Ollama по $URL не отвечает. Локальную модель на ноутбуке не запускаем."
  exit 1
fi

echo "Поднимаю туннель к gpu: scripts/gpu_ollama.sh (лог flow/logs/gpu_ollama.log)"
mkdir -p flow/logs
nohup scripts/gpu_ollama.sh >> flow/logs/gpu_ollama.log 2>&1 &
for _ in $(seq 1 30); do
  sleep 1
  if curl -s -m 2 "$URL/api/version" >/dev/null; then
    echo "Ollama отвечает: $URL"
    exit 0
  fi
done
echo "Ollama на gpu не ответила за 30 с, см. flow/logs/gpu_ollama.log"
exit 1
