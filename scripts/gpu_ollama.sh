#!/usr/bin/env bash
# Локальная модель считается на ПК `ssh gpu` (Windows, RTX 3090), а не на ноутбуке.
#
# Скрипт запускает там `ollama serve` внутри ssh-сессии и той же сессией пробрасывает порт:
#   ноутбук http://localhost:11435  →  ПК 127.0.0.1:11434
# Права администратора и правило файрвола на ПК не нужны. Ollama работает, пока жива ssh-сессия.
# При обрыве (ПК перезагрузился, сеть пропала) скрипт переподключается через 10 с.
#
# Запуск:     nohup scripts/gpu_ollama.sh >> flow/logs/gpu_ollama.log 2>&1 &
# В .env:     OLLAMA_BASE_URL=http://localhost:11435
# Проверка:   curl -s localhost:11435/api/version
# Остановка:  pkill -f gpu_ollama.sh; pkill -f "11435:127.0.0.1:11434"

LOCAL_PORT="${LOCAL_PORT:-11435}"
OLLAMA_EXE='C:\Users\616\ollama\ollama.exe'

# Если ollama на ПК уже слушает порт (осталась от прошлой сессии), просто держим туннель открытым.
REMOTE_CMD="netstat -ano | findstr LISTENING | findstr 127.0.0.1:11434 >nul && (echo ollama already running& ping -n 2147483647 127.0.0.1 >nul) || (set OLLAMA_HOST=127.0.0.1:11434&& set OLLAMA_KEEP_ALIVE=30m&& ${OLLAMA_EXE} serve)"

while true; do
  echo "$(date '+%F %T') подключаюсь к gpu, порт ${LOCAL_PORT}"
  ssh -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
      -L "${LOCAL_PORT}:127.0.0.1:11434" gpu "${REMOTE_CMD}"
  echo "$(date '+%F %T') соединение с gpu оборвалось (код $?), переподключаюсь через 10 с"
  sleep 10
done
