#!/usr/bin/env bash
# Проверка воспроизводимости (план, шаг 9.5): чистое окружение из requirements.txt → Restart & Run All
# analysis.ipynb → results/summary_metrics.json совпадает с исходным.
#
#   scripts/check_reproducibility.sh          # реальные данные (после scripts/run_analysis.sh)
#   scripts/check_reproducibility.sh --demo   # то же на demo/ (mock): проверка самой процедуры, не результатов
#
# Файлы проекта не меняются: проект копируется во временную папку, там создаётся новое .venv,
# ноутбук выполняется заново (scripts/run_analysis.sh той копии), затем JSON сравнивается с исходным.
# Код выхода: 0 — совпало; 1 — различается; 2 — нечего проверять; 3 — ноутбук в копии упал.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-}"
if [[ "$MODE" == "--demo" ]]; then SUMMARY=demo/results/summary_metrics.json; else SUMMARY=results/summary_metrics.json; fi
if [[ ! -f "$SUMMARY" ]]; then
  echo "Нет $SUMMARY: сначала scripts/run_analysis.sh $MODE"
  exit 2
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/lab3_repro.XXXXXX")
echo "== Временная папка: $TMP"

# 1. Копия проекта без окружения, ключей и служебных файлов флоу.
rsync -a --exclude '.venv' --exclude '.env' --exclude '__pycache__/' --exclude 'flow/state/' --exclude 'flow/logs/' \
  ./ "$TMP/proj/"
# Исходный JSON убираем из копии: ноутбук должен создать его заново.
mv "$TMP/proj/$SUMMARY" "$TMP/summary_original.json"

# 2. Чистое окружение на Python 3.12 из requirements.txt (так же, как в README).
echo "== Окружение из requirements.txt"
if command -v uv >/dev/null 2>&1; then
  uv venv --seed --python 3.12 "$TMP/proj/.venv" -q
  uv pip install -q --python "$TMP/proj/.venv/bin/python" -r requirements.txt
else
  python3.12 -m venv "$TMP/proj/.venv"
  "$TMP/proj/.venv/bin/python" -m pip install -q -r requirements.txt
fi
"$TMP/proj/.venv/bin/python" --version

# 3. Restart & Run All ноутбука в копии.
echo "== Выполнение analysis.ipynb"
if ! (cd "$TMP/proj" && scripts/run_analysis.sh $MODE); then
  echo "== Итог: ноутбук в копии не выполнился (это сбой, а не расхождение); копия для разбора: $TMP"
  exit 3
fi

# 4. Сравнение: побайтно и по ключам верхнего уровня (чтобы было видно, что именно разошлось).
echo "== Сравнение $SUMMARY"
set +e   # код выхода сравнения нужен ниже, не прерываем скрипт
"$TMP/proj/.venv/bin/python" - "$TMP/summary_original.json" "$TMP/proj/$SUMMARY" <<'PYEOF'
import json, sys
a, b = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:3])
same_bytes = open(sys.argv[1], "rb").read() == open(sys.argv[2], "rb").read()
diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
print("Побайтно совпадает:", "да" if same_bytes else "нет")
print("Различаются ключи:", ", ".join(diff) if diff else "нет")
sys.exit(0 if same_bytes and not diff else 1)
PYEOF
RC=$?
set -e
echo "== Итог: $([[ $RC -eq 0 ]] && echo 'совпадает' || echo 'НЕ совпадает'); копия для разбора: $TMP"
exit $RC
