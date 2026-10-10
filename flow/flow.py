#!/usr/bin/env python3
"""Оркестратор агентского флоу ЛР3.

Каждый шаг выполняет отдельная сессия Claude Code (opus, effort high, bypass permissions),
по умолчанию в новом окне Terminal. Контекст между шагами передаётся через handoff-файлы
flow/context/<шаг>.md. Задачи для человека копятся в HUMAN_TODO.md.

Статусы шагов:
  done      — шаг выполнен полностью;
  deferred  — сделано всё, что можно без ключей API или без человека; при следующем `run` шаг перезапустится;
  blocked / failed — флоу останавливается, причина записана в handoff.
Гейты G1–G3 по умолчанию мягкие: если проверка гейта не проходит, флоу печатает,
что осталось сделать человеку, и идёт дальше. С флагом --strict-gates флоу на них останавливается.

Команды:
  python3 flow/flow.py status
  python3 flow/flow.py run [--mode window|interactive|inline] [--only ID] [--until ID] [--strict-gates]
  python3 flow/flow.py approve G1 [--force]
  python3 flow/flow.py reset ID [--yes]
  python3 flow/flow.py prompt ID

Переменные окружения: FLOW_MODEL (по умолчанию claude-opus-5-5), FLOW_EFFORT (high),
FLOW_CLAUDE_ARGS (дополнительные аргументы claude, например "--max-budget-usd 20").
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLOW = ROOT / "flow"
STEPS_DIR = FLOW / "steps"
CONTEXT_DIR = FLOW / "context"
STATE_DIR = FLOW / "state"
LOG_DIR = FLOW / "logs"
HUMAN_TODO = ROOT / "HUMAN_TODO.md"
HUMAN_NOTES = FLOW / "HUMAN_NOTES.md"

MODEL = os.environ.get("FLOW_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("FLOW_EFFORT", "high")
EXTRA_ARGS = shlex.split(os.environ.get("FLOW_CLAUDE_ARGS", ""))
PY = ".venv/bin/python"

PIPELINE = [
    {"id": "01", "slug": "setup", "title": "Окружение и каркас репозитория"},
    {"id": "02", "slug": "llm_client", "title": "Клиент LLM: ask(), логирование, кэш, smoke-тест"},
    {"id": "03", "slug": "bank_tooling", "title": "Банк задач: схема, валидатор, черновики, шаблон гипотез"},
    {"id": "04", "slug": "prompt_parser", "title": "Промпты, парсер и автопроверка ответов"},
    {"id": "05", "slug": "runners", "title": "Прогонщики: банк, гипотезы, исполнение кода D"},
    {"id": "06", "slug": "typeE_prep", "title": "Тип E: AI4I 2020, разбиение, классические модели"},
    {"id": "G1", "gate": True, "title": "Человек: ключи, модели, гипотезы, банк задач",
     "check": [PY, "-m", "src.checks", "g1"]},
    {"id": "07", "slug": "prompt_debug", "title": "Отладка промпта на dev-задачах и заморозка"},
    {"id": "08", "slug": "main_runs", "title": "Основной прогон и прогоны под гипотезы"},
    {"id": "09", "slug": "typeE_runs", "title": "Тип E: прогон LLM-классификатора"},
    {"id": "10", "slug": "grading_prep", "title": "Автопроверка, слепые листы, тарифы"},
    {"id": "G2", "gate": True, "title": "Человек: слепая проверка, сверка парсера и тарифов",
     "check": [PY, "-m", "src.checks", "g2"]},
    {"id": "11", "slug": "analysis", "title": "Ноутбук анализа: метрики, CI, тесты, графики"},
    {"id": "12", "slug": "report_draft", "title": "Каркас отчёта и проверка воспроизводимости"},
    {"id": "13", "slug": "qa_audit", "title": "Аудит: соответствие заданию, корректность, исправления"},
    {"id": "G3", "gate": True, "title": "Человек: выводы, разбор ошибок, отчёт, защита"},
]
BY_ID = {s["id"]: s for s in PIPELINE}
AGENT_STEPS = [s for s in PIPELINE if not s.get("gate")]

ICONS = {"done": "✅", "deferred": "⏸ ", "blocked": "⛔", "failed": "❌", "todo": "··",
         "unknown": "❓", "approved": "✅", "pending": "🧑"}


# ---------- состояние ----------

def handoff_path(step: dict) -> Path:
    return CONTEXT_DIR / f"{step['id']}_{step['slug']}.md"


def exit_path(step: dict) -> Path:
    return STATE_DIR / f"{step['id']}.exit"


def running_path(step: dict) -> Path:
    return STATE_DIR / f"{step['id']}.running"


def approved_path(gate: dict) -> Path:
    return STATE_DIR / f"{gate['id']}.approved"


def step_status(step: dict) -> str:
    if step.get("gate"):
        return "approved" if approved_path(step).exists() else "pending"
    path = handoff_path(step)
    if not path.exists():
        return "todo"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            if line.startswith("STATUS:"):
                value = line.split(":", 1)[1].strip().lower().split()[0]
                return value if value in ICONS else "unknown"
            return "unknown"
    return "unknown"


def is_running(step: dict) -> bool:
    path = running_path(step)
    if not path.exists():
        return False
    try:
        os.kill(int(path.read_text().strip()), 0)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        path.unlink(missing_ok=True)
        return False


# ---------- промпт ----------

HEADER = """\
Ты — агент-исполнитель шага {id} «{title}» (шаг {n} из {total}) в агентском флоу выполнения ЛР3 \
«Можно ли доверять LLM в задачах классического ML». Каждый шаг выполняет отдельная сессия Claude Code. \
Прошлых сессий ты не видишь. Тебе доступны только их передаточные записи (handoff) ниже, файлы проекта и память.

Цель всего флоу — почти готовая лабораторная. Человеку должно остаться только вписать ключи API и выполнить \
пункты из HUMAN_TODO.md (решения, эталоны, слепая проверка, выводы). После этого повторный запуск флоу \
доделывает отложенные шаги.

## Порядок работы
1. Прочитай `CLAUDE.md` (правила преподавателя, они обязательны) и `PLAN.md`. Задание лежит в `ЛР LLM и AutoML.pdf`, стр. 1–9.
2. Изучи handoff-записи и заметки человека ниже. Сверь их с реальным состоянием файлов: записи могут быть неточными.
3. Выполни задачу шага (раздел «Задача шага»). Перед тем как писать handoff, проверь результат: запусти тесты и команды.

## Правила
- {autonomy}
- Если нет ключей API или ещё не выполнены пункты человека, доведи до готовности всё остальное: код, тесты на mock, \
команды запуска. Демо-результаты пиши только в `demo/`. Затем заверши шаг со `STATUS: deferred` и точно укажи, \
что нужно для его завершения. При повторном запуске шаг продолжит с этого места.
- Используй субагентов (Agent tool) для независимых подзадач: параллельная реализация модулей и тестов, ревью своего кода, \
сверка с актуальной документацией API (WebSearch/WebFetch). Не запускай их без нужды.
- В память (memory) сохраняй только устойчивые неочевидные факты, которые пригодятся будущим шагам: особенности API провайдеров, \
грабли окружения. Состояние работы записывай не в память, а в handoff.
- Не выходи за рамки шага. Работу, отмеченную как работа человека, не делай. Вместо неё оставь заготовку и пункт в HUMAN_TODO.md.
- В конце закоммить результат шага локально: `git add -A && git commit -m "step {id}: <кратко>"`. Без push. \
Проверь, что `.env` и `demo/` не попали в коммит.
"""

AUTONOMY_HEADLESS = (
    "Работай автономно и не задавай вопросов: человека рядом нет. Если нужно решение человека, прими разумное "
    "обратимое допущение, запиши его в handoff, а подтверждение вынеси в HUMAN_TODO.md."
)
AUTONOMY_INTERACTIVE = (
    "Человек наблюдает за сессией и может ответить, но работай автономно. Вопрос задавай, только если без ответа "
    "дальше никак. Решения, которые можно отложить, оформляй как допущение и пункт в HUMAN_TODO.md."
)

FOOTER = """\
## Завершение шага (обязательно)
1. HUMAN_TODO.md: добавь пункты в раздел нужного гейта (G1, G2 или G3). Если пункт не привязан к гейту, добавь его \
в «Добавлено агентами». Формат: `- [ ] (шаг {id}) <что сделать> — <где и как> — <как проверить, что сделано>`. \
Существующие пункты не дублируй, устаревшие поправь.
2. Перезапиши целиком файл `{handoff}` строго по шаблону:

```
STATUS: done            # первая строка; одно из значений: done | deferred | blocked | failed
## Сделано
## Ключевые файлы и точки входа
## Решения и допущения
## Проверка
## Что нужно для завершения      # для deferred/blocked: ключи, пункты HUMAN_TODO, команды
## Открытые вопросы и риски
## Для следующих шагов
```
Пиши сжато, не больше 60 строк: это единственный канал контекста для следующих агентов. Без секретов.
3. Сделай коммит (см. правила); handoff тоже коммитится.
"""


def build_prompt(step: dict, interactive: bool) -> str:
    idx = PIPELINE.index(step)
    parts = [HEADER.format(id=step["id"], title=step["title"], n=AGENT_STEPS.index(step) + 1,
                           total=len(AGENT_STEPS),
                           autonomy=AUTONOMY_INTERACTIVE if interactive else AUTONOMY_HEADLESS)]

    plan = "\n".join(
        f"- {s['id']} {ICONS[step_status(s)]} {step_status(s):9} {s['title']}" + ("  ← ты здесь" if s is step else "")
        for s in PIPELINE)
    parts.append(f"## Флоу целиком\n{plan}")

    prev = [s for s in PIPELINE[:idx] if not s.get("gate") and handoff_path(s).exists()]
    if prev:
        ctx = "\n\n".join(f"### Шаг {s['id']} — {s['title']}\n\n{handoff_path(s).read_text(encoding='utf-8').strip()}"
                          for s in prev)
    else:
        ctx = "(это первый шаг, предыдущих записей нет)"
    parts.append(f"## Handoff предыдущих шагов\n\n{ctx}")

    if handoff_path(step).exists():
        parts.append("## Прошлая попытка этого шага\nШаг уже запускался. Продолжи с того места, где остановилась "
                     "прошлая попытка, а не начинай заново.\n\n" + handoff_path(step).read_text(encoding="utf-8").strip())

    gates = [s for s in PIPELINE[:idx] if s.get("gate")]
    if gates:
        lines = []
        for g in gates:
            state = f"пройден {approved_path(g).read_text().strip()}" if approved_path(g).exists() else \
                "НЕ пройден: пункты человека не выполнены, проверь, какие из нужных тебе готовы"
            lines.append(f"- {g['id']} ({g['title']}): {state}")
        parts.append("## Гейты человека\n" + "\n".join(lines))

    if HUMAN_NOTES.exists():
        parts.append("## Заметки человека (flow/HUMAN_NOTES.md)\n\n" + HUMAN_NOTES.read_text(encoding="utf-8").strip())

    task_file = STEPS_DIR / f"{step['id']}_{step['slug']}.md"
    parts.append("## Задача шага\n\n" + task_file.read_text(encoding="utf-8").strip())

    footer = FOOTER.format(id=step["id"], handoff=handoff_path(step).relative_to(ROOT))
    if interactive:
        footer += "4. Сообщи человеку, что шаг завершён и можно ввести /exit.\n"
    parts.append(footer)
    return "\n\n".join(parts) + "\n"


# ---------- запуск сессии ----------

def dim(s: str) -> str: return f"\033[2m{s}\033[0m"
def cyan(s: str) -> str: return f"\033[36m{s}\033[0m"
def bold(s: str) -> str: return f"\033[1m{s}\033[0m"


def summarize_input(inp: dict) -> str:
    for key in ("command", "file_path", "description", "pattern", "url", "query", "skill", "path"):
        if key in inp:
            text = str(inp[key])
            break
    else:
        text = json.dumps(inp, ensure_ascii=False)
    text = text.replace("\n", " ⏎ ")
    return text[:160] + ("…" if len(text) > 160 else "")


def render_event(line: str) -> None:
    """Печатает событие stream-json в человекочитаемом виде."""
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        if line.strip():
            print(line.rstrip())
        return
    kind = ev.get("type")
    pad = "    ↳ " if ev.get("parent_tool_use_id") else ""
    if kind == "system" and ev.get("subtype") == "init":
        print(dim(f"[init] model={ev.get('model')} session={ev.get('session_id')}"))
    elif kind == "assistant":
        for block in ev.get("message", {}).get("content", []):
            if block.get("type") == "text" and block.get("text", "").strip():
                print(pad + block["text"].strip())
            elif block.get("type") == "tool_use":
                print(pad + cyan(f"→ {block.get('name')}: {summarize_input(block.get('input') or {})}"))
    elif kind == "result":
        cost = ev.get("total_cost_usd")
        mins = (ev.get("duration_ms") or 0) / 60000
        print(bold(f"[результат] {ev.get('subtype')} · ходов {ev.get('num_turns')} · {mins:.1f} мин"
                   + (f" · ${cost:.2f}" if cost else "")))
    sys.stdout.flush()


def claude_cmd(step: dict) -> list[str]:
    return ["claude", "--model", MODEL, "--effort", EFFORT, "--dangerously-skip-permissions",
            "-n", f"lab3-{step['id']}-{step['slug']}", *EXTRA_ARGS]


RATE_LIMIT_RETRIES = 10


def run_headless(step: dict, prompt: str) -> tuple[int, float | None]:
    """Одна headless-сессия. Возвращает (код выхода, время сброса лимита или None, если лимит не мешал)."""
    log_path = LOG_DIR / f"{step['id']}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    cmd = [*claude_cmd(step), "-p", "--output-format", "stream-json", "--verbose", prompt]
    reset_at, hit_limit = None, False
    with open(log_path, "w", encoding="utf-8") as log, subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            render_event(line)
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            info = ev.get("rate_limit_info") or {}
            if ev.get("type") == "rate_limit_event" and info.get("status") == "rejected":
                reset_at = info.get("resetsAt")
            if ev.get("type") == "result" and ev.get("api_error_status") == 429:
                hit_limit = True
    print(dim(f"Полный лог: {log_path.relative_to(ROOT)}"))
    if hit_limit:
        return proc.returncode, float(reset_at) if reset_at else time.time() + 15 * 60
    return proc.returncode, None


def exec_step(step: dict, mode: str) -> int:
    """Выполняет шаг в текущем терминале (сюда попадает и новое окно Terminal)."""
    for d in (STATE_DIR, LOG_DIR, CONTEXT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    exit_path(step).unlink(missing_ok=True)
    running_path(step).write_text(str(os.getpid()))
    interactive = mode == "interactive"
    prompt = build_prompt(step, interactive)
    (STATE_DIR / f"{step['id']}.prompt.md").write_text(prompt, encoding="utf-8")
    print(bold(f"=== Шаг {step['id']}: {step['title']} ===  ({MODEL}, effort {EFFORT}, {mode})"))
    code = 1
    try:
        if interactive:
            code = subprocess.run([*claude_cmd(step), prompt], cwd=ROOT).returncode
        else:
            for attempt in range(1, RATE_LIMIT_RETRIES + 2):
                code, reset_at = run_headless(step, prompt)
                if reset_at is None:
                    break
                # Упёрлись в лимит подписки Claude: ждём сброса и запускаем шаг заново
                # (новая сессия увидит уже сделанное по файлам и git).
                wait_s = max(60, reset_at - time.time() + 60)
                print(bold(f"Лимит Claude исчерпан. Жду сброса до {datetime.fromtimestamp(reset_at + 60):%H:%M} "
                           f"(попытка {attempt}/{RATE_LIMIT_RETRIES})."))
                time.sleep(wait_s)
                prompt = build_prompt(step, interactive)
    finally:
        running_path(step).unlink(missing_ok=True)
        exit_path(step).write_text(str(code))
    status = step_status(step)
    print(bold(f"=== Шаг {step['id']} завершён: {ICONS.get(status, '')} {status} (код {code}) ==="))
    if mode != "inline":
        print("Окно можно закрыть.")
    return code


def open_terminal_window(step: dict, mode: str) -> None:
    inner = (f"cd {shlex.quote(str(ROOT))} && {shlex.quote(sys.executable)} flow/flow.py _exec "
             f"{step['id']} --mode {mode}")
    script = inner.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run(["osascript", "-e", 'tell application "Terminal"', "-e", "activate",
                    "-e", f'do script "{script}"', "-e", "end tell"], check=True, stdout=subprocess.DEVNULL)


def wait_for_exit(step: dict) -> int:
    path = exit_path(step)
    start = time.time()
    try:
        while not path.exists():
            time.sleep(5)
            mins = int((time.time() - start) // 60)
            print(f"\r  ⏳ шаг {step['id']} выполняется в отдельном окне… {mins} мин", end="", flush=True)
    except KeyboardInterrupt:
        print("\nОжидание прервано, сессия в окне продолжает работать. Когда она закончит, снова запустите "
              "`python3 flow/flow.py run`.")
        sys.exit(130)
    print()
    try:
        return int(path.read_text().strip())
    except ValueError:
        return 1


def launch(step: dict, mode: str) -> int:
    if is_running(step):
        print(f"Шаг {step['id']} уже выполняется, жду завершения.")
        return wait_for_exit(step)
    exit_path(step).unlink(missing_ok=True)
    if mode == "inline" or sys.platform != "darwin":
        return exec_step(step, "inline")
    open_terminal_window(step, "interactive" if mode == "interactive" else "headless")
    return wait_for_exit(step)


# ---------- гейты ----------

def todo_section(gate_id: str) -> str:
    if not HUMAN_TODO.exists():
        return ""
    out, inside = [], False
    for line in HUMAN_TODO.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            inside = line[3:].startswith(gate_id)
        if inside:
            out.append(line)
    return "\n".join(out)


def run_gate_check(gate: dict, quiet: bool = False) -> bool:
    check = gate.get("check")
    if not check:
        return False
    try:
        res = subprocess.run(check, cwd=ROOT, capture_output=quiet, text=True)
    except FileNotFoundError as e:
        if not quiet:
            print(f"Проверку не удалось запустить: {e}")
        return False
    return res.returncode == 0


def approve(gate: dict, note: str = "") -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    approved_path(gate).write_text(f"{datetime.now():%Y-%m-%d %H:%M}{note}\n")


# ---------- команды ----------

def cmd_status(_args) -> int:
    for s in PIPELINE:
        st = step_status(s)
        extra = "  (выполняется)" if not s.get("gate") and is_running(s) else ""
        print(f"{s['id']:>3} {ICONS.get(st, '')} {st:9} {s['title']}{extra}")
    open_items = sum(1 for line in HUMAN_TODO.read_text(encoding="utf-8").splitlines()
                     if line.strip().startswith("- [ ]")) if HUMAN_TODO.exists() else 0
    print(f"\nОткрытых пунктов для человека в HUMAN_TODO.md: {open_items}")
    return 0


def cmd_run(args) -> int:
    if args.only and (args.only not in BY_ID or BY_ID[args.only].get("gate")):
        print(f"--only принимает id агентского шага, а не {args.only}")
        return 2
    pending_gates = []
    for step in PIPELINE:
        if args.only and step["id"] != args.only:
            continue
        status = step_status(step)
        if step.get("gate"):
            if status == "approved":
                continue
            if run_gate_check(step, quiet=True):
                approve(step, " (проверка пройдена автоматически)")
                print(f"🧑 {step['id']}: проверка пройдена, гейт отмечен.")
                continue
            pending_gates.append(step["id"])
            print(f"\n🧑 Гейт {step['id']} — {step['title']}: работа человека ещё не выполнена.")
            if args.strict_gates or not step.get("check"):
                print(todo_section(step["id"]))
                print(f"\nКогда закончите: python3 flow/flow.py approve {step['id']} && python3 flow/flow.py run")
                if args.strict_gates:
                    return 3
            else:
                print("   Иду дальше (мягкий гейт). Шаги, которым он нужен, будут отложены (deferred).")
            continue
        if status == "done" and not args.only:
            continue
        print(f"\n▶ Шаг {step['id']}: {step['title']}" + (f" (повтор, был {status})" if status != "todo" else ""))
        launch(step, args.mode)
        status = step_status(step)
        print(f"{ICONS.get(status, '')} Шаг {step['id']}: {status}")
        if status not in ("done", "deferred"):
            print(f"Флоу остановлен. Причина записана в {handoff_path(step).relative_to(ROOT)}. "
                  f"Исправьте её и снова запустите `python3 flow/flow.py run`.")
            return 1
        if args.until == step["id"]:
            break
    deferred = [s["id"] for s in AGENT_STEPS if step_status(s) == "deferred"]
    print("\nПроход завершён.")
    if deferred:
        print(f"Отложенные шаги: {', '.join(deferred)}. Выполните пункты HUMAN_TODO.md "
              f"(гейты: {', '.join(pending_gates) or '—'}) и снова запустите `python3 flow/flow.py run`.")
    return 0


def cmd_approve(args) -> int:
    gate = BY_ID.get(args.gate)
    if not gate or not gate.get("gate"):
        print(f"{args.gate} не является гейтом")
        return 2
    if gate.get("check") and not args.force:
        print("Проверка гейта:", " ".join(gate["check"]))
        if not run_gate_check(gate):
            print("Проверка не пройдена. Исправьте замечания или используйте --force.")
            return 1
    approve(gate, " (force)" if args.force else "")
    print(f"Гейт {gate['id']} отмечен как пройденный.")
    return 0


def cmd_reset(args) -> int:
    if args.id not in BY_ID:
        print(f"Нет шага {args.id}")
        return 2
    targets = PIPELINE[PIPELINE.index(BY_ID[args.id]):]
    files = []
    for s in targets:
        files += [approved_path(s)] if s.get("gate") else [handoff_path(s), exit_path(s)]
    files = [f for f in files if f.exists()]
    if not args.yes:
        print("Будут удалены (файлы проекта и git не трогаются):")
        print("\n".join(f"  {f.relative_to(ROOT)}" for f in files) or "  (нечего удалять)")
        print("Повторите с --yes.")
        return 0
    for f in files:
        f.unlink()
    print(f"Сброшено: {len(files)} файлов состояния.")
    return 0


def cmd_prompt(args) -> int:
    step = BY_ID.get(args.id)
    if not step or step.get("gate"):
        print(f"Нет агентского шага {args.id}")
        return 2
    print(build_prompt(step, interactive=False))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Агентский флоу ЛР3")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=["window", "interactive", "inline"], default="window",
                   help="window: headless-сессия в новом окне Terminal (по умолчанию); "
                        "interactive: интерактивная сессия в новом окне; inline: в текущем терминале")
    r.add_argument("--only", help="выполнить только этот шаг (даже если он done)")
    r.add_argument("--until", help="остановиться после этого шага")
    r.add_argument("--strict-gates", action="store_true", help="останавливаться на непройденных гейтах")
    a = sub.add_parser("approve")
    a.add_argument("gate")
    a.add_argument("--force", action="store_true")
    rs = sub.add_parser("reset")
    rs.add_argument("id")
    rs.add_argument("--yes", action="store_true")
    pr = sub.add_parser("prompt")
    pr.add_argument("id")
    ex = sub.add_parser("_exec")
    ex.add_argument("id")
    ex.add_argument("--mode", choices=["headless", "interactive", "inline"], default="headless")
    args = p.parse_args()

    if args.cmd == "_exec":
        exec_step(BY_ID[args.id], args.mode)
        return 0
    return {"status": cmd_status, "run": cmd_run, "approve": cmd_approve,
            "reset": cmd_reset, "prompt": cmd_prompt}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
