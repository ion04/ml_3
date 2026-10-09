"""Smoke-тест: один короткий запрос к каждой включённой модели (шаг 1 хода работы из PDF).

Запуск: .venv/bin/python -m src.smoke_test
Код выхода 0, только если ответили все включённые модели (mock не проверяется).
Вызовы пишутся в logs/calls.jsonl с meta.experiment = "smoke".
"""

import sys

from src import llm

PROMPT = "Сколько будет 2 + 2? Ответь одним числом."
MAX_TOKENS = 50


def main() -> int:
    models = [m for m in llm.load_models().values() if m["enabled"] and m["provider"] != "mock"]
    rows = []
    for cfg in models:
        missing = llm.missing_env(cfg)
        if missing:
            # Без ключа запрос не отправляем и в журнал не пишем: это не вызов модели.
            rows.append([cfg["key"], cfg["role"], "-", "-", "-", "-", "нет переменных: " + ", ".join(missing)])
            continue
        print(f"→ {cfg['key']} ...", flush=True)
        # use_cache=False: smoke-тест проверяет связь сейчас, а не берёт старый ответ из журнала.
        res = llm.ask(cfg["key"], PROMPT, 0.0, max_tokens=MAX_TOKENS, use_cache=False,
                      meta={"experiment": "smoke", "task_id": "smoke"})
        if res.error is None and res.text:
            answer = " ".join(res.text.split())[:40]
            status = f"ok: «{answer}»"
            if cfg["provider"] == "ollama":
                ps = llm.ollama_ps(cfg["base_url"], llm._expand_env(cfg["model"]))
                if ps:
                    status += f" | в памяти {ps['size_mb']:.0f} МБ, на GPU {ps['size_vram_mb']:.0f} МБ"
                status += f" | сервер {cfg['base_url']}"
        else:
            status = "ОШИБКА: " + (res.error or "пустой ответ")[:200]
        rows.append([cfg["key"], cfg["role"], res.model_reported or "-",
                     res.tokens_in if res.tokens_in is not None else "-",
                     res.tokens_out if res.tokens_out is not None else "-",
                     f"{res.latency_s:.2f}" if res.latency_s is not None else "-", status])

    header = ["model_key", "role", "model_reported", "tok_in", "tok_out", "latency_s", "статус"]
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header) - 1)]
    print()
    for r in [header] + rows:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)) + "  " + str(r[-1]))

    ok = sum(str(r[-1]).startswith("ok") for r in rows)
    print(f"\nОтветили {ok} из {len(rows)} включённых моделей.")
    return 0 if rows and ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
