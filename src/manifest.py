"""Паспорт прогонов: results/run_manifest.json (шаг 08).

Запуск:
  .venv/bin/python -m src.manifest           # по logs/calls.jsonl и results/ → results/run_manifest.json
  .venv/bin/python -m src.manifest --demo    # по demo/calls.jsonl и demo/results/ → demo/results/run_manifest.json

Что записывается (только факты из журнала, таблиц и конфигов, без оценок качества):
- даты прогонов и число вызовов/ошибок по экспериментам (main, typeE, cot, ...);
- по каждой модели: идентификатор из запроса, model_reported из ответов, параметры, температуры,
  даты, число вызовов, техошибок и вызовов, так и оставшихся без ответа;
- версия и отпечаток промпта, параметры эксперимента, выбранные гипотезы и хэш их коммита;
- по каждой таблице results/*.jsonl: строк, ошибок, parse_ok; по d_exec.jsonl: запусков кода;
- версии Python и ключевых пакетов, sha256 requirements.txt, коммит HEAD на момент сборки.
Раздел ollama_server (железо ПК с локальной моделью, src.server_info) сохраняется как есть.

Файл каждый раз пересобирается целиком: он производный, первоисточник — журнал.
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from src import checks, llm, prompts, run_bank, typeE

ROOT = llm.ROOT
MANIFEST_NAME = "run_manifest.json"
PACKAGES = ["openai", "gigachat", "httpx", "scikit-learn", "numpy", "pandas", "scipy", "statsmodels"]


def call_stats(records: list[dict]) -> dict:
    """Счётчики по записям журнала. Одна запись = одна попытка вызова.

    unresolved — запросы (cache_key), у которых есть только ошибки и нет ни одного ответа:
    их дозапрашивает повторный запуск той же команды.
    """
    ok_keys = {r["cache_key"] for r in records if r["error"] is None}
    err_keys = {r["cache_key"] for r in records if r["error"] is not None}
    times = sorted(r["ts_utc"] for r in records)
    return {
        "first_utc": times[0] if times else None,
        "last_utc": times[-1] if times else None,
        "records": len(records),
        "ok": sum(r["error"] is None for r in records),
        "errors": sum(r["error"] is not None for r in records),
        "retried_requests": len(err_keys & ok_keys),     # была ошибка, потом получен ответ
        "unresolved_requests": len(err_keys - ok_keys),  # ответа так и нет
    }


def by_experiment(records: list[dict]) -> dict:
    groups = {}
    for r in records:
        groups.setdefault((r.get("meta") or {}).get("experiment", "?"), []).append(r)
    return {exp: call_stats(rs) for exp, rs in sorted(groups.items())}


def by_model(records: list[dict], models_cfg: dict) -> dict:
    """Сводка по моделям только для оценочных вызовов (без smoke и dev)."""
    groups = {}
    for r in records:
        if (r.get("meta") or {}).get("experiment") not in checks.NON_EVAL_EXPERIMENTS:
            groups.setdefault(r["model_key"], []).append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        ok = [r for r in rs if r["error"] is None]
        cfg = models_cfg.get(key, {})
        info = {
            "provider": rs[0]["provider"],
            "model_requested": sorted({r["model_requested"] for r in rs}),
            "model_reported": sorted({r["model_reported"] for r in ok if r["model_reported"]}),
            "params": sorted({json.dumps(r["params"], ensure_ascii=False, sort_keys=True) for r in rs}),
            "effective_temperature": sorted({r["effective_temperature"] for r in rs}),
            "min_temperature": cfg.get("min_temperature"),
            "max_tokens": sorted({r["max_tokens"] for r in rs}, key=str),
            "experiments": sorted({(r.get("meta") or {}).get("experiment", "?") for r in rs}),
            **call_stats(rs),
            "tokens_in": sum(r["tokens_in"] or 0 for r in ok),
            "tokens_out": sum(r["tokens_out"] or 0 for r in ok),
            "finish_reason_length": sum(r.get("finish_reason") == "length" for r in ok),
        }
        # Когда получены ответы в каждом эксперименте (без попыток с техошибкой):
        # модели могли ответить в разное время (дозапросы после недоступности API).
        periods = {}
        for r in ok:
            periods.setdefault((r.get("meta") or {}).get("experiment", "?"), []).append(r["ts_utc"])
        info["periods"] = {exp: {"first_utc": min(ts), "last_utc": max(ts)} for exp, ts in sorted(periods.items())}
        # params хранятся строками JSON (чтобы взять уникальные), возвращаем объекты.
        info["params"] = [json.loads(p) for p in info["params"]]
        if rs[0]["provider"] == "ollama":
            vram = [r["local_vram_mb"] for r in ok if r.get("local_vram_mb")]
            size = [r["local_model_size_mb"] for r in ok if r.get("local_model_size_mb")]
            info["ollama_base_url"] = sorted({r.get("ollama_base_url") or "?" for r in rs})
            info["local_vram_mb_max"] = max(vram) if vram else None
            info["local_model_size_mb_max"] = max(size) if size else None
        out[key] = info
    return out


def table_stats(results_dir: Path) -> dict:
    """Строк, техошибок и parse_ok в каждой таблице results/*.jsonl (кроме d_exec)."""
    out = {}
    for path in sorted(results_dir.glob("*.jsonl")):
        if path.name == "d_exec.jsonl":
            continue
        rows = run_bank.read_jsonl(path)
        if not rows or "error" not in rows[0]:   # таблица голосов H7 и т. п.: только число строк
            out[path.name] = {"rows": len(rows)}
            continue
        out[path.name] = {
            "rows": len(rows),
            "errors": sum(r["error"] is not None for r in rows),
            "parse_ok": sum(r.get("parse_ok") is True for r in rows),
            "models": sorted({r["model_key"] for r in rows}),
            "prompt_version": sorted({str(r.get("prompt_version")) for r in rows}),
        }
    return out


def exec_stats(path: Path) -> dict | None:
    rows = run_bank.read_jsonl(path)
    if not rows:
        return None
    return {
        "runs": len(rows),
        "code_found": sum(bool(r.get("code_found")) for r in rows),
        "pass_at_1": sum(bool(r.get("pass_at_1")) for r in rows),
        "timed_out": sum(bool(r.get("timed_out")) for r in rows),
        "first_utc": min(r["ts_utc"] for r in rows),
        "last_utc": max(r["ts_utc"] for r in rows),
    }


def environment() -> dict:
    req = ROOT / "requirements.txt"
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "requirements_sha256": hashlib.sha256(req.read_bytes()).hexdigest() if req.exists() else None,
    }


def build(log_path: Path, results_dir: Path, demo: bool) -> dict:
    cfg = run_bank.load_experiment()
    records = run_bank.read_jsonl(log_path)
    commit = checks.last_commit(checks.HYPOTHESES_MD)
    return {
        "_comment": ("ДЕМО (mock-модели): не результаты эксперимента. " if demo else "")
                    + "Собрано src.manifest по журналу и таблицам; руками не править.",
        "demo": demo,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": checks._git(["rev-parse", "HEAD"]) or None,
        "log": run_bank.show(log_path),
        "hypotheses": {
            "ids": cfg.get("hypotheses") or [],
            "commit": commit[0] if commit else None,
            "commit_time": commit[1].isoformat() if commit else None,
        },
        "prompt": {
            "version_config": cfg.get("prompt_version"),
            "hash_config": cfg.get("prompt_hash"),
            "version_code": prompts.PROMPT_VERSION,
            "hash_code": prompts.prompt_hash(),
        },
        "params": {k: cfg.get(k) for k in ["temperature_main", "sc_repeats", "sc_temperature", "languages",
                                           "seed", "bootstrap_n", "typeE"]},
        "typeE_split_hash": typeE.current_split_hash(),
        "calls_by_experiment": by_experiment(records),
        "models": by_model(records, llm.load_models()),
        "results": table_stats(results_dir),
        "d_exec": exec_stats(results_dir / "d_exec.jsonl"),
        "environment": environment(),
    }


def write(manifest: dict, path: Path) -> None:
    """Пересобирает файл, сохраняя раздел ollama_server (его пишет src.server_info)."""
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if "ollama_server" in old:
        manifest["ollama_server"] = old["ollama_server"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Паспорт прогонов → run_manifest.json")
    ap.add_argument("--demo", action="store_true", help="demo/calls.jsonl и demo/results/")
    args = ap.parse_args(argv)

    results_dir = run_bank.out_dir(args.demo)
    manifest = build(run_bank.log_path(args.demo), results_dir, args.demo)
    path = results_dir / MANIFEST_NAME
    write(manifest, path)

    print(f"Записано: {run_bank.show(path)}")
    for exp, s in manifest["calls_by_experiment"].items():
        print(f"  {exp:16} записей {s['records']:5}  ок {s['ok']:5}  ошибок {s['errors']:4}  "
              f"без ответа {s['unresolved_requests']:3}  {s['first_utc']} … {s['last_utc']}")
    for name, s in manifest["results"].items():
        extra = f"  ошибок {s['errors']:3}  parse_ok {s['parse_ok']:5}" if "errors" in s else ""
        print(f"  {name:34} строк {s['rows']:5}{extra}")
    unresolved = sum(s["unresolved_requests"] for s in manifest["calls_by_experiment"].values())
    if unresolved:
        print(f"ВНИМАНИЕ: {unresolved} запросов без ответа (техошибки). Повторите команду прогона: "
              "дозапрашиваются только они.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
