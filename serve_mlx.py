"""Run the MLX LM server while recording its own allocator and process memory."""

import argparse
import json
import logging
import os
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Any


def memory_snapshot(mx: Any) -> dict[str, Any]:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "scope": "server lifetime, including model loading; not per document",
        "mlx_active_bytes": mx.get_active_memory(),
        "mlx_cache_bytes": mx.get_cache_memory(),
        "mlx_peak_active_bytes": mx.get_peak_memory(),
        "process_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
    }


def monitor(mx: Any, report: Path, interval: float, stopped: Event) -> None:
    history = report.with_suffix(".jsonl")
    report.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    while True:
        snapshot = memory_snapshot(mx)
        snapshot["server_seconds"] = perf_counter() - started
        temporary = report.with_name(report.name + ".tmp")
        temporary.write_text(json.dumps(snapshot) + "\n", encoding="utf-8")
        temporary.replace(report)
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(snapshot) + "\n")
        if stopped.wait(interval):
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--memory-report", type=Path, required=True)
    parser.add_argument("--memory-interval", type=float, default=5.0)
    args, server_args = parser.parse_known_args()
    if args.memory_interval <= 0:
        parser.error("--memory-interval must be positive")
    import mlx.core as mx
    from mlx_lm import server

    stopped = Event()
    worker = Thread(
        target=monitor,
        args=(mx, args.memory_report, args.memory_interval, stopped),
        daemon=True,
    )
    worker.start()
    sys.argv = [sys.argv[0], *server_args]
    try:
        server.main()
    finally:
        stopped.set()
        worker.join(timeout=10)
        logging.info("MLX memory history saved to %s", args.memory_report.with_suffix(".jsonl"))


if __name__ == "__main__":
    main()
