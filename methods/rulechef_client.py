"""Completion transports used during RuleChef fitting, never during inference."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from methods.rulechef import RuleChefConfig


class CodexClient:
    """Translate RuleChef's completion calls to the authenticated Codex CLI.

    Each request is independent and runs outside the repository with shell,
    apps, web search and delegation disabled. Save and audit every event.
    """

    def __init__(self, config: RuleChefConfig):
        self.config = config
        self.calls: list[dict[str, Any]] = []
        self.failure: str | None = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        config_file = Path.home() / ".codex/config.toml"
        defaults = tomllib.loads(config_file.read_text()) if config_file.exists() else {}
        self.model = config.model or defaults.get("model")
        self.reasoning = config.reasoning_effort or defaults.get("model_reasoning_effort", "high")

    @staticmethod
    def _bubblewrap(
        inputs: Path,
        outputs: Path,
        state: Path,
    ) -> tuple[list[str], dict[str, Any], dict[str, str]]:
        binary = shutil.which("codex")
        bwrap = shutil.which("bwrap")
        if not binary or not bwrap:
            raise RuntimeError("Isolated Codex fitting requires codex and bubblewrap (bwrap).")
        host_home = Path.home()
        host_state = Path(os.environ.get("CODEX_HOME", str(host_home / ".codex"))).resolve()
        auth = host_state / "auth.json"
        if not auth.is_file():
            raise RuntimeError("Log in to Codex before using the isolated synthesis backend.")
        state.mkdir(mode=0o700)
        shutil.copy2(auth, state / "auth.json")
        prefix = [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
        ]
        runtime = [
            "/usr",
            "/bin",
            "/lib",
            "/lib64",
            "/sbin",
            "/etc/ssl",
            "/etc/resolv.conf",
            "/etc/hosts",
            "/etc/nsswitch.conf",
            "/etc/passwd",
            "/etc/group",
            "/etc/gai.conf",
            "/etc/ld.so.cache",
        ]
        for path in runtime:
            if Path(path).exists():
                prefix += ["--ro-bind", path, path]
        prefix += [
            "--ro-bind",
            str(Path(binary).resolve()),
            "/opt/rulechef-codex",
            "--ro-bind",
            str(inputs),
            "/workspace",
            "--bind",
            str(outputs),
            "/output",
            "--bind",
            str(state),
            str(host_state),
            "--chdir",
            "/workspace",
            "--",
        ]
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(host_home),
            "CODEX_HOME": str(host_state),
            "LANG": "C.UTF-8",
            "TERM": "dumb",
        }
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
            if key in os.environ:
                environment[key] = os.environ[key]
        project = str(Path(__file__).resolve().parents[1])
        probe = (
            "import json,pathlib; "
            f"p=pathlib.Path({project!r}); "
            "assert not p.exists(), 'Original project is visible'; "
            f"s=pathlib.Path({str(host_state)!r}); "
            "assert sorted(x.name for x in s.iterdir())==['auth.json']; "
            "print(json.dumps({'original_project_visible':False,"
            "'fresh_codex_state':True,'state_files':['auth.json']}))"
        )
        check = subprocess.run(
            [*prefix, "/usr/bin/python3", "-c", probe],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        )
        audit = json.loads(check.stdout)
        audit.update(
            isolation="bubblewrap mount/user/PID namespaces with explicit allowlist",
            runtime_readonly_mounts=[path for path in runtime if Path(path).exists()],
            model_connection_network=True,
            model_tools_disabled=True,
        )
        return prefix, audit, environment

    def create(self, **kwargs: Any) -> SimpleNamespace:
        if self.failure:
            raise RuntimeError(f"Previous completion failed: {self.failure}")
        try:
            return self._create(**kwargs)
        except Exception as error:
            self.failure = str(error)
            raise

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        index = len(self.calls) + 1
        with tempfile.TemporaryDirectory(prefix="rulechef-completion-") as temporary:
            root = Path(temporary)
            inputs, outputs = root / "input", root / "output"
            inputs.mkdir()
            outputs.mkdir()
            trace = (
                Path(self.config.trace_directory).resolve() / f"call_{index:03d}"
                if self.config.trace_directory
                else root
            )
            trace.mkdir(parents=True, exist_ok=True)
            (inputs / "AGENTS.md").write_text(
                "Return the requested JSON using only the supplied messages. "
                "Do not use tools, read files, browse, or delegate.\n"
            )
            response_file = outputs / "response.json"
            isolated = self.config.codex_isolation == "bubblewrap"
            if self.config.codex_isolation not in {"bubblewrap", "none"}:
                raise ValueError("codex_isolation must be bubblewrap or none.")
            prefix, audit, environment = (
                self._bubblewrap(inputs, outputs, root / "state")
                if isolated
                else ([], {"isolation": "none", "original_project_hidden": False}, None)
            )
            (trace / "isolation_audit.json").write_text(json.dumps(audit, indent=2))
            prompt = (
                "Act only as the completion model for RuleChef. Follow the messages below. "
                "Return only the JSON requested by those messages. Do not use tools.\n\n"
                + json.dumps(kwargs["messages"], ensure_ascii=False)
            )
            (trace / "request.json").write_text(json.dumps(kwargs, ensure_ascii=False, indent=2))
            (trace / "prompt.txt").write_text(prompt)
            command = [
                "/opt/rulechef-codex" if isolated else "codex",
                "exec",
                "--ignore-user-config",
                "--ignore-rules",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--json",
                "--cd",
                "/workspace" if isolated else str(inputs),
                "--output-last-message",
                "/output/response.json" if isolated else str(response_file),
                "-c",
                'web_search="disabled"',
                "-c",
                "features.shell_tool=false",
                "-c",
                "features.apps=false",
                "-c",
                "features.multi_agent=false",
                "-c",
                "features.skill_search=false",
                "-c",
                "features.code_mode=false",
                "-c",
                "features.code_mode_host=false",
                "-c",
                f'model_reasoning_effort="{self.reasoning}"',
            ]
            if self.model:
                command += ["--model", self.model]
            started = time.perf_counter()
            print(f"RuleChef model call {index}: {self.model or 'Codex default'}", flush=True)
            with (
                (trace / "events.jsonl").open("w") as events,
                (trace / "stderr.log").open("w") as errors,
            ):
                completed = subprocess.run(
                    [*prefix, *command, "-"],
                    input=prompt,
                    text=True,
                    stdout=events,
                    stderr=errors,
                    timeout=self.config.call_timeout_seconds,
                    check=False,
                    env=environment,
                )
            record: dict[str, Any] = {
                "index": index,
                "model": self.model,
                "reasoning_effort": self.reasoning,
                "seconds": time.perf_counter() - started,
                "returncode": completed.returncode,
                "requested_output_tokens": kwargs.get("max_completion_tokens"),
                "output_token_limit_enforced": False,
                "usage": {},
                "tool_calls": [],
                "diagnostics": [],
                "isolation": audit,
            }
            for line in (trace / "events.jsonl").read_text().splitlines():
                event = json.loads(line)
                if event.get("type") == "turn.completed":
                    for key, count in event.get("usage", {}).items():
                        record["usage"][key] = record["usage"].get(key, 0) + count
                item = event.get("item", {})
                if item.get("type") == "error":
                    record["diagnostics"].append(item.get("message", ""))
                elif event.get("type") in {"item.started", "item.completed"} and item.get(
                    "type"
                ) not in {
                    "agent_message",
                    "reasoning",
                    None,
                }:
                    record["tool_calls"].append(item.get("type"))
            self.calls.append(record)
            (trace / "metadata.json").write_text(json.dumps(record, indent=2))
            if completed.returncode or not response_file.exists():
                raise RuntimeError(f"Codex completion failed; inspect {trace / 'stderr.log'}")
            if record["tool_calls"]:
                raise RuntimeError("Synthesis attempted tools; refusing this training result.")
            content = response_file.read_text()
            json.loads(content)
            (trace / "response.json").write_text(content)
            print(f"RuleChef model call {index} finished in {record['seconds']:.1f}s", flush=True)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )


class RecordedClient:
    """Keep SDK prompts, responses, timing and token usage with the same contract."""

    def __init__(self, client: Any, config: RuleChefConfig):
        self.client = client
        self.config = config
        self.model = config.model
        self.calls: list[dict[str, Any]] = []
        self.failure: str | None = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> Any:
        if self.failure:
            raise RuntimeError(f"Previous completion failed: {self.failure}")
        index = len(self.calls) + 1
        trace = None
        if self.config.trace_directory:
            trace = Path(self.config.trace_directory) / f"call_{index:03d}"
            trace.mkdir(parents=True, exist_ok=True)
            (trace / "request.json").write_text(json.dumps(kwargs, ensure_ascii=False, indent=2))
        started = time.perf_counter()
        record: dict[str, Any] = {"index": index, "model": kwargs["model"]}
        try:
            response = self.client.chat.completions.create(**kwargs)
            usage = response.usage
            record["usage"] = usage.model_dump() if usage is not None else {}
            if trace:
                (trace / "response.json").write_text(response.choices[0].message.content)
            return response
        except Exception as error:
            self.failure = str(error)
            record["error"] = self.failure
            raise
        finally:
            record["seconds"] = time.perf_counter() - started
            self.calls.append(record)
            if trace:
                (trace / "metadata.json").write_text(json.dumps(record, indent=2))
