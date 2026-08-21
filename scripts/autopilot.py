#!/usr/bin/env python3
"""Conservative, auditable maintenance checks for this repository.

This tool never reads outside the repository. Network access is opt-in and is
limited to the public upstreams declared in config/autopilot_watchlist.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
WATCHLIST = REPOSITORY / "config" / "autopilot_watchlist.yaml"
SAFE_FIX_PATHS = ("app", "tests", "scripts")
CHECK_NAMES = ("test", "lint", "typecheck")


@dataclass(frozen=True)
class Finding:
    """A single maintainer-visible result."""

    severity: str
    category: str
    message: str
    approval: str | None = None


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a repository-local command without invoking a shell."""
    return subprocess.run(command, cwd=REPOSITORY, check=False, text=True, capture_output=True)


def git_timestamp(paths: list[str]) -> int | None:
    result = run(["git", "log", "-1", "--format=%ct", "--", *paths])
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return int(result.stdout.strip())


def documentation_findings() -> list[Finding]:
    """Flag documentation that substantially predates maintained source files."""
    source_changed = git_timestamp(["app", "scripts", "config", "pyproject.toml", "Makefile"])
    docs_changed = git_timestamp(["README.md", "CONTRIBUTING.md", "docs"])
    if source_changed is None or docs_changed is None or source_changed <= docs_changed:
        return []

    days_behind = (source_changed - docs_changed) / 86_400
    if days_behind < 14:
        return []
    return [
        Finding(
            severity="warning",
            category="documentation",
            message=f"Documentation is {days_behind:.0f} days older than maintained source files.",
            approval="maintainer-review",
        )
    ]


def public_json(url: str) -> dict[str, Any]:
    """Read a small public JSON endpoint with a bounded timeout."""
    request = urllib.request.Request(url, headers={"User-Agent": "music-label-assistant-autopilot"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def upstream_findings() -> list[Finding]:
    """Compare public upstream releases with an explicitly reviewed baseline."""
    config = yaml.safe_load(WATCHLIST.read_text(encoding="utf-8")) or {}
    findings: list[Finding] = []
    for upstream in config.get("upstreams", []):
        try:
            if upstream["kind"] == "github_release":
                payload = public_json(
                    f"https://api.github.com/repos/{upstream['repository']}/releases/latest"
                )
                latest = payload.get("tag_name")
            elif upstream["kind"] == "pypi":
                payload = public_json(f"https://pypi.org/pypi/{upstream['package']}/json")
                latest = payload.get("info", {}).get("version")
            else:
                findings.append(
                    Finding("warning", "upstream-watch", f"Unknown watch type: {upstream['kind']}")
                )
                continue
        except (KeyError, OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            findings.append(
                Finding(
                    "warning",
                    "upstream-watch",
                    f"Could not check {upstream.get('name', 'an upstream')}: {error}",
                )
            )
            continue

        known = upstream.get("known_release")
        if latest and latest != known:
            findings.append(
                Finding(
                    "notice",
                    "upstream-watch",
                    f"{upstream['name']} latest release is {latest}; reviewed baseline is {known}.",
                    approval=upstream.get("approval", "maintainer-review"),
                )
            )
    return findings


def build_report(checks: dict[str, str], include_upstreams: bool) -> tuple[dict[str, Any], str]:
    """Build structured and readable output for a single observation."""
    findings = documentation_findings()
    for name in CHECK_NAMES:
        if checks[name] != "success":
            findings.append(
                Finding("error", "verification", f"{name} check finished as {checks[name]}.")
            )
    if include_upstreams:
        findings.extend(upstream_findings())

    data = {
        "generated_at": datetime.now(UTC).isoformat(),
        "checks": checks,
        "findings": [asdict(finding) for finding in findings],
        "action_required": bool(findings),
    }
    lines = [
        "# Maintainer digest",
        "",
        "## Verification",
        "",
        "| Check | Result |",
        "| --- | --- |",
        *(f"| {name} | {checks[name]} |" for name in CHECK_NAMES),
        "",
        "## Findings",
        "",
    ]
    if not findings:
        lines.append("No material maintenance finding. No Issue or PR is needed.")
    else:
        for finding in findings:
            suffix = f" Approval: `{finding.approval}`." if finding.approval else ""
            lines.append(
                f"- **{finding.severity} · {finding.category}:** {finding.message}{suffix}"
            )
    lines.extend(
        [
            "",
            "## Automation decision",
            "",
            "Safe repairs are limited to Ruff fixes and always require a passing draft PR. "
            "Model, licence, privacy, data, and training findings require a maintainer.",
        ]
    )
    return data, "\n".join(lines) + "\n"


def changed_paths() -> list[str]:
    result = run(["git", "diff", "--name-only"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return [line for line in result.stdout.splitlines() if line]


def safe_fix() -> int:
    """Apply the narrowly allowed Ruff fixes, then validate the entire repository."""
    fix = run([sys.executable, "-m", "ruff", "check", "--fix", *SAFE_FIX_PATHS])
    if fix.returncode != 0:
        print(fix.stdout, end="")
        print(fix.stderr, file=sys.stderr, end="")
        return fix.returncode

    invalid = [path for path in changed_paths() if not path.startswith(SAFE_FIX_PATHS)]
    if invalid:
        print(
            f"Refusing changes outside the safe-fix allowlist: {', '.join(invalid)}",
            file=sys.stderr,
        )
        return 2

    for command in (
        [sys.executable, "-m", "pytest"],
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "mypy", "app"],
    ):
        result = run(command)
        print(result.stdout, end="")
        print(result.stderr, file=sys.stderr, end="")
        if result.returncode != 0:
            return result.returncode
    print("safe-fix validation passed")
    return 0


def observe(arguments: argparse.Namespace) -> int:
    checks = {name: getattr(arguments, name) for name in CHECK_NAMES}
    data, markdown = build_report(checks, arguments.check_upstreams)
    Path(arguments.report).write_text(markdown, encoding="utf-8")
    Path(arguments.json_report).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(markdown)
    return 0


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(description=__doc__)
    subparsers = command_parser.add_subparsers(dest="command", required=True)

    observe_parser = subparsers.add_parser("observe", help="write a maintainer observation report")
    for name in CHECK_NAMES:
        observe_parser.add_argument(
            f"--{name}", choices=("success", "failure", "skipped"), default="success"
        )
    observe_parser.add_argument("--check-upstreams", action="store_true")
    observe_parser.add_argument("--report", default="autopilot-digest.md")
    observe_parser.add_argument("--json-report", default="autopilot-digest.json")
    observe_parser.set_defaults(handler=observe)

    safe_fix_parser = subparsers.add_parser("safe-fix", help="run only approved automatic repairs")
    safe_fix_parser.set_defaults(handler=lambda _: safe_fix())
    return command_parser


def main() -> int:
    arguments = parser().parse_args()
    os.chdir(REPOSITORY)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
