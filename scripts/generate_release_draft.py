#!/usr/bin/env python3
"""Generate review-only release notes and promotion drafts from real commits."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def commits_since(base_ref: str, head_ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "log", "--format=- %s (%h)", f"{base_ref}..{head_ref}"],
        cwd=REPOSITORY,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip())
    return [line for line in result.stdout.splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="previous released tag or commit")
    parser.add_argument("--head", default="HEAD", help="candidate release commit")
    parser.add_argument("--output", default="release-draft.md")
    arguments = parser.parse_args()

    commits = commits_since(arguments.base, arguments.head)
    if not commits:
        print("No commits since the selected base; no release draft created.")
        return 0

    draft = "\n".join(
        [
            "# Release draft (maintainer review required)",
            "",
            f"Comparison: `{arguments.base}...{arguments.head}`",
            "",
            "## Included commits",
            "",
            *commits,
            "",
            "## Review checklist",
            "",
            "- [ ] Verify user-facing claims against reproducible public evidence.",
            "- [ ] Review model, dataset, and dependency licences.",
            "- [ ] Confirm no private audio, annotations, checkpoints, or outputs are included.",
            "- [ ] Confirm compatibility notes and upgrade steps.",
            "",
            "## Promotion draft (do not post automatically)",
            "",
            "A reviewable update is ready for music-annotation-assistant. It includes the "
            "changes listed above; please verify the release notes and compatibility "
            "before sharing.",
            "",
        ]
    )
    (REPOSITORY / arguments.output).write_text(draft, encoding="utf-8")
    print(draft)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
