# OSS Autopilot

OSS Autopilot keeps routine maintenance visible without manufacturing activity.
It is deliberately conservative: it observes continuously, repairs only a
small allowlist, and asks a human for consequential decisions.

## Architecture

```text
scheduled Actions / Dependabot
        |
        +-- CI + `scripts/autopilot.py observe` --> job summary + report artifact
        |       |                                      |
        |       `-- material failure/change ------------+--> one deduplicated Issue
        |
        +-- `scripts/autopilot.py safe-fix` --> pytest + ruff + mypy --> draft PR
        |
        `-- manual release-draft workflow --> notes + promotion draft artifact
```

`scripts/autopilot.py` is intentionally standard-library based (apart from the
repository's existing PyYAML dependency). It generates Markdown and JSON that
can be inspected locally or uploaded by Actions. The upstream watchlist lives
in [config/autopilot_watchlist.yaml](config/autopilot_watchlist.yaml); it is
only a notifier, never an updater.

## Triggers

| Trigger | Frequency | Result |
| --- | --- | --- |
| CI | push and pull request | pytest, Ruff, mypy |
| Dependabot | weekly | constrained Python and Action update PRs |
| Autopilot observation | Monday 09:17 UTC | checks, docs freshness, upstream watch, digest artifact |
| Safe repair | Wednesday 09:37 UTC | only Ruff safe fixes; a draft PR only if a real diff survives all checks |
| Release draft | manual | release notes and launch-post drafts; never publishes a release or post |

Manual workflow dispatch is available for an immediate observation or repair.

## Permissions and approval gates

All workflows use the minimum necessary `GITHUB_TOKEN` permissions. Observation
can write Issues only to report a material, deduplicated failure. Safe repair
can write a branch and pull request, but never `main`. Dependabot is limited to
the declared ecosystems and opens normal reviewable PRs.

The following always require explicit maintainer review:

- model/provider version or quality changes, calibration thresholds, benchmarks,
  training settings, and public quality claims;
- model, dataset, or dependency licence changes;
- data retention, uploads, telemetry, and privacy policy changes;
- releases, publication, or any external promotional message.

The repository is protected from the private project path specified in
[AGENTS.md](AGENTS.md). No workflow, script, or prompt is allowed to traverse
or use it.

## Failure, rollback, and noise control

An unsuccessful check produces a report and (only when material) an Issue; it
does not trigger a repair. The safe-repair workflow creates no commit when
there is no diff, and it creates no PR if validation fails. Closing the draft
PR or reverting its merge is the rollback path. `concurrency` prevents a newer
run racing an older one.

The observing workflow only opens an Issue when the report contains actionable
findings. It looks for an existing open Issue with the same `autopilot` title
and comments there instead of opening duplicates. Routine green runs are job
summaries and artifacts, not GitHub noise.

## Cost controls

- Python 3.12 and the small dev dependency set are used; model, evaluation, and
  separation extras are never installed by automation.
- Upstream requests are a tiny fixed watchlist with short timeouts; failures are
  warnings, not retries.
- No hosted AI/API call is required for the MVP. A future AI triager must keep
  the same approval gates and redact paths, credentials, audio, and annotations.

## Local operation

```bash
make setup
make autopilot-observe
make autopilot-safe-fix
```

`autopilot-observe` skips public network checks by default so it is deterministic
locally. Run `python scripts/autopilot.py observe --check-upstreams` when an
upstream comparison is wanted. Before enabling write workflows, protect `main`
in GitHub with required CI checks and require a maintainer review for PRs.

## Planned extensions

1. Add a public, redistributable fixture and a licensed benchmark dataset gate.
2. Add a human-approved provider/model compatibility matrix.
3. Add issue-form classification and a maintainer-owned release checklist.
4. Add an AI triager only after it can operate on redacted GitHub metadata and
   its suggestions remain draft-only.
