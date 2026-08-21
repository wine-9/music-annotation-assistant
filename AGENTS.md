# Repository agent rules

This repository is local-first, reproducible, and conservative about music labels.
Automation is welcome only when it creates a useful, reviewable result.

## Non-negotiable boundaries

- Never read, copy, index, or reference `/Users/voxrockschool/Projects/音乐标注AI`.
  That location can contain private contract work, checkpoints, iMark material, and
  training artefacts which are outside this repository's scope.
- Do not add audio, annotations, model weights, checkpoints, credentials, or
  private exports to the repository.
- Do not create empty commits, synthetic Issues/PRs, stars, comments, or
  promotional posts to simulate project activity.
- Never push directly to `main`. Every automated code change must be proposed
  in a pull request and pass the repository checks.

## Approval-required changes

Do not automate changes affecting model quality, evaluation claims, model or
dependency licences, data privacy, training procedures, thresholds calibrated
from data, or a public release. Create a concise diagnostic issue or draft and
wait for a maintainer.

## Safe automated repairs

The only enabled code repair is `ruff check --fix` within `app/`, `tests/`, and
`scripts/`. It must pass pytest, Ruff, and mypy before a draft PR is created.
Other failures are observations, not permission to rewrite code.

Read [MAINTAINER_AUTOPILOT.md](MAINTAINER_AUTOPILOT.md) before modifying any
automation.
