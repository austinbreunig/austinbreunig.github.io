# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues in `austinbreunig/austinbreunig.github.io`. Use the `gh` CLI.

## Conventions

- Create an issue: `gh issue create --title "..." --body "..."`
- Read an issue: `gh issue view <number> --comments`
- Comment: `gh issue comment <number> --body "..."`
- Close: `gh issue close <number>`
- Triage state is recorded as labels (see `triage-labels.md`).

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.

## Wayfinding operations

Used by `/wayfinder`. The **map** is one issue with child issues.

- **Map**: an issue labelled `wayfinder:map`, holding the Destination / Notes / Decisions-so-far / Fog body. Edit it with `gh issue edit <map> --body-file <file>`.
- **Child ticket**: a GitHub **sub-issue** of the map, labelled `wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`, or `wayfinder:task`. Link it with
  `gh api -X POST repos/austinbreunig/austinbreunig.github.io/issues/<map>/sub_issues -F sub_issue_id=<id>`.
  `<id>` is the issue's database id (`gh api repos/austinbreunig/austinbreunig.github.io/issues/<number> --jq .id`), not its number.
- **Blocking**: GitHub's native "blocked by":
  `gh api -X POST repos/austinbreunig/austinbreunig.github.io/issues/<ticket>/dependencies/blocked_by -F issue_id=<blocker id>`.
  A ticket is unblocked when every blocker is closed.
- **Frontier**: open sub-issues of the map that have no open blockers and no assignee. Lowest number first.
- **Claim**: `gh issue edit <ticket> --add-assignee @me`, before any work.
- **Resolve**: post the answer with `gh issue comment`, run `gh issue close`, then add a one-line link to the map's Decisions-so-far.
- **Privacy**: this repo is public. Keep local file paths and private details out of issues.
