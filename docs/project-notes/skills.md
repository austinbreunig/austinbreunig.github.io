# skills

Repo: https://github.com/austinbreunig/skills

## Title and summary

**skills**: Claude Code skills for productivity and data engineering development.

Audience: a visitor who has never used Claude Code skills (hiring managers, other engineers). Shows process thinking: how a product goes from idea to production with agents.

Takeaway: a working system where agents run on their own and stop only at decisions that are the user's.

## Free block (the author's own words, unlabeled, verbatim)

Skills for productivity and data engineering development. Lifecycle makes development a whole lot easier. Instead of relying on a heavy multi agent workflow, lifecycle provides a guided workflow where a user and agents interact in a collaborative way.

**Still to write (author):** one or two sentences on how lifecycle extends and wires together Matt Pocock's skills to give a more seamless development workflow. Source: `docs/Lifecycle Skills.md` in the skills repo.

(Typos lightly fixed; the author writes the final page words.)

## Page structure (after the summary, before the repo link)

Editorial rows: gutter label, content right. Plain HTML. One small inline-JS widget (no libraries) with a plain no-JS fallback. Guided visuals are sandboxed iframes per the embedding decision.

1. *(no label)*: the free block above
2. **The cycle**: Start to Verdict, human steps marked. Guided visual + small clickable cycle strip (no-JS fallback: the same steps as a plain list).
3. **Human step vs agent step**: the core idea. A human step is a decision only you can make; `/lifecycle` runs every other step on its own and stops at the next human step with the exact next command.
4. **How it wires Matt Pocock's skills**: Pocock's skills do the work; lifecycle navigates and adds `mindful`, `seam-lock`, `ramp`, `cycle-retro`. Guided visual.
5. **Lifecycle**: its 5 skills, one line each. Not a long list.
6. **Productivity**: `/hq`, `/brief`, `/eod`, `/jot`, one line each.
7. **Human**: `/human`, shared by both plugins.
8. **What it built**: plumber, maply, bap. Link only pages that are live; unfinished ones say "Coming soon" with no link.
9. **Status**: every merge to `main` is a release; no version numbers; tested with mattpocock-skills 1.3.1. No user counts. Link to Matt Pocock's skills as what it builds on.

Not on the page: an Install block (link the README instead), a live demo, tabs, a skill-by-skill catalog of all 10.

Privacy: `hq` appears only as "a private HQ repo". No private repo names or contents anywhere, including screenshots.

## Visual list

| Row | What it shows | Type |
|---|---|---|
| The cycle | The cycle with human steps marked (from `docs/Lifecycle Skills.md`) | guided visual (agent, `human` skill) + inline clickable strip |
| How it wires Pocock's skills | Which Pocock skills lifecycle calls, and where its own skills slot in | guided visual (agent, `human` skill) |
| Lifecycle | A `/lifecycle` run stopping at a human step, with its exact next command | screenshot (you) |
| Productivity | A `/brief` output | screenshot (you) |

Guided visual briefs go in `visuals/<name>.brief.md` with Question / Reader / Steps / Source.

## Screenshot checklist

Readable at 1200px wide, theme matching the site, no private repo names, paths or ticket titles in view.

1. **`/lifecycle` stop**: run `/lifecycle` in a repo mid-cycle so it stops at a human step. Capture the stop message with the "next command" line.
2. **`/brief`**: run `/brief`. Crop or blur private repo names (`hq` and any others). Keep the structure visible: where you left off, open PRs, recent jots.
