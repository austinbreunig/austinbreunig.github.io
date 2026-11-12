# Cycle 2 seams (spec #26)

Pointers for every ticket (cycle 2 tickets: #28 to #36, #38, #39; #34, #35, #37 are closed as not planned, #34 and #35 merged into #33). State each rule once here. The checker is `check.py`; the contract is `site-contract.md`.

## S1. Labeled row without steps (page body shape)

Used by all five project pages. `ol.steps` is not used.

```html
<section class="row">
  <h2 class="row-label">Scale</h2>
  <div class="row-body"> ...p, .pair, .code-block, .visual, .widget... </div>
</section>
```

- Two screenshots side by side: `<div class="pair"><div class="pair-item media">img or stand-in</div><div class="pair-item media">...</div></div>`.
- Code: `<pre class="code-block"><code>...</code></pre>`.
- Status: `<p class="status-line">...</p>` inside a `Status` row (plumber: a plain `p.status-line` after its rows).
- Stand-in (any kind): the element carries `data-placeholder="<what is missing>"`. Image stand-in = `div.media[data-placeholder]` with no `img`. Text stand-in = `p[data-placeholder]`.
- Unbuilt cross-link = plain text, no `<a>`. Built = relative `<a>`.

## S2. Guided-visual embed (owner: #29; later consumers: the author, for the slots in #30 #31 #32)

Agents build no visual content. #29 builds the embed and demos it on `visuals/_demo.html` (ships live). #30, #31, #32 leave a stand-in slot `div.media[data-placeholder="visual: NAME"]` (names: `how-bap-works`, `layers-to-scores`, `monthly-refresh`, `python-to-map`, `how-its-built`). The author builds each visual later with a live server and swaps the slot for this markup. Skills has no guided visuals.

Markup, identical for every visual. `NAME` is the visual name. Page in `projects/`, visual in `visuals/`.

```html
<figure class="visual" data-visual="NAME">
  <div class="visual-bar">
    <a class="button" href="../visuals/NAME.html" target="_blank" rel="noopener">Open full</a>
    <button class="visual-reload" type="button">Reload</button>
  </div>
  <div class="visual-frame media">
    <iframe src="../visuals/NAME.html" title="ALT TEXT" sandbox="allow-scripts"></iframe>
  </div>
  <p class="visual-fallback" hidden>This visual did not load. <a href="../visuals/NAME.html">Open it on its own page.</a></p>
  <noscript><p class="visual-fallback"><a href="../visuals/NAME.html">Open the visual on its own page.</a></p></noscript>
</figure>
```

Protocol (a visual page posts to its parent; the embed script listens):

| Message (`event.data`) | When | Embed does |
|---|---|---|
| `{type: "visual-ready"}` | visual loaded | cancel the 4 s timer |
| `{type: "visual-height", px: <integer>}` | on load and on resize | set iframe height to `px` |

Rules:
- Embed script checks `event.source` is the iframe's own `contentWindow`, never trusts the origin string (sandbox without `allow-same-origin` gives origin `"null"`).
- No `visual-ready` within 4000 ms: hide the iframe, unhide `.visual-fallback`. Reload: restore the iframe, set `src` again, restart the timer.
- One script, `<script src="../embed.js">`, at the end of body, only on pages with a `.visual`. No inline copy per page. Visual pages post with `parent.postMessage(msg, "*")`. They take no data from the parent.
- `visuals/NAME.html` is a standalone page: its own inline `<style>` and `<script>` are allowed there (rule 1 exception). The checker skips `visuals/`.
- No `loading="lazy"` (it would starve the 4 s timer for visuals below the fold).
- Every visual page MUST post `visual-ready` and `visual-height` (copy the script in `fixtures/visual-stub.html`). The brief template (`visuals/<name>.brief.md`) gets a fifth line, `Protocol: copies visual-stub script`, and each visual ticket checks it: open the embed, the fallback must NOT show.
- JS off: the iframe keeps a CSS `min-height` (#29) and the `noscript` link sits under it. A visual must keep working with reduced motion (no autoplay motion; a static end state) and carry the `title` text alternative.
- Open questions are in the spec, not here.

Stub for demo and fixtures: `fixtures/visual-stub.html`.

## S3. Script rule (owner: #28)

| Page | Script allowed |
|---|---|
| `projects/plumber.html`, `projects/skills.html` | one inline `<script>` (no `src`) for decided widgets |
| any page with `.visual` | plus exactly one `<script src="../embed.js">` |
| anything else | none |

Never: third-party `src`, `<script type="module">`, `fetch`, `XMLHttpRequest`, `WebSocket`, `import`, `eval`, `document.write`. `<iframe>` is allowed only inside `figure.visual > div.visual-frame`, with `sandbox="allow-scripts"`, a `title`, and a relative `src` starting `../visuals/`.

The checker must reject: a script on a page not in the table; a `src` that is not `../embed.js`; a second embed script; an iframe outside `.visual-frame`; an iframe without `sandbox`/`title`; an absolute `src` on an iframe.

Fixtures (each has `<!-- check-as: ... -->` and `<!-- expect: pass|fail -->`; the selftest list in `check.py` is #28's to extend):

| Fixture | check-as | expect |
|---|---|---|
| `good-embed-project.html` | `projects/bap.html` | pass |
| `good-widget-project.html` | `projects/plumber.html` | pass |
| `bad-script-page.html` (script on `maply`) | `projects/maply.html` | fail |
| `bad-iframe-page.html` (iframe outside `.visual-frame`, no sandbox) | `projects/bap.html` | fail |

## S4. Widget logic (owners: #33 #36)

All widget logic is plain functions in the page's one inline script. Names and results are fixed here. The golden file is `fixtures/widgets.golden.json`; the harness is `widget-harness.js`:

```
node docs/contracts/widget-harness.js <file.js | page.html>
```

It pulls the inline script from an `.html` (or reads a `.js`), runs it in a VM with no DOM and no network, and compares. The stub is `stubs/plumber-widgets.stub.js` (returns the golden answers for the listed inputs only; replace it in place, #33 inlines the real functions in the page).

Functions (pure, no DOM access inside them):

- The harness must be run with `--require checkResult` or `--require strategyRows` (comma list) so a renamed or wrapped function fails instead of skipping. Page tickets keep these functions at the top level of the inline script (no IIFE) so the harness can see them. #33 runs with `--require strategyRows,checkResult`.
- `checkResult(choice)` (good = PASS, wrong-signature = FAIL, bad-crs = WARN) where `choice` is `"good"`, `"wrong-signature"` or `"bad-crs"`. Returns `{status: "PASS"|"WARN"|"FAIL", message: string}`. Unknown choice throws `Error`.
- `strategyRows(strategy, rows, partitions)` where `strategy` is `"local"` or `"localmp"`, `rows` is an array of ids, `partitions` an integer >= 1 (non-integers throw). Returns `{yamlLine, partitions: [[id...]...], returned: [id...]}`. `local`: one partition holding all rows. `localmp`: rows dealt round-robin into `partitions` lists (empty lists allowed). `returned` is always the original `rows` order. Unknown strategy or `partitions < 1` throws `Error`.
- `phaseInfo(name)` for the clickable pipeline: returns `{signature, config}` for each phase in `PHASES`; unknown name throws.
- Strip (skills): data array `CYCLE_STEPS = [{name, human: boolean}]`; the no-JS list is the same array written as HTML. #36 checks the step names by hand against the cycle doc; names; come from the skills repo's lifecycle doc, not from memory.

`PHASES` for plumber (the 12-line YAML in the README is the source; #33 copies it, the check is that every phase in the YAML has a `phaseInfo` entry).

## S5. Ownership map (no file has two owners)

| Path | Ticket |
|---|---|
| `docs/project-notes/*`, repo cleanliness | #27 |
| `docs/contracts/check.py`, `site-contract.md` edits, `style.css` rules for S1 to S3 classes, `docs/spec.md` section 6 | #28 |
| `embed.js`, `style.css` section `/* page: embed */`, `visuals/_demo.html` (ships live, never deleted) | #29 |
| `projects/bap.html`, `images/bap/*` (visual slot is a stand-in) | #30 |
| `projects/conservation-dashboard.html`, `images/conservation-dashboard/*` (visual slots are stand-ins) | #31 |
| `projects/maply.html`, `images/maply/*` (visual slots are stand-ins) | #32 |
| `projects/plumber.html` (whole file: pipeline, strategy switch, check playground, script), `images/plumber/*`, `style.css` section `/* page: plumber */` | #33 |
| `projects/skills.html`, `images/skills/*` | #36 |
| `index.html` cards, cross-link edits | #38 |
| any launch file, to fix failures only | #39 |

#33 is the only owner of `plumber.html` (no marker split). It may append class lines to `classes.txt`; widget animation classes `widget-partition` and `widget-row` are already there (#28 writes their CSS).
`skills.html` has two writers. #36 creates it with one marker, `<!-- links:built -->`, around the What-it-built row (#38 edits only inside it). The page's only visual is the clickable strip (inline JS, from `CYCLE_STEPS`); it has no guided visuals and no `slot:` markers. #36 checks step names against the cycle doc by hand.
