# Site contract

The shared rules every page ticket follows. Source: spec #19. The checker is `docs/contracts/check.py`.

## Files and owners

| Path | Owner ticket |
|---|---|
| `style.css`, `.nojekyll` | #20 |
| `index.html`, `images/profile.jpg`, `images/favicon.*`, `images/<project>/thumb.*` | #21 |
| `resume.html`, `files/AustinBreunig_Resume.pdf` | #22 |
| `projects/bap.html`, `images/bap/*` | #23 |
| `docs/contracts/*` | seam lock (read-only for tickets; change by adding a line, not rewriting) |

After #21, #22, #23 are done, #24 may edit any launch file to fix failures. #25 only merges.

## Rules

1. Only `style.css` holds CSS. No `<style>`, no `style=""` (guided visuals are the one exception, not at launch).
2. Pages use only class names from `classes.txt`. A new name goes into `classes.txt` and `style.css` in the same change.
3. Colors come from the tokens in spec section 3 (`--bg --surface --ink --muted --accent --accent2 --line`), declared on `:root` in `style.css`. No raw hex outside that block.
4. Links between pages and to shared files are relative. Root pages use `style.css`, `resume.html`. Pages in `projects/` use `../style.css`, `../index.html`, `../resume.html`. No link starts with `/` or the site URL.
5. Same-site paths a page may reference: `style.css`, `index.html`, `resume.html`, `projects/<name>.html`, `images/...`, `files/AustinBreunig_Resume.pdf`.
6. Placeholder links (email, LinkedIn, GitHub, favicon) use `href="#todo"` or a missing-file-free stand-in, and carry `data-placeholder="<what is missing>"`. The same attribute marks every stand-in text or image. Ticket #24 lists them with `grep -rn data-placeholder`.
7. Every page: `<html lang="en">`, charset and viewport meta, `<title>`, meta description, one `h1`, `<link rel="stylesheet" href="[../]style.css">`, Google Fonts link for IBM Plex Serif and IBM Plex Mono only.
8. No `<script>` at launch.
9. Breakpoints: 640px (landing), 700px (project, resume).
10. Do not write or edit the author's words. Stand-ins are marked.

## Top bar (identical on every page, only the prefix changes)

`P` is `` at the root and `../` in `projects/`.

```html
<header class="topbar">
  <a class="topbar-name" href="P index.html">Austin Breunig</a>
  <nav class="topbar-links" aria-label="Main">
    <a href="#todo" data-placeholder="email">Email</a>
    <a href="#todo" data-placeholder="LinkedIn URL">LinkedIn</a>
    <a href="#todo" data-placeholder="GitHub URL">GitHub</a>
    <a href="P resume.html">Resume</a>
  </nav>
</header>
```

(Written as `P index.html` here only for readability; real hrefs have no space.) The checker compares the header text and link order across pages.

## Page body shapes

Pages wrap content in `<main class="wrap">`.

Landing card (one per project, five total). Linked card: the `<a class="card">` wraps everything. Unfinished: a `<div class="card">` with `<p class="card-soon">Coming soon</p>` and no `<a>`.

```html
<a class="card" href="projects/bap.html">
  <div class="card-thumb media"><img src="images/bap/thumb.jpg" alt=""></div>
  <h3 class="card-title">bap</h3>
  <p class="card-line">...</p>
</a>
```

Project page (`projects/<name>.html`): `h1.project-title`, `p.project-summary`, `a.repo-link` in the first row; then `div.row > div.row-body.free-block`; then `section.row > h2.row-label + div.row-body > ol.steps > li.step`; then a last row with `a.button`. A step is `span.step-label` ("STEP 1"), `h3.step-title`, `p.step-text`, and `div.step-visual.media`. The even-step flip is CSS (`.step:nth-child(even)`), not markup. See `fixtures/good-project.html`.

Resume: `header.resume-head` with `h1`, `p.contact`, `a.button`; then `section.resume-section` with `h2`; then `div.job` rows holding `div.job-gutter` (`span.job-role`, `span.job-org`, `span.job-dates`) and `div.job-body`. Projects are `ul.proj-list > li` with `span.proj-name` and `p.proj-meta` (`span.meta-label` for "Tools:"/"Method:"). Education reuses `div.job`.

## Run the checker

```
python3 docs/contracts/check.py            # check all site pages in the repo
python3 docs/contracts/check.py --selftest # prove the checker on the fixtures
```
