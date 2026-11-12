# Spec: Portfolio website

Source of truth for building `austinbreunig.github.io`. Every decision here comes from a closed ticket on the "Map: Portfolio website" issue. Terms are in `GLOSSARY.md`.

Items marked **PLACEHOLDER** are not known yet. The build uses a stand-in and a later step fills it in.

## 1. What we are building

A personal portfolio site: a **landing page**, one **project page** per public repo, and a **resume page**.

- Plain hand-written HTML plus one shared CSS file. No generator, no framework, no build step.
- Hosted on GitHub Pages from the root of `main`, in the existing repo `austinbreunig/austinbreunig.github.io`.
- Feel: elegant earth tones. Down-to-earth, with precision, skill, and wisdom showing through.
- The author writes all the words. Agents ask prompts and record answers; they do not write page copy.

Pages: `index.html`, `resume.html`, and `projects/<name>.html` for `bap`, `conservation-dashboard`, `maply`, `plumber`, `skills`. `hq` is private and has no page.

Out of scope: a page for `hq`, public repos outside `projects/`, a site generator, CMS, blog, analytics, comments, a custom domain.

## 2. File layout

```
/
├── index.html
├── resume.html
├── style.css                 the one shared stylesheet
├── .nojekyll                 empty file
├── projects/
│   ├── bap.html
│   ├── conservation-dashboard.html
│   ├── maply.html
│   ├── plumber.html
│   └── skills.html
├── images/
│   ├── profile.jpg
│   └── <project>/            thumb.jpg + screenshots
├── visuals/
│   ├── <name>.html           guided visual
│   └── <name>.brief.md       its brief
├── files/
│   └── AustinBreunig_Resume.pdf
└── docs/                     spec, tracker docs, project notes
```

Rules:
- One HTML file per page. Project pages are flat files (served as `/projects/<name>`).
- Only `style.css` holds CSS. Inline styles live only inside guided visuals.
- All links between pages are relative, so the site works in a local preview and live. Pages under `projects/` reach shared files with `../`.
- `.nojekyll` stops Pages from turning `.md` files into web pages.
- The unused profile candidates and the `.docx` resume master do not ship.

## 3. Look and feel

Dark forest-green page, sand text, ochre accent, IBM Plex type.

| Token | Value | Use |
|---|---|---|
| `--bg` | `#2a3328` | page |
| `--surface` | `#344033` | cards, media boxes |
| `--ink` | `#ece4d2` | body text |
| `--muted` | `#a9b39c` | small and secondary text |
| `--accent` | `#d9a441` | links, buttons, section names |
| `--accent2` | `#9fb88a` | labels, picture gradient |
| `--line` | `#4a5946` | borders |

- Headings: IBM Plex Serif, weight 600. Body: IBM Plex Mono, about 0.92rem, line height 1.6.
- Corners 4px. Borders 1px `--line` on cards and the top bar.
- Quotes: italic Plex Serif with a 3px accent bar on the left.
- Fonts load from Google Fonts: IBM Plex Serif and IBM Plex Mono only.
- Max content width 1040px. Phone breakpoints: 640px (landing), 700px (project, resume).
- Reference prototype: branch `prototype/style-board`, `prototype-style.html` (board D). Throwaway; copy values from here, not the file.

## 4. Top bar (every page)

Name on the left. Email, LinkedIn, GitHub, and Resume links on the right. 1px `--line` border under it. It wraps on a phone. "Resume" links to `resume.html`.

**PLACEHOLDER:** the real email, LinkedIn URL, and GitHub URL.

## 5. Landing page (`index.html`)

Layout A (prototype: `prototype/landing-layout`).

- **Hero:** square profile picture (about 200px) on the left; name (`h1`) and the bio on the right. On a phone (under 640px) it stacks: centered 140px picture, then name and bio.
- **Projects:** a "Projects" heading, then a responsive grid of cards (`minmax(260px, 1fr)`). One card per project: thumbnail, name, one line. One column on a phone.
- **Unfinished projects:** the card still shows. In place of a link it shows a "Coming soon" label. No placeholder page exists and nothing else is on the card. When a project page is done, its card links to it.
- At launch only `bap` links through.

**Bio (final, 55 words, author's words):**

> I'm a geospatial engineer who turns small ideas into creative, scalable solutions by designing flexible, reliable data pipelines. I cut noise, clutter, and complexity from existing workflows and rebuild them to be clean, easy to change, and built to last. Away from the screen, I push myself outdoors, climbing and bikepacking in search of adventure.

**Profile picture:** snow, plaid shirt, looking to the side. Square crop of the face and upper body, about 600×600 JPEG, under 200 KB, built from the full-size original (which stays out of the repo). Saved as `images/profile.jpg`. Alt text: "Photo of Austin Breunig".

**PLACEHOLDER:** project card thumbnails (`images/<project>/thumb.jpg`) and the one-line summary per project, except `bap`.

## 6. Project page (`projects/<name>.html`)

Each project page has its own layout. A Workflow section does not fit every project, so there is no fixed list of sections. The layout (headers, sections, functionality, visuals) is decided in that project's grilling session and built with the author in their own editor.

Fixed on every project page:

1. **Title and summary:** large serif project name (`clamp(2.2rem, 7vw, 4rem)`), one-line summary, and a small repo link.
2. **Repo link:** an accent button to the public repo, at the end.
3. The top bar and the shared look and feel (section 3). Pages reach `style.css` with `../`.

Everything between the summary and the repo button is decided per project. Available building blocks, not requirements: a free block of the author's words, editorial rows with a left gutter label (150px), numbered steps with a visual beside each (16:10), guided visuals, screenshots. Rows are separated by a 1px `--line` top border. On a phone (under 700px) the gutter label stacks above the content.

Not on a project page unless a grilling session chooses it: Further reading, a hero visual, a gallery.

### Content per project

Each project gets one grilling session that asks the key prompts, records the author's answers, and decides the page's own structure. The session ends with a visual list: for each visual, where it goes on the page, what it shows, and whether it is a guided visual or a screenshot. The author takes screenshots; an agent makes guided visuals.

| Project | Notes | Status |
|---|---|---|
| `bap` | Summary, audience, and free block written (`docs/project-notes/bap.md`). | Page layout to be decided again (grilling reopened). `projects/bap.html` will be redone. |
| `conservation-dashboard` | Grilling ticket open | **PLACEHOLDER** (card says "Coming soon") |
| `maply` | Grilling ticket open | **PLACEHOLDER** (card says "Coming soon") |
| `plumber` | Grilling ticket open | **PLACEHOLDER** (card says "Coming soon") |
| `skills` | Grilling ticket open | **PLACEHOLDER** (card says "Coming soon") |

## 7. Guided visuals

An interactive explainer made by an agent with the `human` skill, embedded in a project page where its grilling session places it. Prototype: `prototype/visual-embed`.

- Markup: `<iframe sandbox="allow-scripts" title="Guided visual: ..." src="../visuals/<name>.html">`, width 100%, no border. It sits in the standard media box (`--surface`, `--line`, 4px corners). A muted caption below: "Guided visual: <what to do>".
- Height starts at 260px and grows to the height the visual reports.
- Under the frame: "Open full ↗" (the visual's own file in a new tab).
- Each visual is one self-contained HTML file with no external dependencies. It ends with `parent.postMessage({type:'visual-height', h: document.documentElement.scrollHeight}, '*')` after every redraw.
- On a phone the step text and visual stack, visual at full width.
- **Brief** (`visuals/<name>.brief.md`, four lines): **Question** (what it answers), **Reader** (who it is for and what they know), **Steps** (what to show, in order), **Source** (files, code, or notes to draw from).

**PLACEHOLDER:** the visuals themselves. None are chosen yet.

## 8. Resume page (`resume.html`)

Layout B with bullets (prototype: `prototype/resume-layout`). Text and format come from the master resume, which is kept outside the repo.

- Same top bar. Header: name as a big serif `h1` (`clamp(2.2rem, 7vw, 3.6rem)`), a muted contact line (phone, email, location, with `mailto:` and `tel:` links), and an accent **Download PDF ↓** button linking to `files/AustinBreunig_Resume.pdf`.
- Sections "Experience" and "Education": accent serif `h2` with a 1px `--line` top border.
- Job row: two columns. A 170px left gutter holds the role (serif, accent), org · place, and dates (muted). The right holds the blurb, a small uppercase "Day-to-day" label with the tool list, then the projects.
- Projects are bullets with a 7px ochre dot: bold serif name, a dash, the description. Under it, muted "Tools:" and "Method:" lines, their labels in `--accent2`.
- Other projects: a small label, then a plain `<ul>` (Cotality, 2022 to present job only).
- Education: same gutter rows. Degree in the gutter; school and year below; the capstone line on the right.
- Under 700px the gutter stacks above the content.
- Hand-write the HTML (`h1`, `h2`, `p`, `ul`, `li`, `span`). No JS. Do not convert docx to HTML.
- PDF: `soffice --headless --convert-to pdf --outdir out/ AustinBreunig_Resume.docx`. LibreOffice swaps Georgia for Noto Serif; if exact Georgia matters, export from Word or Google Docs instead. Check the fonts.
- When the resume changes, update the docx, the HTML, and the PDF together.

## 9. Quality bar

Checked by hand on the local preview before launch, then again on the live URL. Nothing is added to the repo.

- **Mobile:** every page at 360px, 768px, and 1280px, in dev tools and once on a real phone. No sideways scroll, text readable without zoom, tap targets at least 44px.
- **Accessibility:** text contrast 4.5:1 or better (the palette passes: sand on green 10.4, muted on green 6.0, ochre on green 5.8, muted on surface 5.0). `lang="en"`, one `h1`, headings in order. Every `img` has `alt` (`alt=""` if decorative). Every iframe has a `title`. Tab through each page: visible focus, no traps. Works at 200% zoom. Chrome Lighthouse accessibility score of 95 or higher.
- **Weight:** images 200 KB or less. Each page without its iframes 1 MB or less. The only third-party request is Google Fonts. No analytics or other scripts.
- **Links:** click every link. Run `npx linkinator` against `python3 -m http.server`, internal and external. Repeat live.
- **Metadata:** every page has `<title>`, meta description, and viewport. The landing page has Open Graph tags (title, description, profile picture). A favicon. No `sitemap.xml` or `robots.txt`.

**PLACEHOLDER:** the favicon image.

## 10. Build order

Work on a branch called `site`. Never commit site work straight to `main`: every push to `main` goes live.

1. `style.css` with the tokens, type, top bar, and shared pieces (cards, media box, buttons). Add `.nojekyll`.
2. Landing page with the bio, picture, and five cards (four say "Coming soon").
3. Resume page and PDF.
4. Project page template, built with `bap`.
5. Visual embed pattern. Build one sample visual to test it.
6. Run the quality bar.
7. Cutover (next section).
8. After launch, add each project page as its grilling and visuals are finished, and link its card.

The prototype branches are throwaway reference. Copy decisions from the tickets and this spec, not files.

## 11. Cutover

The old site is an 811-byte placeholder with dummy links. Pages builds from the root of `main` (legacy build, HTTPS on, no custom domain). Nothing real links to the old site, so there are no redirects.

**Before:** build on `site`. Preview locally with `python3 -m http.server` from the repo root, on desktop and on a phone on the same network. Launch needs the landing page, the resume page with its PDF, and the `bap` project page real. Run the quality bar.

**Go live:**
1. Tag current `main` as `old-site` and push the tag.
2. Merge `site` into `main` once. Push.

**After:**
1. Wait for the Pages build to show `built` (`gh api repos/austinbreunig/austinbreunig.github.io/pages`).
2. Open the live URL on desktop and a phone. Hard refresh. Click every link, open the PDF, confirm the guided visuals load over HTTPS.
3. Delete the six `prototype/*` branches on the remote. Keep the `old-site` tag.

**Rollback:** `git revert` the merge commit and push. The old placeholder is back in about a minute.

## Launch changes (set at sign-off)

- No stand-in profile picture. Launch uses the photo chosen in the profile ticket (#8).
- No stand-in resume text. The resume page is built only from the author's real resume text.
- The resume PDF and the Download PDF button wait for a later pass. They are not in the launch.
- All other open-question defaults stand.
