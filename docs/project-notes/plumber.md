# plumber

Repo: https://github.com/austinbreunig/plumber
PyPI: https://pypi.org/project/ab-plumber/

## Title and summary

**plumber**: A pipeline framework making it easy to wire together a complete and scalable workflow.

Audience: engineers, leaning towards geospatial use cases.

## Free block (the author's own words, unlabeled, verbatim)

One of the biggest challenges I've faced is scaling a small idea from proof of concept quickly to a reliable and performant pipeline. Many geospatial projects I worked on seem to invariably blend the geospatial processing and business logic with the pipeline machinery itself, creating hard to reverse coupling. The result of this is a pipeline tailored specific to the logic it runs, and if that logic changes the pipeline itself must be changed too. To have a simple to use, simple to understand, and reliable framework which sees the code without affecting it or being effected by it, is the main goal of plumber. Shims or adapters are used to take any codebase and make it compatible with plumber. The idea is that there is a common interface between the code plumber runs and plumber itself. plumber acts as a flexible delivery driver simply running it in the order you define. An easy to define configuration file is used by plumber to state each phase and the order of the workflow. An execution engine is defined here, where the current options are local (sequential running on local machine) and localmp (utilizing local multiprocessing partitioned on a user defined argument). Added execution engines for pyspark and beam will later be added.

Tools (author's words): Python. The current interface takes a dataframe (or geodataframe). Pyspark and beam are up next for improvements.

Fact fix agreed: the author said parquet I/O is required. In fact the input and output functions are the user's own; plumber writes GeoParquet only for checkpoints and the optional check sample. The page says that.

(Typos above are lightly fixed; the author writes the final page words.)

## Page structure (after the summary, before the repo link)

Editorial rows: gutter label, content right. Plain HTML + inline hand-written JavaScript, no libraries, no server. Every widget shows its plain result without JavaScript. Widgets are inline, not iframes.

1. *(no label)*: the free block above
2. **The idea**: the phase contract `run(gdf, **params) -> gdf` and the adapter shim. Widget: clickable pipeline (click a phase to see its signature and config entry).
3. **A pipeline in 12 lines**: the YAML example from the README (same clickable pipeline ties to it).
4. **Same code, new strategy**: `local` to `localmp`. Widget: strategy switch. A toggle shows the one-line YAML change and a small animation of rows going into partitions, out to workers, and back in original row order. Screenshot of a real `plumber run --strategy localmp` log with Phase headings.
5. **Check before you run**: `plumber check`. Widget: check playground (pick good phase / wrong signature / bad CRS, see PASS/WARN/FAIL). Screenshot of a real `plumber check` run.
6. **Built with plumber**: bap, one sentence, linking to the bap page.
7. **Next**: PySpark and Beam engines, "next, not built".

Status line: "v0.1.2, one user (me), local and multiprocess only." No scale numbers (bap's benchmark stays on the bap page).

Links: PyPI (`ab-plumber`), bap page, repo (fixed). No Install row (README has it). No mention of the `/plumber-setup` skill.

## Visual list

| Row | Visual | Type |
|---|---|---|
| Same code, new strategy | `plumber run --strategy localmp` log with Phase headings | screenshot (you) |
| Check before you run | `plumber check` showing PASS/WARN/FAIL | screenshot (you) |

No guided visual: the strategy switch replaces it.

## Screenshot checklist

Run in a plumber repo with a multi-phase `plumber.yaml` (for example the bap pipeline). Dark or light terminal to match the site, readable at 1200px wide, no private paths in view.

1. **Run log**: `plumber run --strategy localmp --worker-count 4`. Capture the Run log so the Phase headings show.
2. **Check**: `plumber check`. Capture the PASS/WARN/FAIL lines. If you can, set up one WARN or FAIL so all states show.
