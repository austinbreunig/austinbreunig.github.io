# conservation-dashboard

Repo: https://github.com/austinbreunig/conservation-dashboard

**conservation-dashboard** is a cloud-hosted dashboard of moose sightings in Boulder County, refreshed monthly by a scheduled pipeline.

It's an example of using GCP cloud resources to update a web map on a schedule. It's for hiring managers and clients first, GIS people second.

Live dashboard: https://conservation-dashboard-hjpsz7yrsq-uc.a.run.app (open, no login)

## Free block (your words, 2-4 sentences)

(not written yet. Prompt: you were exploring Cloud Run and Cloud Scheduler, and what it takes to trigger and publish a pipeline. No client story and no quoted client prompt.)

## Page structure (after the summary, before the repo link)

Editorial rows: gutter label, then content. Plain HTML only: no embedded live app, no sliders, no tabs.

1. *(no label)*: the free block above
2. **The Dashboard**: what it looks like and how it works.
   - A link "Open the live dashboard", with a note that it may take a few seconds to wake up (it scales to zero).
   - A screenshot, and one sentence on how a score is made.
   - Say once that the moose sightings are synthetic (generated for the demo) and the other layers are real.
   - Layers:
     - Trailheads, trail segments, critical wildlife habitats, wildlife corridors, county roads, county boundary (Boulder County ArcGIS)
     - Moose sightings (synthetic)
3. **The Pipeline**: acquire, normalize, 500 m grid, spatial join, score, publish to GeoParquet on GCS
4. **Monthly Refresh**: Cloud Scheduler (1st of each month, 06:00) triggers a Cloud Run Job, which writes to GCS; the dashboard reads it
5. **Architectural Decisions**: GeoParquet + DuckDB over PostGIS, scale-to-zero, no data cache, monthly not weekly
6. **Status**: what is built and what is next

## Visual list

| Section | Visual | Type |
|---|---|---|
| The Dashboard | main map with scored cells and moose | screenshot (you) |
| The Dashboard | the score breakdown or "how is this calculated" note | screenshot (you) |
| The Pipeline | raw layers to scored cells | guided visual (agent, `human` skill) |
| Monthly Refresh | Scheduler, Job, GCS, dashboard | guided visual (agent, `human` skill) |

## Screenshot checklist

Open the live dashboard. Light theme, 1200px wide.
1. **Main map**: default view, with scored cells and moose points visible and the legend in frame.
2. **Score note**: open the "how is this calculated" note and capture it.

## Guided visual briefs (format from the embedding ticket)

**layers-to-scores**
- Question: How do seven layers become one score per 500 m cell?
- Reader: a hiring manager or GIS person who has not seen the repo.
- Steps: acquire, normalize, grid, spatial join, score (density times a weighted mix of proximity to habitat, corridor, trail, road), publish.
- Source: repo `spec/architecture.md` and `config/scoring.yaml`.

**monthly-refresh**
- Question: What happens on the 1st of each month to update the map?
- Reader: same.
- Steps: Cloud Scheduler cron `0 6 1 * *`, OAuth call to the Cloud Run Job, the Job writes GeoParquet to GCS, the dashboard service reads it, scale-to-zero when idle.
- Source: repo `docs/decisions/cloud-run-deployment.md` and `docs/runbooks/deploy.md`.

## Follow-up outside this ticket

The repo README says "weekly-refreshing" and describes a nonprofit client. The pipeline runs monthly. Fix the README before the page links to it.
