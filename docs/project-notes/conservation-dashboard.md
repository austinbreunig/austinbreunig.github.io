# conservation-dashboard

Repo: https://github.com/austinbreunig/conservation-dashboard

Notes from the grilling session (ticket #11). The author writes the page words; this file only records answers.

## Decided

- **Status on the page:** current state is stable. Note planned improvements and further work.
- **Audience:** hiring managers and clients.
- **Summary (draft, reword in your own voice):** A weekly-refreshing dashboard that brings a conservation nonprofit's spreadsheets together to show where to focus restoration.
- **Client:** (confirm: real or practice brief? Say so plainly on the page.)
- **Tools to name:** GeoPandas, DuckDB with GeoParquet, Streamlit, Folium, Cloud Run / GCS.
- **Left out on purpose:** PostGIS / Cloud SQL. Reason: no permanent free tier and it costs money while idle. GeoParquet on GCS plus DuckDB is nearly free at this data size and needs no server. PostGIS can be added later since the data is open GeoParquet. (Source: `spec/architecture.md` in the repo. Put it in your own words.)

## To do before the card links to the page

- Add a GitHub repo description.
- Make the README visitor-ready (it is mostly setup notes now).

## Free block (author's words)

(not written yet — why you built it, what should be true for the nonprofit once it works, where it got hard, what you learned, planned improvements and further work)

## Page layout

(to decide in this grilling: which sections, headers and functionality fit this project. No fixed Workflow section.)

## Visual list

(not decided yet)
