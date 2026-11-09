# bap

Repo: https://github.com/austinbreunig/bap

**BAP (Big Ass Polygons)** clusters nearby polygons and wraps each cluster in a tight, square-edged boundary.

It's for GIS people with thousands of small polygons who want a few simple outlines. This project was also used to showcase the functionality of plumber.

Delineating new boundaries on aggregate fields is a common pattern in organizing and restructuring geospatial data. I wanted to work on building a clean interface between the core algorithm (DBSCAN-like clustering) and a CLI. The package is simple, but user-friendly. The simplicity of the package and workflow was intentional, as the core intention was to showcase (and test) the functionality of plumber.

## Page structure (after the summary, before the repo link)

Editorial rows: gutter label, then content. Plain HTML, no sliders or interactive maps, no install or flag table (the README has them), no Try it row.

1. *(no label)*: the free block above, unchanged
2. **Before and after**: two images side by side (stacked on a phone): the sample input polygons, then the same data with the cluster boundaries over it. Comes first because it explains bap at a glance.
3. **How it works**: the pipeline: validate, project to UTM, cluster, boundary, tightness gate (>= 0.75), IDs, project back
4. **Scale**: measured numbers only, from `docs/benchmarks.md`: 1,000,000 polygons in 254 s, 1.5 GB peak memory. No mention of the unrun giant-cluster case or the 5M estimate.
5. **Built with plumber**: one sentence. Plain text with no link until the plumber page is live.

## Visual list

| Section | Visual | Type |
|---|---|---|
| Before and after | sample input polygons (4 blocks plus 10 strays) | screenshot (you) |
| Before and after | same data with cluster boundaries over it | screenshot (you) |
| How it works | pipeline stages with one polygon group shown at each stage | guided visual (agent, `human` skill) |

## Screenshot checklist

Use `data/sample/sample.parquet` in the bap repo and the output of `bap data/sample/sample.parquet out.parquet --key-field zone --id-field parcel_id`. Use the same map view and zoom for both shots so they line up. Light theme, 1200px wide.

1. **Input**: open the sample file in any viewer (geopandas `.plot()` or QGIS). Capture all 50 polygons.
2. **Output**: same view, with the polygons plus the `out.parquet` boundaries drawn over them. Capture it.

## Guided visual brief (format from the embedding ticket)

**how-bap-works**
- Question: How does bap turn thousands of polygons into a few tight outlines?
- Reader: a GIS person who knows polygons but not clustering.
- Steps: validate, project to UTM, cluster (per epsilon), boundary, tightness gate, IDs, project back.
- Source: bap `README.md` (pipeline) and `CONTEXT.md` (terms).
