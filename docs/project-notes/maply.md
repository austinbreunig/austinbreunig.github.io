# maply

Repo: https://github.com/austinbreunig/maply

**maply** lets you draw shapes in a Jupyter notebook and get shapely, WKT, GeoJSON, or a GeoDataFrame back.

It's for geospatial engineers who need hand-made geometry for minimal reproducible examples and algorithm tests.

Status: Alpha. `MaplyCanvas` has no CRS.

## Free block (your words, 2-4 sentences)

(not written yet. Prompts: typing coordinate tuples by hand is slow when testing a spatial algorithm. It should feel like a sketchpad: open a cell, draw, get a shapely object, no setup. Why two widgets: a blank XY canvas `MaplyCanvas` and a real map `GeoCanvas`.)

## Page structure (after the summary, before the repo link)

Editorial rows: gutter label, then content. No install or quick-start block (the README has it). No tabs or special behavior: plain HTML.

1. *(no label)*: the free block above
2. **Two canvases**: `MaplyCanvas` vs `GeoCanvas`, side by side
3. **Draw, then export**: one code block showing `.wkt`, `.geojson`, `.geometries`, `.to_geodataframe()` output
4. **Python to map**: `add_geometry` pushes a shapely shape onto the map, edit it, read it back
5. **How it's built**: pure-Python core (model, geometry, transform, interaction) plus thin widget wrappers (render, widget, geocanvas). Tools in one line: ipycanvas, ipyleaflet + Geoman, shapely. The key choice: drawing logic has no widget imports, so it is unit-tested without a Jupyter kernel.
6. **Status**: Alpha. `MaplyCanvas` has no CRS.

No mention of bap or plumber (not confirmed as true). No live demo: the widget needs a live kernel, so the page uses visuals only.

## Visual list

| Section | Visual | Type |
|---|---|---|
| Two canvases | `MaplyCanvas` with a polygon-with-hole | screenshot (you) |
| Two canvases | `GeoCanvas` on a real map with a drawn shape | screenshot (you) |
| Draw, then export | cell output showing `.wkt` and `.geojson` for the drawn shape | screenshot (you) |
| Python to map | shapely polygon pushed onto the map, edited, read back | guided visual (agent, `human` skill) |
| How it's built | pure layers vs UI layers | guided visual (agent, `human` skill) |

## Screenshot checklist

Use `examples/quickstart.ipynb` in the maply repo. Light theme, 1200px wide, crop to the widget.

1. **MaplyCanvas**: run the first cell. Draw a polygon, then draw a smaller polygon inside it so it becomes a hole. Capture the canvas and toolbar.
2. **GeoCanvas**: run the `GeoCanvas` cell. Draw one shape on the map. Capture the map with the Geoman toolbar.
3. **Export output**: after drawing, run `canvas.wkt` and `canvas.geojson`. Capture the cell inputs and outputs together.

## Guided visual briefs (format from the embedding ticket)

**python-to-map**
- Question: How does a shapely shape get onto the map and back?
- Reader: a geospatial engineer who knows shapely but not maply.
- Steps: shapely polygon, `add_geometry`, drawn on the map, edited by hand, `geo.wkt` read back.
- Source: maply `README.md` ("Draw on a real map") and `src/maply/geocanvas.py`.

**how-its-built**
- Question: Why can most of maply be tested without Jupyter?
- Reader: a developer deciding whether to trust or contribute to it.
- Steps: the pure layers (model, tools, transform, interaction, geometry) on one side, the UI layers (render, widget, geocanvas) on the other, and the rule that pure code never imports widgets.
- Source: maply `CLAUDE.md` architecture section.
