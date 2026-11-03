# ImageScape tours

Interactive 360 tour. Drag to look around, tap rings to move between scenes, play a guided tour, walk the whole route,
or record the guided tour as a video (the Save As dialog opens in Documents in Chrome and Edge).

## Publish with GitHub Pages
Repo **Settings > Pages > Build and deployment > Source: Deploy from a branch > Branch `main`, folder `/docs`**.
The tour appears at `https://<user>.github.io/imagescape-tours/`.

## Layout
- `ImageScape_360_NNN.jpg` the full-resolution originals (not served by Pages)
- `docs/` the website: `index.html`, `app.js`, `scenes.js`, `links.json`, and `img/` (8192 px + preview of every scene)
- `tools/build_scenes.py` makes the 8192 px and preview images from the originals
- `tools/estimate_all.py` measures where each photo was taken relative to its neighbours (writes `docs/links.json`)
- `tools/build_site.py` rebuilds `docs/scenes.js` from the images, the measurements and `tools/site_config.json`
- `tools/site_config.json` scene titles, start scene, the guided tour, and hand-placed links between distant scenes

## Adding or changing photos
1. Put new `ImageScape_360_NNN.jpg` files in the repo root.
2. `python3 tools/build_scenes.py`, `python3 tools/estimate_all.py`, `python3 tools/build_site.py`
3. Commit and push; Pages republishes in a minute or two.

## Transitions
Between scenes the viewer picks one of:
- geometry move: the camera travels between the two measured photo positions, both photos reprojected onto an
  estimated room box (used when the measurement is confident)
- zoom-blur dissolve: for pairs too different to measure
- plain turn: when two scenes share one photo
