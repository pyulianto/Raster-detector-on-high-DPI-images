# Raster detector on high DPI images

Streamlit application to detect repeated engineering symbols (e.g., screws/towers) in high‑DPI raster drawings. The app uses an edge‑based distance‑transform matcher that is scale‑ and small‑rotation tolerant, plus geometric verification to minimize false positives. It supports multiple reference templates loaded from the `refs/` folder, with per‑template grouped results and a combined overlay.

## Features

- High‑DPI friendly: processes the full‑resolution scene (no forced downscale)
- Robust matching: distance‑transform on edges, multi‑scale, small rotation set
- Geometric verification: aspect‑ratio, verticalness (for tower profile), boundary alignment using template‑edge DT, border margin gate
- Multi‑template: drop reference PNG/JPGs in `refs/` and tick which ones to include
- UI conveniences: 150×150 framed thumbnails with checkboxes, grouped tables, rich debug (raw before NMS, after NMS, verification details), JSON export of final detections

## Requirements

```
python>=3.10
streamlit
opencv-contrib-python-headless
numpy
pillow
```

Install with:

```bash
pip install -r requirements.txt
```

## Project layout

```
.
├─ app_ght.py               # Streamlit app (fixed-mode parameters)
├─ requirements.txt
├─ MasterImage.png          # Example scene (used by the app)
└─ refs/                    # Put your reference templates here
   ├─ Tower.png
   └─ ScrewHorizontal.png   # (example) any PNG/JPG/JPEG/BMP/WEBP
```

Notes:
- Reference images may have transparency; the alpha channel is used as a mask.
- The app will rotate references heuristically when the filename contains `horiz` or the image is wider than tall (to match the “tower” profile without changing global rotations).

## How to run

```bash
streamlit run app_ght.py
```

Then open the URL shown (usually `http://localhost:8501`). The app loads `MasterImage.png` by default and scans the `refs/` directory for templates.

### Using multiple references
1. Place one or more template files in `refs/` (PNG/JPG/JPEG/BMP/WEBP).  
2. Start the app; thumbnails (150×150) appear with “Include” checkboxes (checked by default).  
3. Click “Process” to detect all checked templates.  
4. View:
   - Combined overlay with per‑template colors
   - “Candidates before NMS,” “Candidates after NMS,” “Accepted candidates” tables
   - Download final detections (JSON) and annotated PNG

## Troubleshooting

- No detections, but candidates exist before NMS: increase the shortlist (`TOP_K_AFTER_NMS`) or enable spacing-based shortlist if many near‑duplicates crowd the top‑K.
- Good candidates rejected by verification with `tpl_boundary_dt=…`: either improve local edge contrast (CLAHE + slightly lower Canny thresholds inside the candidate patch) or raise `template_edge_dt_max` cautiously.
- False positives along page borders: the app uses a border‑margin gate; ensure your templates mask out background and keep the symbol tight.

## Development

Push current branch to GitHub:

```bash
git add .
git commit -m "Update"
git push
```

Repository: `https://github.com/pyulianto/Raster-detector-on-high-DPI-images`  
Default branch: `main`

## License

This project is provided as‑is without warranty. Add a license file if you need specific terms.


