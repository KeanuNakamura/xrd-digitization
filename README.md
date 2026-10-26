# XRD PDF Scraping

Extract figures, captions, and surrounding context from scientific PDFs, then enrich each crop with structured vision analysis. The default path stops at scrape + metadata — curve digitization is an optional bonus.

```
PDF → GROBID parse → figure crops + captions + context
    → OpenAI vision analysis → <stem>.figure_analysis.json
    → (optional) --digitize → calibrated CSV / preview PNGs
```

## Setup

```bash
git clone --recurse-submodules git@github.com:KeanuNakamura/xrd-digitization.git
cd xrd-digitization
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you already cloned without submodules: `git submodule update --init`.

Requires a running [GROBID](https://grobid.readthedocs.io/) server and `OPENAI_API_KEY` for figure analysis.

## Usage

### PDF → scrape → analysis (default)

```bash
export OPENAI_API_KEY=...

python scripts/scrape_and_digitize.py paper.pdf output/
python scripts/scrape_and_digitize.py pdf_files/ output/
```

For each PDF, writes `output/<stem>/`:

| Path | Contents |
|------|----------|
| `figures/<id>.png` | Cropped figure images |
| `<stem>.figure_analysis.json` | Per-figure path, caption, paper context, and `analysis` metadata |
| `extra/` | Original PDF plus GROBID TEI / parse records |

Digitization is **off** unless you pass `--digitize`.

### Output shape

Each entry in `<stem>.figure_analysis.json` looks like:

```json
{
  "figure": 1,
  "figure_path": "figures/fig_1.png",
  "caption": "Figure 1. XRD spectra as a function of …",
  "text": "…surrounding paper context…",
  "analysis": {
    "is_xrd": true,
    "sample": "…",
    "curves": [{ "condition": "…", "peak_positions": [], "phases": [] }],
    "trends": ["…"]
  }
}
```

Paper-level synthesis (when available) is stored alongside as `paper_analysis`.

See [`examples/scrape/`](examples/scrape/) for sample XRD crops and a portable JSON sketch, and [`examples/DEMO.md`](examples/DEMO.md) for a walkthrough.

### Useful flags

| Flag | Effect |
|------|--------|
| `--grobid-url URL` | GROBID base URL (default `http://localhost:8070`) |
| `--figure-dpi N` | Render DPI for crops (default `300`) |
| `--model NAME` | OpenAI vision model override |
| `--overwrite` | Replace an existing `output/<stem>/` |
| `--benchmark` | Crop all figures, classify XRD vs not, copy into global `xrd/` / `not_xrd/` (no digitize) |
| `--digitize` | After scrape+analysis, triage and digitize XRD curves (bonus) |
| `--all-figures` | With `--digitize`: digitize every crop, not only XRD-classified ones |

### Analyze already-cropped figures

```bash
export OPENAI_API_KEY=...
python scripts/analyze_figures.py path/to/figure_pngs/ output/figure_metadata/
```

## Demo

The nanosized-powders paper mixes TEM, SAXS, and XRD. After scrape + analysis, the XRD-related crops (e.g. figures 1 and 5) land under `figures/` with captions and context in `figure_analysis.json`:

<table>
  <tr>
    <td align="center"><strong>Figure 1</strong><br/><img src="examples/scrape/fig_1.png" alt="Scraped XRD figure 1" width="100%"/></td>
    <td align="center"><strong>Figure 5</strong><br/><img src="examples/scrape/fig_5.png" alt="Scraped XRD figure 5" width="100%"/></td>
  </tr>
</table>

Full walkthrough: [`examples/DEMO.md`](examples/DEMO.md).

---

## Bonus: curve digitization

Optional path that turns a single-curve (or stacked shared-x) XRD PNG into calibrated `(x, y)` CSV plus a preview PNG. Needs [Tesseract](https://github.com/tesseract-ocr/tesseract) for axis OCR (`brew install tesseract` on macOS). `CLIPDROP_API_KEY` is only required when in-plot text must be removed.

### End-to-end with scrape

```bash
export OPENAI_API_KEY=...
export CLIPDROP_API_KEY=...   # only if a figure needs ClipDrop

python scripts/scrape_and_digitize.py paper.pdf output/ --digitize
```

With `--digitize`, non-XRD crops move to `non_xrd_figures/`; XRD crops are triaged and digitized under `xrd_figures/` (use `--all-figures` to skip the XRD filter).

### Digitize one PNG

```bash
python scripts/digitize_figure.py examples/figure_3.png output/
# → output/figure_3/figure_3.csv
# → output/figure_3/figure_3_digitized.png
```

Add `--clipdrop` to remove in-plot text before tracing (uses ClipDrop credits).

### Digitization examples

CNRS figure 1034 — original, reconstructed curve, overlay (SID ≈ 1.02):

<table>
  <tr>
    <td align="center" width="33%"><strong>Original</strong></td>
    <td align="center" width="33%"><strong>Digitized</strong></td>
    <td align="center" width="33%"><strong>Overlay</strong></td>
  </tr>
  <tr>
    <td align="center"><img src="examples/figure_1034_original.png" alt="Original XRD figure 1034" width="100%"/></td>
    <td align="center"><img src="examples/figure_1034_digitized.png" alt="Digitized XRD curve for figure 1034" width="100%"/></td>
    <td align="center"><img src="examples/figure_1034_overlay.png" alt="Overlay of original vs digitized figure 1034" width="100%"/></td>
  </tr>
</table>

CNRS figure 1051 — original, reconstructed curve, overlay (SID ≈ 0.68):

<table>
  <tr>
    <td align="center" width="33%"><strong>Original</strong></td>
    <td align="center" width="33%"><strong>Digitized</strong></td>
    <td align="center" width="33%"><strong>Overlay</strong></td>
  </tr>
  <tr>
    <td align="center"><img src="examples/figure_1051_original.png" alt="Original XRD figure 1051" width="100%"/></td>
    <td align="center"><img src="examples/figure_1051_digitized.png" alt="Digitized XRD curve for figure 1051" width="100%"/></td>
    <td align="center"><img src="examples/figure_1051_overlay.png" alt="Overlay of original vs digitized figure 1051" width="100%"/></td>
  </tr>
</table>

#### Figures with in-plot text

With `--clipdrop`, in-plot labels are cleared, then the cleaned interior is pasted back onto the original axes so tick OCR still works:

<table>
  <tr>
    <td align="center" width="25%"><strong>Original</strong></td>
    <td align="center" width="25%"><strong>Clean</strong></td>
    <td align="center" width="25%"><strong>Axes restored</strong></td>
    <td align="center" width="25%"><strong>Digitized</strong></td>
  </tr>
  <tr>
    <td align="center"><img src="examples/figure_8_original.png" alt="Original figure 8 with Miller index labels on peaks" width="100%"/></td>
    <td align="center"><img src="examples/figure_8_clean.png" alt="ClipDrop-cleaned plot interior" width="100%"/></td>
    <td align="center"><img src="examples/figure_8_axes_restored.png" alt="Cleaned interior on original axes" width="100%"/></td>
    <td align="center"><img src="examples/figure_8_digitized.png" alt="Digitized calibrated XRD curve for figure 8" width="100%"/></td>
  </tr>
</table>

> Monshi, A., Foroughi, M. R., & Monshi, M. R. (2012). Modified Scherrer Equation to Estimate More Accurately Nano-Crystallite Size Using XRD. *World Journal of Nano Science and Engineering*, *2*, 154–160. https://doi.org/10.4236/wjnse.2012.23020

### Spectral Information Divergence (SID)

SID scores digitized spectra against ground truth (`compute_sid.py`). Lower is better; overlays report raw/modified SID, peak F1, and a final XRD score written to `<figure_id>_scores.json` / `sid_summary.json`.
