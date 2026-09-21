# Autonomous Scientific Figure Digitizer

Fully autonomous pipeline that digitizes scientific figures into numerical XY data.

**Design principle:** OpenAI vision performs semantic interpretation and planning. Local computer-vision and numerical code perform pixel-level measurement and curve extraction. The model does **not** hallucinate dense curve coordinates.

This project is standalone and does not depend on any other digitization codebase.

## Installation

```bash
cd autonomous_digitizer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="..."
# optional:
export OPENAI_MODEL="gpt-4.1"
python digitize.py example.png --debug
```

Copy `.env.example` to `.env` if you prefer dotenv loading (`python-dotenv` is used by the CLI).

## CLI

```bash
python digitize.py path/to/figure.png

python digitize.py figure.png \
  --output outputs/run1 \
  --model gpt-4.1 \
  --max-retries 2 \
  --samples 1000 \
  --debug

python digitize.py figure.png --no-cache
python digitize.py figure.png --offline   # no API calls (limited fallbacks / tests)
```

## Pipeline architecture

| Stage | Role |
|------|------|
| 1 Input | Load PNG/JPG (TIFF/BMP/WebP supported), preserve original |
| 2 Figure analysis | OpenAI structured JSON: layout, subplots, shared axes |
| 3 Segmentation | Refine LLM boxes with whitespace/ink CV; crop subplots |
| 4 Subplot analysis | OpenAI per subplot: axes, legends, curve separation |
| 5 Plot area + ticks | Local spine detection + tick calibration (LLM ranges as anchors) |
| 6 Shared axes | Inherit x/y mappings across aligned subplots |
| 7–10 Curves | Strategy engine → color / continuity / stacked tracing → CSV |
| 11–12 Outputs | `digitized_figure.json`, reconstructions, comparisons |
| 13–14 QA | Local coverage metrics + OpenAI visual QA with finite retries |

### Strategy engine

Given subplot analysis, the pipeline autonomously selects:

- `single` — one curve, column-wise median trace
- `color` — LAB/KMeans segmentation then per-cluster trace
- `continuity` — multi-trajectory tracking for same-color curves
- `stacked` — continuity with vertical ordering; **displayed** values only (no silent offset removal)

Chosen strategy and reason are stored in `report.json`.

### Shared axes

If subplot A lacks visible x-tick labels but shares an x-axis with B:

1. Calibrate B
2. Transfer the **data range** onto A's plot rectangle (not raw pixels)
3. Record `calibration_type: shared_axis_inheritance` and `inherited_from`

### Multiple curves

Color-separated series use perceptual LAB clustering while excluding background, near-black axes, and annotation masks. Same-color series use column-wise candidates with continuity penalties. Dashed/overlapping curves remain a hard case (see Limitations).

### QA / retries

After extraction, local coverage is computed and (when online) OpenAI compares original vs reconstruction. On failure, up to `--max-retries` alternate strategies are tried (`recolor_cluster`, `continuity_trace`, etc.). The loop is finite.

## Outputs

```text
outputs/example/
  original/figure.png
  analysis/figure_analysis.json
  analysis/subplot_A_analysis.json
  subplots/subplot_A.png
  data/subplot_A/curve_001.csv
  reconstructions/reconstructed_A.png
  reconstructions/reconstructed_full_figure.png
  comparisons/comparison_A.png
  debug/...
  digitized_figure.json
  report.json
  pipeline.log          # with --debug
```

OpenAI responses are cached under `.cache/<image_hash>/` unless `--no-cache`.

## Tests

Synthetic matplotlib figures with known ground truth:

```bash
cd autonomous_digitizer
pip install -r requirements.txt
pytest -q
```

Coverage includes single/multi curves, 2×2 panels, shared x/y, markers, dashed, stacked offsets, annotations, legends, and log axes, with RMSE-style metrics where applicable.

## Limitations (honest)

- Tick **values** currently rely primarily on OpenAI-provided axis ranges / tick hints mapped onto CV-detected tick **positions**. Full local OCR of tick labels is not yet a primary path.
- Same-color dashed lines and heavily overlapping traces are heuristic and may return `partial`.
- Vertical offsets are detected and flagged; automatic offset *removal* is not performed unless future logic can infer it reliably.
- Bar charts, heatmaps, 3D plots, and images-as-plots are typically marked `not_digitizable`.
- Plot-area detection is CV-heuristic and can fail on unusual spines or heavy grid art.
- OpenAI structured-output schema strictness can vary by model; the client falls back to JSON mode and a repair pass.

## Example invocation

```bash
export OPENAI_API_KEY="sk-..."
python digitize.py /path/to/xrd_figure.png --output outputs/xrd1 --debug --samples 800
```

## License

Use within your own research / tooling constraints. API usage is billed to your OpenAI account.
