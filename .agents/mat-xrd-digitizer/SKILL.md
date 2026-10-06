---
name: mat-xrd-digitizer
description: Digitize an image of an XRD plot into a numeric .xy data file by extracting visual peaks.
category: materials
---

# XRD Digitizer

## Goal

Convert an image or screenshot of a **single-curve** X-ray diffraction (XRD) pattern into a digitized numeric `.xy` file for downstream tools like `mat-xrd-phase-analysis`.

The agent visually extracts peak positions (2θ) and relative intensities, then `digitize_plot.py` builds a representative pseudo-Voigt profile.

## Instructions

### 1. Extract peaks visually

Provide an XRD plot image. The agent identifies **every visible peak**, including tiny minor peaks.

Create a JSON peak list (e.g. `peaks.json`) and keep a copy of the original image beside it.

```json
[
  {"2theta": 8.8, "intensity": 0.05, "fwhm": 0.3},
  {"2theta": 15.8, "intensity": 0.08, "fwhm": 0.3},
  {"2theta": 33.1, "intensity": 1.00, "fwhm": 0.3}
]
```

- `intensity` is normalized 0–1.0 (tallest peak = 1.0)
- `fwhm` defaults to 0.3

This skill handles **one curve only** (CNRS-style single patterns).

### 2. Generate the digitized `.xy` file

```bash
# Env: base-agent
python .agents/mat-xrd-digitizer/scripts/digitize_plot.py peaks.json \
  --output digitized_plot.xy \
  --min-x 5.0 \
  --max-x 80.0
```

| Parameter | Meaning | Default |
|---|---|---|
| `input` | Peak JSON list | required |
| `--output` | Output `.xy` path | `digitized.xy` |
| `--min-x` / `--max-x` | 2θ range | 5 / 90 |
| `--points` | Number of samples | 4000 |
| `--noise` | Noise amplitude | 0.01 |
| `--background` | Exponential baseline amplitude | 0.05 |

Also writes `<output_stem>.png` preview next to the `.xy`.

## Autonomous OpenAI pipeline

Fully autonomous path (no interactive agent): vision extracts peaks, then runs `digitize_plot.py`.

```bash
# Env: base-agent
export OPENAI_API_KEY=...
python .agents/mat-xrd-digitizer/scripts/run_openai_digitize.py path/to/figure.png
```

Writes `figure/figure.json` (peaks), `figure_digitized.xy`, `figure_digitized.png`.

### CNRS benchmark batch

```bash
# Env: base-agent
python .agents/mat-xrd-digitizer/scripts/run_openai_digitize.py data/CNRS_figures --png-dir \
  --output-dir data/CNRS_digitized_agent \
  --json-dir data/CNRS
```

Per figure:

```text
data/CNRS_digitized_agent/figure_N_digitized/
  pattern_N.png
  pattern_N.json              # peak list
  figure_N.csv
  figure_N_digitized.xy
  figure_N_digitized.png
  figure_N_overlay.png        # SID vs data/CNRS/pattern_N.json
```

Useful flags: `--limit N`, `--overwrite`, `--request-delay 2`, `--max-retries 10`,
`--from-json-dir path/to/peaks/` (offline, no API).

## Constraints

- **Approximation**: pseudo-Voigt reconstruction, not pixel-perfect curve tracing.
- **Vision accuracy**: 2θ quality depends on axis clarity.
- **Single curve only**: multi-curve / multi-panel figures are out of scope.
- Scripts expect the `base-agent` Conda environment.
