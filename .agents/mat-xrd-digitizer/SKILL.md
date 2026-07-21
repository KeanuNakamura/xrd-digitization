name	mat-xrd-digitizer
description	Digitize XRD plot figures into numeric peak JSON and .xy files, including batch processing of GROBID-extracted figures and multi-curve plots.
category
materials

# XRD Digitizer

## Goal

Convert XRD plot images into numeric digitized data files that can be used by downstream analysis tools like `mat-xrd-phase-analysis`.

This skill supports two workflows:

1. **Single-image digitization**: digitize a specific user-provided XRD plot image.
2. **Batch GROBID figure digitization**: automatically scan all extracted figures in:

```text
grobid_output/sample_pdfs/{example_dir}/figures/
```

for every `{example_dir}` under:

```text
grobid_output/sample_pdfs/
```

and digitize all XRD-like figures found.

The skill uses the AI Agent's built-in Vision/Language Model capabilities to visually inspect XRD plots, identify axes, detect curves, extract peak positions and approximate relative intensities, and then use the provided digitization script to generate representative `.xy` profiles.

The digitized `.xy` output is an approximation generated from extracted peaks using pseudo-Voigt peak profiles. It is intended for downstream phase matching, indexing, and approximate XRD analysis.

---

## Instructions

## 1. Locate Figures to Digitize

### Batch GROBID Workflow

Unless the user provides a specific image path, the agent should batch-process figures from:

```text
grobid_output/sample_pdfs/
```

For each directory:

```text
grobid_output/sample_pdfs/{example_dir}/
```

the agent should look for figures inside:

```text
grobid_output/sample_pdfs/{example_dir}/figures/
```

The agent should recursively or directly inspect image files in each `figures` directory. Supported image extensions include:

```text
.png
.jpg
.jpeg
.tif
.tiff
.webp
```

For each figure image, the agent should determine whether it appears to contain an XRD pattern or diffraction-related plot. The agent should prioritize figures with:

* 2θ, two theta, theta, diffraction angle, q, or similar x-axis labels
* intensity, counts, arbitrary units, normalized intensity, or similar y-axis labels
* sharp diffraction peaks
* stacked XRD patterns
* multiple colored or offset curves
* labels such as XRD, PXRD, diffraction pattern, Rietveld, observed/calculated/difference, sample names, phases, temperatures, or compositions

If a figure is not an XRD-like plot, the agent should skip it and briefly record that it was skipped if producing a processing summary.

---

## 2. Distinguish Single-Plot vs Multi-Panel Figures

Before extracting peaks, determine whether the image contains:

| Type | Description | JSON structure | Example |
|---|---|---|---|
| **Single plot** | One axes area; may contain one or many curves | Top-level `curves` array | `figure_1`, `figure_6` |
| **Multi-panel** | Two or more separate subplot axes in one image | Top-level `plots` array | `figure_7` |

### Single plot

Use when multiple curves belong to the **same axes**:
* overlay traces (Rietveld observed/calc/diff)
* vertically stacked patterns sharing one x-axis block (temperature series, e.g. `figure_6`)
* multiple samples offset within one panel

### Multi-panel

Use when the figure contains **separate subplot axes**, each with its own y-axis scale and/or title:
* top/bottom comparison panels (e.g. ICSD vs RRUFF in `figure_7`)
* side-by-side XRD panels
* figure composites where each panel is an independent plot

Do **not** encode separate subplots as `curve_layout: "stacked"` curves in one plot. Instead, create one entry per panel in `plots`.

Each plot entry has its own:
* `plot_id`
* `label` (panel title or identifier)
* `position` (`top`, `bottom`, `left`, `right` when visible)
* `x_axis`, `y_axis`
* `curve_layout` (usually `overlay` for one curve per panel)
* `curves` array

---

## 3. Digitize All Curves in Each Figure / Panel

The agent must digitize the entire XRD figure.

If a figure contains multiple curves, the agent must digitize **all visible curves**, not just one selected curve.

Examples of multi-curve figures include:

* multiple colored XRD traces
* vertically stacked XRD patterns
* observed/calculated/difference Rietveld curves
* before/after treatment curves
* curves labeled by sample name, composition, temperature, pressure, or phase
* curves distinguished by color, line style, marker type, or vertical offset

The agent should identify each curve separately and assign it a stable curve identifier.

Recommended curve naming format:

```text
curve_1
curve_2
curve_3
```

If labels are visible, include them as metadata, for example:

```json
"label": "Sample A"
```

If colors are visible, include them as metadata, for example:

```json
"color": "red"
```

If curves are vertically stacked or offset, include their approximate relative position, for example:

```json
"position": "top"
```

### Stacked vs. Overlay Multi-Curve Layouts

Multi-curve XRD figures fall into two common layout styles. The agent must identify which style applies and encode it in the JSON so `digitize_plot.py` can reproduce the figure correctly.

| Layout | Description | Examples |
|---|---|---|
| **stacked** | Curves are vertically offset so they do not overlap; each curve has its own baseline | `figure_2`, `figure_4` |
| **overlay** | Curves share the same baseline and may overlap | Rietveld observed/calc/diff, color-coded traces on one axis |

Set the layout explicitly at the top level:

```json
"curve_layout": "stacked"
```

If omitted, the script auto-detects **stacked** when multiple curves define `position` (`bottom`, `middle`, `top`) or `baseline_offset`. Otherwise it defaults to **overlay**.

#### Stacked curves: baseline and amplitude metadata

For vertically stacked figures, the agent must record where each curve sits on the y-axis:

```json
"baseline_offset": 3500.0,
"intensity_scale": 1600.0
```

* `baseline_offset` — absolute y-axis value of the curve baseline (read from the figure axis when numeric labels are visible).
* `intensity_scale` — approximate peak height above baseline for the tallest peak in that curve (in the same y-axis units).

For stacked figures with arbitrary y-axis units (no numeric tick labels, e.g. `figure_2`), set `position` on every curve and omit `baseline_offset`; the script will auto-space curves. Use `intensity_scale: 1.0` or omit it.

For stacked figures with numeric y-axis labels (e.g. `figure_4` with 0–10000 counts/pixel), **always** provide `baseline_offset` and `intensity_scale` for every curve.

Example stacked figure with numeric y-axis:

```json
{
  "curve_layout": "stacked",
  "y_axis": {"label": "intensity", "unit": "counts_per_pixel", "min": 0.0, "max": 10000.0},
  "curves": [
    {
      "curve_id": "curve_1",
      "label": "Precursor",
      "color": "blue",
      "position": "bottom",
      "baseline_offset": 500.0,
      "intensity_scale": 1250.0,
      "peaks": [{"2theta": 8.5, "intensity": 1.00, "fwhm": 0.35}]
    },
    {
      "curve_id": "curve_2",
      "label": "Intermediate",
      "color": "orange",
      "position": "middle",
      "baseline_offset": 3500.0,
      "intensity_scale": 1600.0,
      "peaks": [{"2theta": 8.5, "intensity": 0.88, "fwhm": 0.35}]
    }
  ]
}
```

For each curve, the agent should extract all visible peaks, including small minor peaks.

Peak extraction must include:

* major peaks
* shoulder peaks
* weak/minor peaks
* peaks in low-intensity regions
* peaks in high-angle regions
* peaks partially overlapping with other peaks, if visually identifiable

The agent should not discard weak peaks simply because they are small. Tiny visible peaks are important for downstream phase matching and refinement.

---

## 4. Output File Naming and Location

For every original figure image, output files must be saved in a new directory inside of the directory containing the original image. 

For example, if the original image is:

```text
grobid_output/sample_pdfs/{example_dir}/figures/figure_1.png
```

then the agent must write:

```text
grobid_output/sample_pdfs/{example_dir}/figures/figure_1/figure_1.json
grobid_output/sample_pdfs/{example_dir}/figures/figure_1/figure_1_digitized.xy
```
Additionally, the original figure (figure_1.png) should be moved to the figure_1 directory. 
The output JSON filename must use the original figure basename:

```text
{original_figure_stem}.json
```

The output `.xy` filename must use:

```text
{original_figure_stem}_digitized.xy
```

Examples:

```text
figure_1.png  -> figure_1.json, figure_1_digitized.xy
figure_2.jpg  -> figure_2.json, figure_2_digitized.xy
xrd_plot.png  -> xrd_plot.json, xrd_plot_digitized.xy
```

Do not write all outputs to a shared global directory. Keep every output beside its source figure.

### Multi-panel output naming

For a multi-panel figure such as `figure_7.png`, keep the master JSON at:

```text
sample_figures/figure_7/figure_7.json
```

and write numbered outputs per panel:

```text
sample_figures/figure_7/figure_7_digitized_1.xy
sample_figures/figure_7/figure_7_digitized_1.png
sample_figures/figure_7/figure_7_digitized_2.xy
sample_figures/figure_7/figure_7_digitized_2.png
sample_figures/figure_7/figure_7_1.json
sample_figures/figure_7/figure_7_2.json
```

Numbering follows panel order (`1` = top/first panel, `2` = next panel, etc.).

Single-plot figures keep the existing naming with no numeric suffix:

```text
figure_1.json
figure_1_digitized.xy
figure_1_digitized.png
```

---

## 5. JSON Output Format

The JSON file should contain metadata about the source image and a separate peak list for each curve.

For single-curve figures, still use the same multi-curve structure with one curve.

Example format (overlay — curves share a baseline):

```json
{
  "source_image": "figure_1.png",
  "figure_type": "xrd",
  "curve_layout": "overlay",
  "x_axis": {
    "label": "2theta",
    "unit": "degrees",
    "min": 5.0,
    "max": 80.0
  },
  "y_axis": {
    "label": "intensity",
    "unit": "normalized",
    "min": 0.0,
    "max": 1.0
  },
  "curves": [
    {
      "curve_id": "curve_1",
      "label": "Observed",
      "color": "black",
      "intensity_normalization": "normalized_within_curve",
      "peaks": [
        {"2theta": 8.8, "intensity": 0.05, "fwhm": 0.3},
        {"2theta": 33.1, "intensity": 1.00, "fwhm": 0.3}
      ]
    },
    {
      "curve_id": "curve_2",
      "label": "Calculated",
      "color": "red",
      "intensity_normalization": "normalized_within_curve",
      "peaks": [
        {"2theta": 9.1, "intensity": 0.04, "fwhm": 0.3},
        {"2theta": 32.9, "intensity": 1.00, "fwhm": 0.3}
      ]
    }
  ]
}
```

Example format (stacked — curves vertically offset):

```json
{
  "source_image": "figure_4.png",
  "figure_type": "xrd",
  "curve_layout": "stacked",
  "x_axis": {"label": "2theta", "unit": "degrees", "min": 7.0, "max": 37.0},
  "y_axis": {"label": "intensity", "unit": "counts_per_pixel", "min": 0.0, "max": 10000.0},
  "curves": [
    {
      "curve_id": "curve_1",
      "label": "Precursor",
      "color": "blue",
      "position": "bottom",
      "baseline_offset": 500.0,
      "intensity_scale": 1250.0,
      "intensity_normalization": "normalized_within_curve",
      "peaks": [{"2theta": 8.5, "intensity": 1.00, "fwhm": 0.35}]
    },
    {
      "curve_id": "curve_2",
      "label": "Intermediate",
      "color": "orange",
      "position": "middle",
      "baseline_offset": 3500.0,
      "intensity_scale": 1600.0,
      "intensity_normalization": "normalized_within_curve",
      "peaks": [{"2theta": 17.0, "intensity": 1.00, "fwhm": 0.35}]
    },
    {
      "curve_id": "curve_3",
      "label": "Final",
      "color": "green",
      "position": "top",
      "baseline_offset": 6500.0,
      "intensity_scale": 2700.0,
      "intensity_normalization": "normalized_within_curve",
      "peaks": [{"2theta": 17.0, "intensity": 1.00, "fwhm": 0.35}]
    }
  ]
}
```

Example format (multi-panel — separate subplots in one figure):

```json
{
  "source_image": "figure_7.png",
  "figure_type": "xrd",
  "figure_layout": "multi_panel",
  "notes": "ICSD reference (top) vs RRUFF experimental (bottom).",
  "plots": [
    {
      "plot_id": "plot_1",
      "label": "ICSD icsd_156218",
      "position": "top",
      "curve_layout": "overlay",
      "x_axis": {"label": "2theta", "unit": "degrees", "min": 0.0, "max": 45.0},
      "y_axis": {"label": "intensity", "unit": "normalized", "min": 0.0, "max": 1.0},
      "curves": [
        {
          "curve_id": "curve_1",
          "label": "ICSD icsd_156218",
          "color": "orange",
          "intensity_normalization": "normalized_within_curve",
          "peaks": [{"2theta": 29.6, "intensity": 1.00, "fwhm": 0.25}]
        }
      ]
    },
    {
      "plot_id": "plot_2",
      "label": "RRUFF R060558",
      "position": "bottom",
      "curve_layout": "overlay",
      "x_axis": {"label": "2theta", "unit": "degrees", "min": 0.0, "max": 45.0},
      "y_axis": {"label": "intensity", "unit": "normalized", "min": 0.0, "max": 1.0},
      "curves": [
        {
          "curve_id": "curve_1",
          "label": "RRUFF R060558",
          "color": "blue",
          "intensity_normalization": "normalized_within_curve",
          "peaks": [{"2theta": 24.5, "intensity": 1.00, "fwhm": 0.25}]
        }
      ]
    }
  ]
}
```

Rules:

* `source_image` must be the filename of the original figure.
* `figure_type` should usually be `"xrd"` for digitized XRD figures.
* Use top-level `plots` for multi-panel figures; use top-level `curves` for single-plot figures.
* `figure_layout: "multi_panel"` is recommended when `plots` is present.
* `curve_layout` should be `"stacked"` for vertically offset multi-curve figures and `"overlay"` when curves share a baseline. Use `"auto"` or omit to let the script infer from metadata.
* For stacked figures with numeric y-axis labels, every curve must include `baseline_offset` and `intensity_scale`.
* For stacked figures with arbitrary y-axis units, set `position` on every curve; offsets are auto-computed.
* `x_axis.min` and `x_axis.max` should be estimated from the figure axis if visible.
* If the x-axis range cannot be confidently read, default to:

  * `min = 5.0`
  * `max = 80.0`
* Intensities should be normalized between `0.0` and `1.0`.
* Intensities should be normalized independently for each curve unless the figure clearly uses a shared intensity scale.
* For normalized y-axes (`y_axis.max` ≤ 2 or `unit: normalized`), set `intensity_scale: 1.0` on curves or rely on script auto-scaling.
* For clean simulated/reference patterns (e.g. ICSD), set per-plot `"noise": 0.0` and `"background": 0.0`.
* For noisy experimental patterns (e.g. RRUFF), use small per-plot noise/background values and visually curate relative intensities; automated peak-height extraction from noisy baselines is unreliable.
* In multi-panel comparison figures, use **shared 2θ peak positions** across panels when the patterns represent the same phase, but keep **panel-specific relative intensities**.
* `fwhm` should default to `0.3` unless the visual peak widths suggest a better estimate.
* Every curve must have its own `peaks` list.
* Every visible peak should be reported, including tiny minor peaks.

---

## 6. Generate the Digitized `.xy` File

Use the provided script to generate the `.xy` file from the extracted peaks.

Call the script once per original figure image.

### Single-plot figures

```bash
# Env: base-agent
python .agents/mat-xrd-digitizer/scripts/digitize_plot.py \
  grobid_output/sample_pdfs/{example_dir}/figures/figure_1.json \
  --output grobid_output/sample_pdfs/{example_dir}/figures/figure_1_digitized.xy \
  --min-x 5.0 \
  --max-x 80.0
```

The agent should use the `x_axis.min` and `x_axis.max` values from the JSON when available.

For example, if the extracted JSON says:

```json
"x_axis": {
  "min": 10.0,
  "max": 90.0
}
```

then use:

```bash
# Env: base-agent
python .agents/mat-xrd-digitizer/scripts/digitize_plot.py \
  grobid_output/sample_pdfs/{example_dir}/figures/figure_1.json \
  --output grobid_output/sample_pdfs/{example_dir}/figures/figure_1_digitized.xy \
  --min-x 10.0 \
  --max-x 90.0
```

Parameters:

* `input`: JSON file containing extracted peak parameters.
* `--output`: path to save the resulting `.xy` file.
* `--min-x`: minimum 2θ value to generate.
* `--max-x`: maximum 2θ value to generate.
* `--points`: number of data points in the `.xy` file. Default: `4000`.
* `--noise`: amplitude of experimental noise to add. Default: `0.01`.
* `--background`: amplitude of exponential background baseline. Default: `0.05`.
* `--layout`: curve layout override — `auto` (default), `stacked`, or `overlay`.
* `--stack-gap`: fractional gap between auto-spaced stacked curves. Default: `0.15`.

The script reads `curve_layout`, `baseline_offset`, and `intensity_scale` from the JSON. For stacked figures, y-values in the `.xy` output include the vertical offset so each curve block matches its position in the original figure.

### Multi-panel figures

For JSON with a top-level `plots` array, the script automatically detects the multi-panel layout and writes numbered outputs:

```bash
# Env: base-agent
python .agents/mat-xrd-digitizer/scripts/digitize_plot.py \
  sample_figures/figure_7/figure_7.json \
  --output sample_figures/figure_7/figure_7_digitized.xy \
  --min-x 0.0 \
  --max-x 45.0
```

This produces:
* `figure_7_digitized_1.xy`, `figure_7_digitized_1.png`
* `figure_7_digitized_2.xy`, `figure_7_digitized_2.png`
* `figure_7_1.json`, `figure_7_2.json` (single-plot JSON per panel, enabled by default)

Use `--no-write-plot-jsons` to skip per-panel JSON export.

---

## 7. Multi-Curve `.xy` Formatting

If the figure contains multiple curves, the `.xy` output must preserve curve separation.

The preferred `.xy` format is a comment-delimited multi-block file:

```text
# source_image: figure_4.png
# curve_id: curve_1
# curve_layout: stacked
# label: Precursor
# color: blue
# position: bottom
# baseline_offset: 500.0
# columns: 2theta intensity
7.000 512.340
7.008 518.152
...

# source_image: figure_4.png
# curve_id: curve_2
# curve_layout: stacked
# label: Intermediate
# color: orange
# position: middle
# baseline_offset: 3500.0
# columns: 2theta intensity
7.000 3512.102
7.008 3520.887
...
```

Each curve should have its own block. For **stacked** layouts, intensity values include the vertical `baseline_offset` so the data matches the original figure. For **overlay** layouts, all curves share a baseline near zero.

The final output must still be one `.xy` file per original figure:

```text
figure_1_digitized.xy
```

For **multi-panel** figures, use numbered files instead:

```text
figure_7_digitized_1.xy
figure_7_digitized_2.xy
```

and that file must contain all digitized curves from the figure.

The agent should not overwrite one curve with another.

---

## 8. Expected Batch Processing Behavior

When asked to digitize all GROBID-extracted sample figures, the agent should perform the following high-level process:

```text
for each example_dir in grobid_output/sample_pdfs:
    figures_dir = grobid_output/sample_pdfs/{example_dir}/figures

    if figures_dir does not exist:
        skip example_dir

    for each image file in figures_dir:
        if image is not an XRD-like plot:
            skip image

        visually inspect image
        identify x-axis range
        identify all curves
        determine single-plot vs multi-panel layout
        extract all visible peaks for each curve
        save {figure_stem}.json in figures_dir
        generate digitized outputs with digitize_plot.py
```

The agent should produce or summarize:

* number of example directories inspected
* number of figure directories found
* number of figures inspected
* number of XRD-like figures digitized
* number of skipped non-XRD figures
* output paths created
* any figures that were too low-resolution or ambiguous to digitize confidently

---

## 9. Quality Requirements

The agent must prioritize completeness.

For each XRD-like figure:

* Digitize every visible curve.
* Digitize every visible peak.
* Include minor peaks where visible.
* Preserve curve separation in both JSON and `.xy`.
* Save outputs beside the original figure.
* Use the original figure basename for output names.
* Do not require the user to specify which curve to digitize if multiple curves are present.
* Do not digitize only one curve when multiple curves are visible.
* Do not discard stacked, offset, calculated, observed, or difference curves unless clearly irrelevant.

If the figure is too ambiguous, the agent should still make a best effort and include a note in the JSON:

```json
"notes": "Axis labels are partially unreadable; x-axis range estimated from visible ticks."
```

---

## 10. Single-Image Usage

For a single image, the same output naming rules apply.

If the input image is:

```text
path/to/figure_1.png
```

then output:

```text
path/to/figure_1.json
path/to/figure_1_digitized.xy
path/to/figure_1_digitized.png
```

For multi-panel images:

```text
path/to/figure_7.json
path/to/figure_7_digitized_1.xy
path/to/figure_7_digitized_1.png
path/to/figure_7_digitized_2.xy
path/to/figure_7_digitized_2.png
```

The agent should digitize all curves in that image.

---

## 11. Example Command

```bash
# Env: base-agent
python .agents/mat-xrd-digitizer/scripts/digitize_plot.py \
  grobid_output/sample_pdfs/example_paper/figures/figure_1.json \
  --output grobid_output/sample_pdfs/example_paper/figures/figure_1_digitized.xy \
  --min-x 5.0 \
  --max-x 80.0
```

Multi-panel example:

```bash
# Env: base-agent
python .agents/mat-xrd-digitizer/scripts/digitize_plot.py \
  sample_figures/figure_7/figure_7.json \
  --output sample_figures/figure_7/figure_7_digitized.xy \
  --min-x 0.0 \
  --max-x 45.0
```

---

## Constraints

### Approximation

The digitized plot is a mathematical approximation using pseudo-Voigt profiles. It does not perfectly recreate the exact pixel-by-pixel raw data of the original scan, but it is useful for downstream phase matching and approximate refinement.

### Vision Accuracy

The accuracy of peak positions depends on image resolution, plot clarity, visible axis ticks, and whether the curves overlap.

### Multiple Curves

All visible XRD curves must be digitized separately. Multi-curve figures must produce one JSON file containing all curves and one `.xy` file containing separate curve blocks (single-plot), or numbered `.xy`/`.png` files per panel (multi-panel).

For vertically stacked figures, the JSON must include `curve_layout: "stacked"` plus per-curve `baseline_offset` and `intensity_scale` (when the y-axis has numeric labels) or `position` (for arbitrary-unit axes). Without this metadata, the script will overlay all curves at the same baseline and the digitized preview will not match the original figure.

### Environments

Scripts require the `base-agent` Conda environment. Each executable code block must specify:

```bash
# Env: base-agent
```

---

## References

Pseudo-Voigt profile generation is standard practice in XRD peak fitting and Rietveld-style refinement workflows.
