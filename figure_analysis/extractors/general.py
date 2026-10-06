"""General scientific-figure extraction instructions."""

from __future__ import annotations


def build_instructions() -> str:
    return """
## Priority (v1)

Prioritize SEMANTIC scientific metadata over pixel-level quantitative estimation:
1. sample / curve identities and labels (exact text only; never invent)
2. phases, Miller indices (hkl), conditions, legends
3. curve roles and experimental-series relationships
4. panel / plot structure
Only then, optionally, approximate peak positions/widths when clearly justified.
Prefer null over uncertain numerical estimates.

## Hierarchy (do NOT flatten)

Figure
  → panels[]          # labeled regions like (a)/(b), or panel_1…
      → plots[]       # independent axes regions WITHIN a panel
          → curves[]
          → annotations[]
          → insets[]
          → series[]  # ordered experimental variable relationships

- panel ≠ plot. A single panel may contain multiple independent plots/axes.
- An inset is NOT an independent panel and usually NOT an independent top-level plot.
  Nest insets under their parent plot.
- Phase-marker symbols, reference sticks/bars, vertical guides, and grid lines are NOT curves.

## Panel rules

1. Detect logical panels. Preserve (a)/(b)/(c) or A/B/C when visible; else panel_1, panel_2, …
2. Set panel_type from: XRD, Raman, FTIR, XPS, PL, UV_Vis, microscopy, crystal_structure, schematic, table, other.
3. Provide panel bbox_normalized [x0,y0,x1,y1] in 0–1 when feasible; else null.
4. Set panel.plot_count and panel.curve_count (visual counts WITH confidence), independently from len(plots)/len(curves).

## Plot rules

1. Within each panel, detect each independent plot/axes region as a PlotResult (plot_1, plot_2, …).
2. If a panel has a single plot, still include plots[0].
3. Non-plot panels (microscopy, crystal_structure, schematic, table) may have plots=[] or one plot with matching plot_type.
4. For every plot set curve_count (visually detected number of data curves) with confidence —
   even if you cannot fully describe every curve in curves[].
5. Separate fields into observed / estimated / interpretation.
6. Use null when unknown. Do NOT hallucinate.

## Curves

1. Detect every visually distinct DATA curve/trace.
2. curve_id: curve_1, curve_2, … within the plot.
3. Explicit label → preserve exactly. Unlabeled → label=null; describe appearance; NEVER invent a sample name.
4. vertical_order: 1 = topmost for stacked spectra.
5. role: experimental | fitted | simulated | baseline | unknown.
6. Do not count reference markers / phase sticks as curves.

## Experimental series

When multiple curves vary one ordered experimental variable (temperature, time, composition,
pressure, voltage, concentration, etc.), populate plot.series[] with:
- series_type, variable_name, unit
- ordered_curve_ids (low → high of the variable when order is clear)
- explicit_values when written on the figure
- overall_trend description
- confidence / confidence_level
Also put per-curve condition fields under curve.observed when visible.

## Confidence & provenance

- high / medium / low with optional 0–1 float
- EvidenceField for sample names, phases, conditions (source: legend_text, axis_label, annotation_text, …)

## Hard prohibitions

- Do NOT digitize curves into dense XY data.
- Do NOT invent peak lists or sample names.
- Estimated numbers only when visually justified; semantic fields come first.
""".strip()
