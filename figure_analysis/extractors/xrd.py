"""XRD-specific extraction checklist for the vision prompt."""

from __future__ import annotations


def build_instructions() -> str:
    return """
## XRD-specific extraction

When a plot has plot_type XRD (powder diffraction / 2θ), prioritize:

### HIGH PRIORITY — observed semantic metadata
- sample / trace names from legend (exact text; EvidenceField when useful)
- phase names + marker symbols/colors; mark is_reference when appropriate
- Miller indices / hkl — copy literal labels exactly (e.g. "(101)")
- experimental conditions (temperature, composition, …) if written
- x_axis / y_axis labels and units if readable (2 Theta / degree; Intensity / a.u.)
- axis min/max only when tick labels clearly show them
- explicitly printed peak positions → labeled_peak_positions (NOT estimated)
- explicit FWHM only if printed → explicit_fwhm_values
- inset_present; arrows/boxes/shading → annotations[]
- reference pattern markers (these are NOT curves)
- crystal structures shown alongside
- experimental series (temperature/composition stacks, etc.)

### MEDIUM PRIORITY — interpretation (only if strongly supported)
- peak_splitting_visible, shoulders_visible, peaks_appear_broadened
- secondary_phase_present
- qualitative_relative_intensities
- inset_purpose
- phase_transition_suggested only if clearly indicated

### LOW PRIORITY — estimated (optional; prefer null if uncertain)
- approximate major peak two_theta (visual_axis_estimation only)
- approximate relative intensities / widths with width_is_approximate=true
- NEVER present visual width estimates as fitted FWHM
- inset ranges only when useful and not printed

### critical distinctions
- Explicitly labeled 30.6° → labeled_peak_positions (observed)
- Visually guessed ≈30.6° → estimated.major_peaks OR null (prefer null if unsure)
- Explicitly labeled phases → observed.phases
- Inferred unlabeled phases → interpretation only
- Phase markers / reference sticks ≠ curves
- Inset ≠ independent panel

If the plot is NOT XRD, leave XRD-specific lists empty / null.
""".strip()
