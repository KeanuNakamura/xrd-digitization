# XRD Digitization

Digitize XRD figure images into calibrated CSV curves and preview PNGs, using axis OCR plus [PlotDigitizer](https://github.com/dilawar/PlotDigitizer).

## Setup

```bash
git clone --recurse-submodules git@github.com:KeanuNakamura/xrd-digitization.git
cd xrd-digitization
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you already cloned without submodules: `git submodule update --init`.

Install [Tesseract](https://github.com/tesseract-ocr/tesseract) for axis tick OCR (`brew install tesseract` on macOS).

## Usage

Digitize a folder of figure PNGs:

```bash
python plotdigitizer_pipeline.py examples/ --png-dir --output-dir output/
```

Alternate (deterministic package pipeline):

```bash
python -m xrd_digitization examples/figure_3.png --skip-classification
```

## Output

Each figure writes a CSV of `(x, y)` points plus a `_digitized.png` overlay. Multi-curve stacked plots are split into horizontal bands with one CSV/preview per band.
