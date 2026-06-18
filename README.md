## Setup

### 1. Start GROBID with Docker

Run the GROBID server in a Docker container:

```bash
docker run --rm \
  --init \
  --ulimit core=0 \
  -p 8070:8070 \
  grobid/grobid:0.9.0-crf
```

### 2. Create a Virtual Environment

Create the Python virtual environment:

```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## PDF Parser Usage

Run the parser using the following format:

```bash
python pdf_parser.py <pdf_path> --output <output_path>
include '--xrd-figures-only' for xrd figures/analysis only
include '--quiet' for concise output
```

### Example

```bash
Single File: 
python pdf_parser.py pdf_files/sample_pdfs/tio2_powder.pdf \
  --output grobid_output/sample_pdfs/tio2_powder --quiet --xrd-figures-only

Directory:
python pdf_parser.py pdf_files/scraped_pdfs --output grobid_output/scraped_pdfs --quiet --xrd-figures-only
```

## Scraping XRD Research Articles with OpenAlex

```bash
python xrd_article_scraper.py \
    --output-dir pdf_files/scraped_pdfs \
    --count 10
```

### OpenAlex API Key

An OpenAlex API key can be provided through an environment variable:

```bash
export OPENALEX_API_KEY="your_openalex_api_key"
```

### Output Structure

After a successful run, the output directory will resemble:

```text
pdf_files/
└── scraped_pdfs/
    ├── manifest.jsonl
    └── pdfs/
        ├── article_001.pdf
        ├── article_002.pdf
        └── ...
```

Only articles with an accessible open-access PDF can be downloaded. Some OpenAlex records may contain useful metadata but may not provide a downloadable PDF.




