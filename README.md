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

GROBID will be available at:

```text
http://localhost:8070
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

## Usage

Run the parser using the following format:

```bash
python pdf_parser.py <pdf_path> --output <output_path>
```

### Example

```bash
python pdf_parser.py pdf_files/tio2_powder.pdf \
  --output grobid_output/tio2_powder
```

The parser will process the PDF using the local GROBID server and save the generated files to the specified output directory.

