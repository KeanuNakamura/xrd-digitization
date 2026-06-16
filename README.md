Installation and Usage

This project uses GROBID to extract structured information from scientific PDF documents.

1. Start GROBID

Run the GROBID server with Docker:

docker run --rm \
  --init \
  --ulimit core=0 \
  -p 8070:8070 \
  grobid/grobid:0.9.0-crf

GROBID will be available at:

http://localhost:8070

Keep the Docker container running while using the PDF parser.

2. Set Up the Python Environment

Create a virtual environment:

python3 -m venv .venv

Activate the virtual environment:

macOS or Linux
source .venv/bin/activate
Windows PowerShell
.venv\Scripts\Activate.ps1

Upgrade pip and install the required packages:

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
3. Parse a PDF

Place the PDF you want to process inside the pdf_files/ directory.

Use the following command format:

python pdf_parser.py <pdf-path> --output <output-path>

For example:

python pdf_parser.py \
  pdf_files/tio2_powder.pdf \
  --output grobid_output/tio2_powder

The parser will send the PDF to the local GROBID server and save the processed files to the specified output path.

Example Project Structure
project/
├── pdf_files/
│   └── tio2_powder.pdf
├── grobid_output/
├── pdf_parser.py
├── requirements.txt
├── .gitignore
└── README.md
Complete Example

Open one terminal and start the GROBID Docker container:

docker run --rm \
  --init \
  --ulimit core=0 \
  -p 8070:8070 \
  grobid/grobid:0.9.0-crf

Open a second terminal, activate the Python environment, and run the parser:

source .venv/bin/activate

python pdf_parser.py \
  pdf_files/tio2_powder.pdf \
  --output grobid_output/tio2_powder

To stop GROBID, return to the terminal running Docker and press:

Ctrl+C
