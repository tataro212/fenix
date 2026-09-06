# Fenix

Fenix is a PDF translation and reconstruction pipeline. It extracts positioned
content with PyMuPDF, uses layout detection only for pages that need it,
translates text in bounded concurrent batches, and generates DOCX/PDF output.

## Recovery status

This branch restores source modules and tests that were removed from the default
branch when credential-related files were excluded. Credentials are not stored
in the repository. The historical configuration has been sanitized.

## Setup

Use Python 3.11 or newer in a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set `GEMINI_API_KEY` in `.env`. Do not add `.env` to Git.

The YOLO dependencies are large. If you only need text-heavy PDFs, they should
eventually be moved into an optional dependency group.

## Usage

```bash
python run_optimized_pipeline.py input.pdf --output output --target el --workers 5
```

The command emits a JSON summary and uses conventional exit codes:

- `0`: processing succeeded
- `1`: processing failed
- `2`: invalid command-line input

## Architecture

```text
PDF
 └─ extract positioned blocks once
     └─ classify page complexity
         ├─ text page: PyMuPDF fast path
         └─ mixed page: selective YOLO path
             └─ bounded translation batches
                 └─ ordered reconstruction
                     ├─ DOCX
                     └─ PDF
```

The canonical lightweight page schema lives in `models.py`. The richer
Digital Twin representation lives in `digital_twin_model.py`; imports must use
explicit aliases when both models are needed.

## Validation

```bash
python -m compileall -q .
python test_basic_functionality.py
python test_phase1_implementation.py
python test_intelligent_chunking.py
```

See `MAIN_WORKFLOW_ENHANCED_MODULE_MAP.md` for the historical architecture.

