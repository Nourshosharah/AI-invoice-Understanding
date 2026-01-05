# Intelligent Invoice Understanding System

A comprehensive document AI system for invoice ingestion, line item extraction, indexing, and natural language querying with LLM-powered answer generation.

Supports both PDF documents and image files with OCR technology.

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Overview](#system-overview)
3. [Supported File Formats](#supported-file-formats)
4. [Installation](#installation)
5. [OCR Setup](#ocr-setup)
6. [Usage](#usage)
7. [Project Structure](#project-structure)
8. [Features](#features)

## Quick Start

```bash
# Clone the repository
cd invoice-understanding-system

# Install dependencies (includes OCR libraries)
pip install -r requirements.txt

# Ingest a PDF invoice
python -m src.main ingest path/to/invoice.pdf

# Ingest an image invoice (requires Tesseract OCR)
python -m src.main ingest path/to/invoice.png

# Query the system
python -m src.main query "Show all items from page 1"
```

## System Overview

This system processes invoices from PDF documents and images, extracts line items and metadata, indexes them in a vector database, and supports natural language queries with LLM-generated grounded responses.

### Key Capabilities

- **Multi-Format Ingestion**: Process PDF documents and image files
- **OCR Support**: Extract text from images using Tesseract OCR
- **Line Item Extraction**: Parse structured table data with 7 fields
- **Semantic Indexing**: Store items with metadata for filtering and search
- **Natural Language Querying**: Ask questions in plain English
- **Grounded Responses**: LLM answers strictly based on extracted data
- **OCR Fallback**: Automatic OCR fallback when PDF text extraction is poor

## Supported File Formats

### PDF Documents
- `.pdf` - Standard PDF format

### Image Files
- `.png` - Portable Network Graphics
- `.jpg` / `.jpeg` - JPEG images
- `.bmp` - Bitmap images
- `.tiff` / `.tif` - Tagged Image File Format

## Installation

### Prerequisites

- Python 3.10+
- pip or conda
- **GPU with at least 8GB VRAM** (recommended for Qwen model)
- **Tesseract OCR** (required for image processing - see OCR Setup below)

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16GB | 32GB |
| GPU VRAM | 8GB (quantized) | 16GB (FP16) |  
| Storage | 10GB | 50GB |
**Note** : if you want to deploy it locally , and hosting the models locally , you must have GPu Requirments , you still can run the codes that's have Api call that here you should have internet access 
### Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
pdfplumber>=0.10.0
qdrant-client>=1.6.0
sentence-transformers>=2.2.2
transformers>=4.40.0
accelerate>=0.25.0
torch>=2.0.0
pandas>=2.0.0
pydantic>=2.0.0
python-dateutil>=2.8.0
tqdm>=4.65.0
Pillow>=9.0.0          # Image processing
pytesseract>=0.3.10    # OCR wrapper
python-dateutil>=2.8.0
python-dotenv>=1.0.0
```

## OCR Setup

### Installing Tesseract OCR

Image processing requires Tesseract OCR to be installed on your system.

#### Windows
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (e.g., `tesseract-ocr-w64-setup-v5.3.0.exe`)
3. Default installation path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
4. Add to PATH or specify path with `--tesseract-path`

#### macOS
```bash
brew install tesseract
brew install tesseract-lang  # For additional languages
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-eng  # English language data
```

### Verify Installation

```bash
# Check OCR status
python -m src.main ocr-status

# Should output:
# ✓ Tesseract OCR is installed
# Version: x.x.x
# Path: /usr/bin/tesseract
# Languages: eng, ...
```

### Custom Tesseract Path

If Tesseract is not in your PATH, specify it explicitly:

```bash
python -m src.main ingest invoice.png --tesseract-path "/path/to/tesseract"
```

## Usage

### 1. Ingest a Document (PDF or Image)

```bash
# Ingest a PDF
python -m src.main ingest path/to/invoice.pdf

# Ingest an image (requires Tesseract OCR)
python -m src.main ingest path/to/invoice.png
python -m src.main ingest path/to/invoice.jpg

# Batch process all documents in a directory
python -m src.main ingest ./documents/ --batch
```

Options:
```
--output-dir       Directory to save extracted data (default: data/extracted)
--batch            Process all supported files in directory
--tesseract-path   Path to Tesseract executable (for images)
--no-ocr-fallback  Disable OCR fallback for PDFs
```

### 2. List Indexed Invoices

```bash
python -m src.main list
```

Shows all invoices in the database with basic metadata.

### 3. Query the System

```bash
python -m src.main query "Show all items from page 3 for invoice RXC-101"
```

Query types supported:
- **Structural**: "Show items on page X"
- **Filtered**: "Items from March 2024"
- **Semantic**: "Items related to maintenance"
- **Summarization**: "Summarize items between pages 10-15"

### 4. Interactive Mode

```bash
python -m src.main interactive
```

Start an interactive session for continuous querying.

#### Interactive Mode

```bash
python -m src.main interactive
```

**Example Session**:
```
Invoice QA System v1.0
Type 'exit' to quit, 'help' for commands.

> What items cost more than $500?
Found 4 items over $500:
- Server License (INV-001): $1,000
- Database Cluster (INV-001): $2,500
- Cloud Setup (INV-002): $750
- Annual Support (INV-002): $1,200

> Show me page 3 of invoice RXC-101
[Shows items from page 3]

> exit
Goodbye!
```


### 5. Check OCR Status

```bash
python -m src.main ocr-status
```

Verify Tesseract OCR installation and configuration.

## Project Structure

```
invoice-understanding-system/
├── README.md                    # This file
├── APPROACH.md                  # Design decisions and trade-offs
├── design/
│   └── architecture.md          # System architecture diagrams
├── src/
│   ├── main.py                  # CLI entry point
│   ├── __init__.py
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── invoice_parser.py    # PDF & Image parsing with OCR
│   │   ├── line_item_extractor.py  # Line item extraction logic
│   │   └── llm_fixer.py  
    ├── evaluation/
    │    ├── ragas_evaluation.py  #Rag metric using ragas library or simple evaluation metrics 
    │    
│   ├── indexing/
│   │   ├── __init__.py
│   │   └── vector_store.py      # ChromaDB integration
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── query_router.py      # Query classification and routing
│   │   └── answer_generator.py  # LLM response generation
│   └── utils/
│       ├── __init__.py
│       └── models.py            # Data models and schemas
├── tests/
│   ├── test.png
│   ├── test1.png #screenshot for working system 
├── data/
│   ├── extracted/               # JSON output from extraction
│   └── sample_invoices/         # Sample invoices for testing
└── requirements.txt
```

## Features

### 1. Multi-Format Document Processing

- **PDF Processing**: Native text extraction with pdfplumber
- **Image Processing**: OCR-based text extraction with Tesseract
- **Automatic Format Detection**: Identifies file type from extension
- **Unified API**: Same ingestion workflow for all formats

### 2. OCR Fallback Mechanism

When PDF text extraction yields poor quality results:

1. **Quality Assessment**: Evaluates extracted text quality
2. **Automatic Trigger**: Uses OCR when quality score is low
3. **Page-Level Fallback**: Applies OCR to individual low-quality pages
4. **Full Document OCR**: Complete OCR fallback for problematic PDFs

Quality indicators:
- Low word count
- Missing invoice field patterns
- Fragmented text lines
- Missing expected invoice content

### 3. Hybrid Extraction Strategy

Combines rule-based extraction with LLM post-processing:

1. **Rule-based extraction**:
   - Table detection via column alignment
   - Header detection for column identification
   - Row parsing with pattern matching

2. **OCR for images**:
   - Tesseract OCR text recognition
   - Table structure detection from OCR text
   - Pattern matching for numeric fields

3. **LLM fixing** for ambiguous cases:
   - Malformed numbers (e.g., "1.0O0" → "100")
   - Missing fields (inference from context)
   - Description cleanup

### 4. Per-Line-Item Chunking

Each line item is stored as an independent chunk with:

- **Content**: Item code, description, unit price, quantity, total
- **Metadata**: Page number, invoice ID, dates, file type
- **Embedding**: Semantic representation for similarity search

### 5. Multi-Modal Retrieval

Supports both:

- **Vector similarity** for semantic queries
- **Metadata filtering** for structural queries
- **Hybrid approach** combining both

### 6. Grounded LLM Responses

Answers are generated by:
1. Retrieving relevant line items
2. Formatting as structured context
3. Prompting LLM with strict grounding instructions
4. Returning only information from retrieved data

## Configuration

Create a `.env` file for model configuration (no API keys required):

```env
# Model Configuration
QWEN_MODEL_PATH=Qwen/Qwen2.5-14B-Instruct
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Device Configuration
DEVICE=auto  # or "cuda" / "cpu"

# Paths
QDRANT_PERSIST_DIR=data/qdrant_db
EXTRACTED_DIR=data/extracted

# Generation Settings
MAX_NEW_TOKENS=1024
TEMPERATURE=0.0
```

## Troubleshooting

### OCR Not Available

If you see "Tesseract OCR not available":
1. Install Tesseract (see OCR Setup above)
2. Verify with `python -m src.main ocr-status`
3. Specify path with `--tesseract-path` if not in PATH

### Poor OCR Quality

For better image OCR results:
- Use high-resolution images (300 DPI or higher)
- Ensure good contrast between text and background
- Avoid skewed or rotated images
- Use clear, legible fonts

### PDF Text Extraction Issues

If PDFs have poor text extraction:
- System will automatically use OCR fallback
- Enable explicitly with `--no-ocr-fallback` to disable if needed
- Consider converting problematic PDFs to images first

