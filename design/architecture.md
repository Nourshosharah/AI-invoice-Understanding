# System Architecture

This document provides detailed architecture documentation including data flow diagrams, component specifications, and integration points.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Data Flow Diagrams](#data-flow-diagrams)
3. [Component Specifications](#component-specifications)
4. [Data Models](#data-models)
5. [API Specifications](#api-specifications)
6. [Integration Points](#integration-points)

---

## High-Level Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INVOICE UNDERSTANDING SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        EXTRACTION LAYER                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │   PDF       │  │   Table     │  │   Line      │  │    LLM      │ │   │
│  │  │   Parser    │→ │  Extractor  │→ │   Item      │→ │   Fixer     │ │   │
│  │  │ (pdfplumber)│  │  (Rules)    │  │  Extractor  │  │ (Post-proc) │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        INDEXING LAYER                                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │   │
│  │  │    Chunk    │  │  Embedding  │  │   Vector    │                   │   │
│  │  │  Processor  │→ │  Generator  │→ │   Store     │                   │   │
│  │  │             │  │ (all-MiniLM-L6-v2)    │  │ (QDrant DB)  │                   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       RETRIEVAL LAYER                                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │   │
│  │  │   Query     │  │   Result    │  │   Answer    │                   │   │
│  │  │   Router    │→ │  Aggregator │→ │  Generator  │                   │   │
│  │  │             │  │             │  │ (Qwen2.5-14B-Instruct)    │                   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                          DATA STORES                                  │   │
│  │  ┌───────────────────┐  ┌─────────────────────────────────────────┐  │   │
│  │  │   QDrant DB        │  │   Extracted JSON                        │  │   │
│  │  │   (Vectors+Meta)  │  │   (data/extracted/*.json)              │  │   │
│  │  └───────────────────┘  └─────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **PDF Processing** | pdfplumber | Text and table extraction |
| **Vector Database** | QDrant DB | Semantic search and storage |
| **LLM** | Qwen2.5-14B-Instruct | Extraction fixing and answer generation |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | Semantic representation |
| **Data Format** | JSON | Interchange format |


---

## Data Flow Diagrams

### Invoice Ingestion Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INVOICE INGESTION FLOW                            │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌───────────────┐
    │  PDF File     │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │  Parse PDF    │
    │  (pdfplumber) │
    └───────┬───────┘
            │
            ├─────────────────────────────────────┐
            │                                     │
            ▼                                     ▼
    ┌───────────────┐                     ┌───────────────┐
    │ Extract Text  │                     │ Extract Tables│
    │ (Headers/Foot)│                     │ (Line Items)  │
    └───────┬───────┘                     └───────┬───────┘
            │                                     │
            │                                     ▼
            │                             ┌───────────────┐
            │                             │ Detect Columns│
            │                             │ (Header Info) │
            │                             └───────┬───────┘
            │                                     │
            │                                     ▼
            │                             ┌───────────────┐
            │                             │ Parse Rows    │
            │                             │ (Per Line)    │
            │                             └───────┬───────┘
            │                                     │
            │                                     ▼
            │                             ┌───────────────┐
            │                             │ Extract 7     │
            │                             │ Fields        │
            │                             │ (Code, Desc,  │
            │                             │  Date, Qty,   │
            │                             │  Price, Total)│
            │                             └───────┬───────┘
            │                                     │
            │                                     ▼
            │                             ┌───────────────┐
            │                             │ LLM Post-proc │
            │                             │ (Fix Errors)  │
            │                             └───────┬───────┘
            │                                     │
            └──────────────┬──────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  LineItem JSON Output  │
              │  (7 fields + metadata) │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │   Create Semantic      │
              │   Chunks (Per Item)    │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │ Generate Embeddings    │
              │ (all-MiniLM-L6-v2)     │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  Store in QDrantDB     │
              │  (Vector + Metadata)   │
              └────────────────────────┘
```

### Query Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            QUERY PROCESSING FLOW                             │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌───────────────┐
    │ User Query    │
    │ "Show items   │
    │  from page 3" │
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │ Classify      │
    │ Query Type    │
    └───────┬───────┘
            │
            ├───────────────┬───────────────┐
            │               │               │
            ▼               ▼               ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ Semantic      │ │ Metadata      │ │ Hybrid        │
    │ Query         │ │ Filter Query  │ │ Query         │
    │ "maintenance" │ │ "page = 3"    │ │ "support on   │
    │               │ │               │ │  page 3"      │
    └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
            │                 │                 │
            ▼                 ▼                 ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ Vector        │ │ QDrant DB      │ │ Vector +      │
    │ Similarity    │ │ Metadata      │ │ Metadata      │
    │ Search        │ │ Filter        │ │ Filter        │
    └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
            │                 │                 │
            └────────────┬────┴────────────┬────┘
                         │                 │
                         ▼                 ▼
              ┌─────────────────┐  ┌─────────────────┐
              │ Retrieve Top-K  │  │ Retrieve Filtered│
              │ Similar Items   │  │ Items           │
              └────────┬────────┘  └────────┬────────┘
                       │                    │
                       └─────────┬──────────┘
                                 │
                                 ▼
                       ┌─────────────────┐
                       │ Aggregate       │
                       │ Results         │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Build Context   │
                       │ (Retrieved Items│
                       │  + Formatting)  │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ LLM Generation  │
                       │ (Qwen2.5-14B-Instruct) │
                       │ + Grounding     │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Grounded Answer │
                       │ + Citations     │
                       └─────────────────┘
```

---

## Component Specifications

### 1. PDF Parser (invoice_parser.py)

**Purpose**: Load and extract text/tables from PDF invoices

**Input**: PDF file path

**Output**: Raw text and table structures

**Dependencies**: pdfplumber

**Error Handling**:
- Invalid PDF → Exception with message
- Encrypted PDF → Skip or prompt password
- Empty pages → Return empty list

---

### 2. Line Item Extractor (line_item_extractor.py)

**Purpose**: Parse table structures into structured line items

**Input**: Table data from PDF parser

**Output**: List of LineItem objects

**Extraction Strategy**:
1. Detect column headers (Item Code, Description, Delivery Date, etc.)
2. Map columns to field positions
3. Parse each row using column positions
4. Clean and normalize values

**Field Mapping**:
```
PDF Column          → LineItem Field
─────────────────────────────────────
Item Code           → item_code
Description         → description
Delivery Date       → delivery_date
Quantity            → quantity
Unit Price          → unit_price
Amount/Total        → total_amount
```

---

### 3. LLM Fixer (llm_fixer.py)

**Purpose**: Post-process extraction results with LLM

**Input**: Raw/extracted line items with potential errors

**Output**: Corrected line items

**Common Fixes**:
- OCR typos in numbers (e.g., "1.0O0" → "100.00")
- Missing delivery dates (infer from context)
- Malformed currency symbols
- Description cleanup (remove extra whitespace)

**LLM Prompt Template**:
```python
FIX_PROMPT = """
You are an invoice data expert. Fix extraction errors.

Raw extracted item:
- Item Code: {item_code}
- Description: {description}
- Delivery Date: {delivery_date}
- Quantity: {quantity}
- Unit Price: {unit_price}
- Total: {total_amount}

Page context:
{page_text}

Return JSON with corrected fields. Use null for unknown fields.
Follow invoicehome.com format:
- Dates as YYYY-MM-DD
- Numbers as numeric values (not strings)
- Clean descriptions (no extra whitespace)
"""
```

---
**Metadata Structure**:
```python
metadata = {
    "invoice_id": "INV-001",
    "page_number": 3,
    "line_number": 5,
    "item_code": "ABC123",
    "delivery_date": "2024-03-15",
    "quantity": 10,
    "unit_price": 99.99,
    "total_amount": 999.90
}
```

---

### 4. Vector Store (vector_store.py)

**Purpose**: Store and retrieve line items with ChromaDB

**Input**: Chunks with embeddings

**Output**: Query results


**Query Modes**:
1. **Semantic Only**: `query_text="support items", filters=None`
2. **Filter Only**: `query_text="", filters={"page_number": 3}`
3. **Hybrid**: `query_text="Food", filters={"invoice_id": "INV-001"}`

**Filter Syntax**:
```python
# Simple equality
filters = {"page_number": 5}

# Range query (ChromaDB extended syntax)
filters = {
    "delivery_date": {"$gte": "2024-01-01", "$lte": "2024-12-31"}
}

# AND conditions
filters = {
    "page_number": 3,
    "invoice_id": "RXC-101"
}
```

---

### 6. Query Router (query_router.py)

**Purpose**: Classify queries and route to appropriate retriever

**Input**: User query string

**Output**: Query type + routing decision


**Query Classification**:

| Type | Indicators | Example |
|------|-----------|---------|
| **Structural** | "page X", "pages Y-Z" | "Show page 3 items" |
| **Filtered** | Date ranges, specific values | "March 2024 invoices" |
| **Semantic** | Keywords, natural language | "maintenance support" |
| **Hybrid** | Combinations | "support on page 3" |

**Filter Extraction Examples**:
```
"page 3" → {"page_number": 3}
"pages 10 to 15" → {"page_number": {"$gte": 10, "$lte": 15}}
"March 2024" → {"delivery_date": {"$gte": "2024-03-01", "$lte": "2024-03-31"}}
"invoice RXC-101" → {"invoice_id": "RXC-101"}
```

---

### 7. Answer Generator (answer_generator.py)

**Purpose**: Generate grounded responses from retrieved items

**Input**: Retrieved line items + user query

**Output**: Grounded answer with citations



**Prompt Template**:
```python
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions about invoice line items.

CRITICAL RULES:
1. ONLY use information from the provided context
2. If the answer is not in the context, say "I don't have that information"
3. Do NOT make up or infer information not present in the context
4. Always cite your sources using the provided format
5. Be concise but complete in your responses

When citing sources, use this format:
- Include the invoice ID, page number, and line number for each claim
- Example: "Items from invoice INV-001, page 3, line 5"

Structure your response with:
1. Direct answer to the question
2. Formatted table or list of relevant items (when appropriate)
3. Summary of key findings (when relevant)
"""
answer_prompt = f"""Based on the following retrieved invoice line items, answer the question.

Question: {query}

Retrieved Line Items:
{context}

Please provide a helpful answer based only on the items above.
```

**Response Format**:
```python
{
    "answer": "Found 3 items on page 3 for invoice RXC-101...",
    "sources": [
        {"invoice_id": "RXC-101", "page": 3, "line": 5},
        {"invoice_id": "RXC-101", "page": 3, "line": 7},
        {"invoice_id": "RXC-101", "page": 3, "line": 12}
    ],
    "retrieved_items": [...]  # Original retrieved data
}
```




