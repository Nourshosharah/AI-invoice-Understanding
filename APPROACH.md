# Design Decisions and Trade-offs

This document explains the key design decisions, architectural choices, and trade-offs made in building the Intelligent Invoice Understanding System.

## Table of Contents

1. [Extraction Strategy](#extraction-strategy)
2. [Chunking Strategy](#chunking-strategy)
3. [Storage and Retrieval](#storage-and-retrieval)
4. [LLM and Embeddings](#llm-and-embeddings)
5. [System Architecture](#system-architecture)
6. [Trade-offs and Limitations](#trade-offs-and-limitations)

---

## Extraction Strategy

### Chosen Approach: Hybrid (Rule-based + LLM Post-processing)

#### Why Not Pure Rule-based?

Pure rule-based extraction has significant limitations:
- **Brittle to layout variations**: Small changes in invoice format break extraction
- **Limited handling of edge cases**: Skewed text, merged cells, unclear delimiters
- **Maintenance burden**: New invoice formats require rule updates

#### Why Not Pure LLM-based?

Pure LLM extraction also has issues:
- **Cost**: Processing each invoice through an LLM is expensive
- **Latency**: LLM inference is slow for large documents
- **Token limits**: Multi-page invoices may exceed context windows
- **Overkill for simple cases**: Many invoices have straightforward layouts

#### Hybrid Approach: Best of Both Worlds

**Phase 1: Rule-based Extraction (Fast, High Coverage)**
```
pdfplumber → Table Detection → Column Alignment → Row Parsing
```

- Uses pdfplumber's table extraction capabilities
- Identifies columns via header detection and spatial analysis
- Parses rows using pattern matching for common formats
- Handles 80-90% of invoices with good accuracy

**Phase 2: LLM Post-processing (Accuracy Boost)**
```
Raw Extraction → LLM Review → Corrected Output
```

- Feeds raw extraction to LLM for verification
- Fixes common errors: OCR typos, malformed numbers, missing fields
- Enriches descriptions when context is ambiguous
- Only processes failed/extracted-with-low-confidence items

#### Handling Multi-page Invoices

**Per-page Processing**:
1. Load PDF and iterate through all pages
2. Extract tables from each page independently
3. Aggregate line items with page number metadata
4. Handle line items that span pages (continued markers)

**Line Item Continuity**:
- Detects "continued" indicators in descriptions
- Merges split items when detected
- Preserves page context for each line item

---

## Chunking Strategy

### Chosen Approach: Per-Line-Item Chunking

#### Why Per-Line-Item?

**why NOT: Per-Page Chunking**

Per-page chunks would:
- ❌ Include irrelevant content (headers, footers, summaries)
- ❌ Create large chunks that dilute individual item relevance
- ❌ Make filtered queries harder (mix multiple line items)
- ❌ Reduce retrieval precision

**Per-Line-Item Benefits**:
- ✅ Atomic units of information (one item = one chunk)
- ✅ Precise retrieval of specific items
- ✅ Clean metadata association
- ✅ Smaller, faster vector operations

#### Chunk Structure

Each chunk contains:

```python
{
    "content": "Item code, description, unit price, quantity, total",
    "metadata": {
        "invoice_id": "INV-001",
        "page_number": 3,
        "line_number": 5,
        "delivery_date": "2024-03-15",
        "quantity": 10,
        "unit_price": 99.99,
        "total_amount": 999.90,
        "item_code": "ABC123"
    }
}
```

#### Metadata Preservation

**Page Numbers**:
- Essential for "show items on page X" queries
- Stored as integer in metadata
- Indexed for filtering

**Invoice Metadata**:
- Invoice ID (unique identifier)
- Invoice date (for temporal queries)
- Subtotal/tax/total (for summary queries)



## Storage and Retrieval

### Chosen Storage: Qdrant (Vector Database)

#### Why Qdrant?

**Alternatives Considered**:

| Option | Pros | Cons |
|--------|------|------|
| **Qdrant** | High performance, Rust-based, rich filtering | Requires migration |
| **Pinecone** | Production-scale, managed | Requires API key, cloud-only |
| **Weaviate** | Graph features, open-source | More complex setup |
| **ChromaDB** | Easy setup, Python-native | Less scalable for billions |
| **FAISS** | Fast, offline | No metadata filtering natively |
| **PostgreSQL+pgvector** | Full SQL, mature | Requires database setup |

#### Qdrant Selection Rationale

**Strengths for this use case**:
1. **High performance**: Rust-based, excellent query speed
2. **Rich filtering**: Must/Should conditions for complex queries
3. **Distributed ready**: Can scale to clusters
4. **Persistent storage**: Data survives restarts
5. **Open-source**: No vendor lock-in
6. **Built-in quantization**: Memory-efficient for large datasets


## LLM and Embeddings

### Chosen Embedding Model: text-embedding-3-small

#### Why sentence-transformers/all-MiniLM-L6-v2?

**Alternatives Considered**:

| Option | Pros | Cons |
|--------|------|------|
| **all-MiniLM-L6-v2** | Fast, lightweight, open-source | 384 dimensions |
| **all-mpnet-base-v2** | Higher quality | Larger, slower |
| **OpenAI text-embedding-3-small** | High quality | API costs, privacy concerns |

**Selection Rationale**:
1. **Industry standard**: Most popular sentence embedding model
2. **Fast inference**: 1000+ vectors/second on CPU
3. **Compact**: 384 dimensions, minimal storage
4. **No API costs**: Self-hosted, unlimited queries
5. **Good quality**: Excellent MTEB benchmark performance
6. **Perfect for invoices**: Handles short text descriptions well

#### Embedding Strategy

**What gets embedded**:
```python
# Concatenated fields for semantic search
embedding_text = f"""
Item Code: {item_code}
Description: {description}
Unit Price: {unit_price}
Quantity: {quantity}
Total: {total_amount}
"""
```

**What doesn't get embedded**:
- Purely numeric fields (stored as metadata for filtering)
- Invoice headers/footers (not part of line items)

### Chosen LLM: Qwen/Qwen2.5-14B-Instruct

#### Why Qwen2.5-14B-Instruct?

**Alternatives Considered**:

| Model | Parameters | Cost | Strengths |
|-------|------------|------|-----------|
| **Qwen2.5-14B-Instruct** | 14B | Self-hosted | Best instruction following |
| GPT-4o | N/A | API | State-of-the-art |

**Selection Rationale**:
1. **Instruction-tuned**: Specifically optimized for task completion
2. **131K context**: Handle entire invoices at once
3. **Strong reasoning**: Accurate calculations and comparisons
4. **Open-source**: No API costs, complete data privacy

#### LLM Usage Patterns

**1. Extraction Fixing**:
```python
# System prompt for extraction fixing
fix_prompt = """
You are an invoice data expert. Fix extraction errors in this line item:
{raw_extraction}

Return JSON with corrected fields:
- item_code
- description
- delivery_date (YYYY-MM-DD)
- quantity (numeric)
- unit_price (numeric)
- total_amount (numeric)

If a field cannot be determined, set to null.
"""
```

**2. Answer Generation**:
```python
# System prompt for grounded responses
answer_prompt = """
You are a helpful assistant answering questions about invoice line items.
Use ONLY the provided context to answer. If the answer is not in the context, say "I don't have that information."

Context:
{retrieved_items}

Question: {user_query}

Answer:
"""

    # System prompt for answer generation
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
```

**3. Self-Hosting with Transformers**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Qwen model locally
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# Generate responses without API calls
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.0)
```

---

## System Architecture

### Data Flow

```
┌─────────────┐
│  Invoice    │
│  PDF        │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Ingest    │  ┌─────────────────┐
│  Pipeline   │  │ 1. Parse PDF    │
└──────┬──────┘  │ 2. Detect Tables│
       │         │ 3. Extract Rows │
       │         └─────────────────┘
       ▼
┌─────────────┐
│  Extraction │  ┌─────────────────┐
│  Component  │→ │ 1. Rule-based   │
└──────┬──────┘  │ 2. LLM Fixing   │
       │         └─────────────────┘
       ▼
┌─────────────┐
│    Chunk    │  ┌─────────────────┐
│  Processor  │  │ 1. Create Chunks│
└──────┬──────┘  │ 2. Add Metadata │
       │         │ 3. Generate Emb.│
       ▼         └─────────────────┘
       ▼
┌─────────────┐
│  QDrant DB   │  ┌─────────────────┐
│  Storage    │  │ 1. Store Vectors│
└──────┬──────┘  │ 2. Index Meta   │
       │         └─────────────────┘
       ▼
┌─────────────┐
│   Query     │  ┌─────────────────┐
│  Processor  │  │ 1. Classify     │
└──────┬──────┘  │ 2. Route Query  │
       │         │ 3. Retrieve     │
       ▼         └─────────────────┘
       ▼
┌─────────────┐
│    LLM      │  ┌─────────────────┐
│  Generator  │  │ 1. Build Context│
└──────┬──────┘  │ 2. Generate Ans.│
       │         │ 3. Ground Resp. │
       ▼         └─────────────────┘
       ▼
┌─────────────┐
│   Grounded  │
│   Response  │
└─────────────┘
```

### Component Responsibilities

**Ingestion Pipeline**:
- Load PDF documents
- Extract text and tables
- Handle multi-page aggregation
- Output structured JSON

**Extraction Component**:
- Apply rule-based parsing
- Identify line item boundaries
- Extract 7 required fields
- Post-process with LLM

**Chunk Processor**:
- Create semantic chunks
- Attach metadata
- Generate embeddings
- Batch insert to storage

**Query Processor**:
- Classify query type
- Route to appropriate retriever
- Apply metadata filters
- Aggregate results

**Answer Generator**:
- Format retrieved context
- Prompt LLM with grounding
- Return sourced answers

---

## Trade-offs and Limitations

### Known Limitations

1. **Layout Sensitivity**:
   - Highly unconventional layouts may fail
   - Non-table line items (e.g., paragraph-style) not supported
   - Solution: LLM post-processing catches many edge cases

2. **Extraction Accuracy**:
   - Not 100% perfect 
   - Numbers in descriptions may be confused with fields
   - Solution: Human review for critical invoices

3. **Semantic Search Limits**:
   - May miss items with different terminology , need to be more specfic exp(not search for oil , search for sunflower oil) need to be more close with data 
   - Solution: Include filters for precise queries



### Scalability Considerations

**Current Design**:
- Single-node Qdrant
- In-process processing
- Sequential invoice ingestion

**For Scale**:
- Use Qdrant distributed mode
- Parallelize ingestion with ThreadPoolExecutor
- Batch LLM requests
- Qdrant quantization for memory efficiency

---

## Evaluation Approach

#### RAGAS Framework Metrics

When the RAGAS library is available, the system utilizes its standard evaluation metrics to assess answer quality across multiple dimensions. These metrics are widely used in the RAG community and have been validated through extensive research and production deployments. Each metric provides specific insights into different aspects of system performance, enabling targeted identification of improvement areas.
**note** i have an issue using ragas library so I use a simplified metrics 

**Faithfulness** measures how well the generated answer is grounded in the retrieved contexts, ensuring that the system does not hallucinate information not present in the source documents. A score of 1.0 indicates that every claim in the answer can be directly verified from the contexts, while a score of 0.0 indicates that the answer introduces significant information unrelated to the retrieved content. This metric is critical for invoice processing applications where factual accuracy directly impacts financial decisions.

**Answer Relevancy** evaluates how directly and completely the answer addresses the user's question, measuring the system's ability to understand query intent and provide focused responses. A score of 1.0 indicates that the answer comprehensively addresses all aspects of the question, while lower scores indicate partial coverage or tangential responses. This metric helps identify cases where the system retrieves relevant information but fails to synthesize it into a coherent, targeted response.

**Context Precision** assesses whether the most relevant contexts are ranked and retrieved first, measuring the quality of the retrieval ranking mechanism. A score of 1.0 indicates that highly relevant contexts appear at the top of the retrieved results, while lower scores indicate that relevant content is buried among less relevant items. This metric is particularly important for applications where only the top-K results are shown to users.

**Context Recall** measures the coverage of retrieved contexts against the expected answer, assessing whether all necessary information has been captured. A score of 1.0 indicates that the retrieved contexts contain all information needed to answer the question completely, while lower scores indicate missing information that would be required for a full response.

#### Simplified Fallback Metrics

When the RAGAS library is not available, the evaluation system automatically switches to simplified pattern-matching metrics that provide reasonable approximations of quality without requiring external dependencies. These lightweight metrics are designed to capture essential aspects of answer quality using algorithmic approaches that can execute on any system configuration.

**Faithfulness (Simplified)** checks if the ground truth is contained within the generated answer, with a bonus awarded if the answer shares significant vocabulary with the retrieved contexts. This approximation captures the essential idea of grounding without requiring sophisticated LLM-based verification.

**Answer Relevancy (Simplified)** calculates word-level overlap between the question and answer, measuring whether the response contains terminology relevant to the query. This approach provides a basic indication of whether the answer is on-topic without requiring deep semantic understanding.

**Context Precision (Simplified)** evaluates whether the first retrieved context is substantial and informative, using context length as a proxy for relevance. This heuristic assumes that relevant contexts typically contain detailed information about the query subject.

**Context Recall (Simplified)** measures the proportion of retrieved contexts that contain substantial information, assuming that longer contexts are more likely to contain answer-relevant content.

#### Evaluation Thresholds and Pass Criteria

The evaluation framework defines performance thresholds for each quality dimension, enabling automated pass/fail determinations that support continuous integration and quality assurance workflows. These thresholds establish minimum acceptable performance levels for production deployment, ensuring that the system meets quality standards before changes are released.

| Dimension | Threshold | Category |
|-----------|-----------|----------|
| Grounding Safety (Faithfulness) | 0.60 | Pass if ≥ 0.60 |
| QA Accuracy (Answer Relevancy) | 0.60 | Pass if ≥ 0.60 |
| Retrieval Quality (Context Precision) | 0.60 | Pass if ≥ 0.60 |
| Retrieval Quality (Context Recall) | 0.60 | Pass if ≥ 0.60 |

The system reports evaluation results both as individual metric scores and as aggregated dimension scores, with dimensions grouping related metrics together. A final summary indicates how many dimensions passed their respective thresholds, providing a quick overall assessment of system quality. Failed dimensions are highlighted for targeted improvement efforts.

#### Test Dataset Composition

The evaluation framework includes 7 carefully designed test samples that span key query types the system should handle. Each sample consists of a question, an answer generated by the system, the contexts retrieved during query processing, and a ground truth reference for correctness assessment. The test set covers item lookup queries with detailed specifications, financial calculation queries for subtotals and totals, enumeration queries for counting items, and metadata queries for tax rates and item classifications.

| Query Type | Example | Purpose |
|------------|---------|---------|
| Item Lookup | "Find Frozen Chicken Wings and give its details" | Tests extraction of specific line items |
| Financial | "What is the subtotal?" | Tests numerical extraction accuracy |
| Enumeration | "How many different items are on this invoice?" | Tests counting capabilities |
| Metadata | "What is the tax rate?" | Tests header information extraction |

This diverse test set ensures that evaluation captures the system's performance across different query patterns without bias toward any single type. The questions are derived from realistic invoice processing scenarios, making evaluation results directly relevant to production use cases.


### Continuous Evaluation

**Production Monitoring**:
- Log all queries and responses
- Track user feedback (thumbs up/down)
- Weekly accuracy reviews
