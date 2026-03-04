# 04. RAG: Retrieval-Augmented Generation

> Comprehensive interview preparation guide for Backend AI Engineers.
> Covers theory, architecture, code examples, and 25+ interview Q&A.

---

## Table of Contents

1. [RAG Overview](#1-rag-overview)
2. [Document Chunking Strategies](#2-document-chunking-strategies)
3. [Embedding Models](#3-embedding-models)
4. [Vector Search & Reranking](#4-vector-search--reranking)
5. [Vector Databases](#5-vector-databases)
6. [Document Processing](#6-document-processing)
7. [Advanced RAG Techniques](#7-advanced-rag-techniques)
8. [Evaluation](#8-evaluation)
9. [Production Best Practices (2024-2025)](#9-production-best-practices-2024-2025)
10. [Q&A Section](#10-qa-section)

---

## 1. RAG Overview

### What is RAG and Why It Matters

**Retrieval-Augmented Generation (RAG)** is an architecture pattern that enhances Large Language Models (LLMs) by grounding their responses in external, retrieved knowledge. Instead of relying solely on the model's parametric memory (training data), RAG dynamically retrieves relevant documents at inference time and feeds them as context to the LLM.

**Why RAG matters:**
- **Reduces hallucinations** -- the model answers based on retrieved facts, not guesses
- **Keeps knowledge current** -- no retraining needed; just update the document store
- **Domain-specific answers** -- inject proprietary or niche data the model was never trained on
- **Auditability** -- you can trace every answer back to source documents
- **Cost-effective** -- far cheaper than fine-tuning for most knowledge-grounding use cases

### RAG Pipeline (Visual)

**Indexing Phase (offline):**
```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Document │ -> │  Chunk   │ -> │  Embed   │ -> │  Store   │
│  Ingest  │    │  Split   │    │  Vectors │    │ Vector DB│
└──────────┘    └──────────┘    └──────────┘    └──────────┘
  PDFs, HTML,     Split into      Convert to      Qdrant,
  DOCX, TXT       passages       dense vectors    Pinecone,
                  (256-1024       (768-3072 dim)   pgvector
                   tokens)
```

**Query Phase (online):**
```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Query   │ -> │  Embed   │ -> │ Retrieve │ -> │ Generate │
│  Input   │    │  Query   │    │  Top-K   │    │  Answer  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
  "How does       Same embed      Find K most     LLM produces
   auth work?"    model as        similar chunks  answer grounded
                  indexing                        in retrieved docs
```

### RAG vs Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Knowledge update** | Instant (update docs) | Requires retraining |
| **Cost** | Low (API + vector DB) | High (GPU hours) |
| **Hallucination control** | Strong (grounded in docs) | Moderate |
| **Domain adaptation** | Good for factual Q&A | Better for style/format |
| **Latency** | Higher (retrieval step) | Lower (single inference) |
| **Data requirement** | Any docs, no labels needed | Labeled training pairs |
| **Traceability** | Full (source attribution) | None |
| **Best for** | Knowledge-heavy, evolving data | Tone, format, specialized reasoning |
| **Combination** | Often combined: fine-tune + RAG for best results |

**When to use RAG:**
- Frequently changing knowledge base
- Need source attribution / citations
- Domain-specific Q&A over documents
- Compliance requires traceability

**When to use Fine-Tuning:**
- Need specific output format or tone
- Teaching the model a new skill (e.g., code generation in a custom DSL)
- Reducing prompt size (internalize instructions)

**When to use both:**
- Fine-tune the model to follow RAG instructions better
- Fine-tune embeddings for your domain + RAG pipeline

### Naive RAG vs Advanced RAG vs Modular RAG

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RAG Evolution                                │
├──────────────┬──────────────────┬────────────────────────────────────┤
│  Naive RAG   │  Advanced RAG    │  Modular RAG                      │
├──────────────┼──────────────────┼────────────────────────────────────┤
│ Basic index  │ + Query rewrite  │ + Pluggable components            │
│ Basic search │ + Reranking      │ + Routing (choose retriever)      │
│ Basic prompt │ + Hybrid search  │ + Agentic decisions               │
│              │ + Better chunks  │ + Self-reflection                 │
│              │ + Metadata filter│ + Multi-step retrieval            │
│              │                  │ + Adaptive strategies             │
├──────────────┼──────────────────┼────────────────────────────────────┤
│ Simple but   │ Significant      │ State-of-the-art;                 │
│ limited      │ quality gains    │ production-grade systems          │
└──────────────┴──────────────────┴────────────────────────────────────┘
```

**Naive RAG** -- the basic "retrieve then generate" pipeline. Simple to build but suffers from:
- Retrieval of irrelevant chunks
- Lost context at chunk boundaries
- No query understanding

**Advanced RAG** -- adds pre-retrieval and post-retrieval optimizations:
- Query transformation (rewriting, expansion)
- Reranking retrieved results
- Hybrid search (BM25 + dense)
- Contextual compression

**Modular RAG** -- treats RAG as a composable system:
- Router decides which retriever to use
- Agent can decide whether to retrieve at all
- Self-reflection: check if retrieval was helpful
- Multi-hop: iterative retrieval for complex questions

---

## 2. Document Chunking Strategies

Chunking is arguably the **most impactful** component of a RAG pipeline. Bad chunking = bad retrieval = bad answers.

### Why Chunking Matters

```
Problem: LLMs have limited context windows, and embedding models
         work best on focused, coherent text passages.

Goal:    Split documents into chunks that are:
         - Semantically coherent (one topic per chunk)
         - Right-sized (not too small, not too large)
         - Overlapping enough to preserve boundary context
```

### Strategy 1: Fixed-Size Chunking (with Overlap)

The simplest approach. Split every N characters/tokens with M overlap.

```python
from langchain.text_splitter import CharacterTextSplitter

text = "The cat sat on the mat. The dog ran in the park. Birds flew in the sky."

# Fixed-size chunking
splitter = CharacterTextSplitter(
    separator="",           # split anywhere
    chunk_size=30,          # 30 characters per chunk
    chunk_overlap=10,       # 10-char overlap
    length_function=len,
)

chunks = splitter.split_text(text)
# Chunk 1: "The cat sat on the mat. The"
# Chunk 2: "mat. The dog ran in the park."
# Chunk 3: "the park. Birds flew in the"
```

**Pros:** Simple, predictable chunk sizes, fast.
**Cons:** Breaks mid-sentence, ignores document structure, topics split arbitrarily.

### Strategy 2: Recursive Character Splitting

LangChain's default and most popular splitter. Tries a hierarchy of separators.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=[
        "\n\n",   # 1. Try paragraph breaks first
        "\n",     # 2. Then line breaks
        ". ",     # 3. Then sentence boundaries
        " ",      # 4. Then word boundaries
        "",       # 5. Last resort: character-level
    ],
)

chunks = splitter.split_text(long_document)
```

**Why it works well:** It respects natural text boundaries. It tries `\n\n` first (paragraph), and only falls back to smaller separators if chunks are still too large.

### Strategy 3: Semantic Chunking (Embedding-Based)

Split at points where the semantic meaning changes. Uses embeddings to detect topic shifts.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Semantic chunker detects topic boundaries via embedding similarity
chunker = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount=95,          # break at 95th percentile dissimilarity
)

chunks = chunker.split_text(document_text)
```

**How it works internally:**
```
Sentence 1  ──embed──>  [0.2, 0.8, ...]  ─┐
                                           ├─ cosine_sim = 0.92 (similar)     -> same chunk
Sentence 2  ──embed──>  [0.3, 0.7, ...]  ─┘
                                           ├─ cosine_sim = 0.31 (different)   -> SPLIT HERE
Sentence 3  ──embed──>  [0.9, 0.1, ...]  ─┘
                                           ├─ cosine_sim = 0.88 (similar)     -> same chunk
Sentence 4  ──embed──>  [0.8, 0.2, ...]  ─┘
```

**Pros:** Chunks are topically coherent.
**Cons:** Slower (requires embedding every sentence), variable chunk sizes.

### Strategy 4: Document-Structure-Aware Chunking

Leverage the document's own structure (headers, sections, paragraphs).

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

# Split by markdown headers
headers_to_split_on = [
    ("#",    "Header 1"),
    ("##",   "Header 2"),
    ("###",  "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

md_text = """
# Authentication
## OAuth 2.0
OAuth 2.0 is an authorization framework...
## JWT Tokens
JSON Web Tokens are self-contained...
# Database
## PostgreSQL
PostgreSQL is a relational database...
"""

chunks = splitter.split_text(md_text)
# Each chunk retains its header hierarchy as metadata:
# chunks[0].content = "OAuth 2.0 is an authorization framework..."
# chunks[0].metadata = {"Header 1": "Authentication", "Header 2": "OAuth 2.0"}
```

**For HTML documents:**
```python
from langchain.text_splitter import HTMLHeaderTextSplitter

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = html_splitter.split_text(html_content)
```

### Strategy 5: Code-Aware Chunking (AST-Based)

For code repositories, split by logical code units (functions, classes).

```python
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

# Language-aware splitting
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=100,
)

# Uses Python-specific separators:
# ["\nclass ", "\ndef ", "\n\ndef ", "\n\n", "\n", " ", ""]

code = """
class UserService:
    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: int):
        return self.db.query(User).get(user_id)

    def create_user(self, data: dict):
        user = User(**data)
        self.db.add(user)
        self.db.commit()
        return user
"""

chunks = python_splitter.split_text(code)
# Splits at class/function boundaries, keeping related code together
```

### Chunk Size Considerations

```
┌──────────────────────────────────────────────────────────────────┐
│                    Chunk Size Trade-offs                         │
├────────────┬──────────────────────┬──────────────────────────────┤
│            │   Small (128-256)    │   Large (512-1024+)          │
├────────────┼──────────────────────┼──────────────────────────────┤
│ Precision  │ High (focused)       │ Lower (more noise)           │
│ Recall     │ Lower (may miss)     │ Higher (more context)        │
│ Embedding  │ Better quality       │ Diluted meaning              │
│ # Chunks   │ Many (higher cost)   │ Fewer (lower cost)           │
│ Context    │ May lack context     │ Self-contained               │
│ Use case   │ Factoid Q&A          │ Summarization, analysis      │
└────────────┴──────────────────────┴──────────────────────────────┘

Common defaults:
  - General Q&A:       512 tokens, 50-token overlap
  - Detailed analysis: 1024 tokens, 100-token overlap
  - Code:              1000-2000 characters, 100-char overlap
  - Legal/medical:     256-512 tokens (precision matters)
```

### Overlap Strategies

```python
# Standard overlap: repeat the last N tokens at the start of the next chunk
# Overlap = 10-20% of chunk_size is a good default

# Example: chunk_size=500, overlap=50
#   Chunk 1: tokens[0:500]
#   Chunk 2: tokens[450:950]     <- 50-token overlap
#   Chunk 3: tokens[900:1400]    <- 50-token overlap

# Why overlap?
# Without overlap:  "...the system uses OAuth 2.0 | for authentication. The token..."
#                   query about "OAuth 2.0 authentication" might miss context
# With overlap:     "...the system uses OAuth 2.0 for" | "OAuth 2.0 for authentication. The token..."
#                   query now matches the full concept in chunk 2
```

### Best Practices for Chunking

1. **Start with RecursiveCharacterTextSplitter** -- it is the best general-purpose default
2. **Use semantic chunking** for quality-critical applications
3. **Match chunk size to your embedding model's sweet spot** (most are trained on 256-512 tokens)
4. **Always add overlap** (10-20% of chunk size)
5. **Preserve metadata** (source file, page number, section header)
6. **Test empirically** -- the "best" strategy depends on your data and queries
7. **Consider multi-granularity** -- store both small and large chunks (parent document retriever)

---

## 3. Embedding Models

Embeddings convert text into dense numerical vectors that capture semantic meaning. Two texts with similar meanings will have vectors that are close together in the embedding space.

### How Embeddings Work (Conceptual)

```
"The cat sat on the mat"  ──encoder──>  [0.12, -0.34, 0.78, ..., 0.05]  (768 dims)
"A kitten rested on a rug" ──encoder──> [0.11, -0.31, 0.75, ..., 0.07]  (768 dims)
                                         ↑ very similar vectors!

"The stock market crashed"  ──encoder──> [0.89, 0.45, -0.23, ..., 0.91]  (768 dims)
                                         ↑ very different vector!

cosine_similarity(cat_vector, kitten_vector) = 0.96  (high - similar!)
cosine_similarity(cat_vector, stock_vector)  = 0.12  (low  - different!)
```

### OpenAI Embeddings

```python
from openai import OpenAI

client = OpenAI()

# text-embedding-3-small: fast, cheap, good quality
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="The cat sat on the mat",
    dimensions=512,   # optional: reduce from default 1536
)
vector = response.data[0].embedding  # List[float] of length 512

# text-embedding-3-large: highest quality
response = client.embeddings.create(
    model="text-embedding-3-large",
    input="The cat sat on the mat",
    dimensions=1024,  # optional: reduce from default 3072
)
```

**Batch embedding (efficient for indexing):**
```python
import numpy as np
from openai import OpenAI

client = OpenAI()

texts = ["chunk 1 text...", "chunk 2 text...", "chunk 3 text...", ...]

# Batch up to 2048 texts per request
def batch_embed(texts: list[str], model: str = "text-embedding-3-small",
                batch_size: int = 2048) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(embeddings)
    return np.array(all_embeddings)

vectors = batch_embed(texts)  # shape: (n_texts, 1536)
```

### Open-Source Embedding Models (sentence-transformers)

```python
from sentence_transformers import SentenceTransformer

# Popular open-source models
model = SentenceTransformer("all-MiniLM-L6-v2")    # 384-dim, fast, decent
# model = SentenceTransformer("BAAI/bge-large-en-v1.5")  # 1024-dim, high quality
# model = SentenceTransformer("intfloat/e5-large-v2")    # 1024-dim, high quality

texts = ["The cat sat on the mat", "A kitten rested on a rug"]

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,   # important for cosine similarity
)
# embeddings.shape: (2, 384)

# Compute similarity
from sentence_transformers.util import cos_sim
similarity = cos_sim(embeddings[0], embeddings[1])
print(f"Similarity: {similarity.item():.4f}")  # ~0.83
```

### Embedding Model Comparison

| Model | Dimensions | MTEB Score | Speed | Cost | Best For |
|-------|-----------|------------|-------|------|----------|
| **text-embedding-3-small** (OpenAI) | 1536 (adjustable) | ~62 | Fast (API) | $0.02/1M tokens | General purpose, low budget |
| **text-embedding-3-large** (OpenAI) | 3072 (adjustable) | ~64 | Fast (API) | $0.13/1M tokens | Highest quality (API) |
| **all-MiniLM-L6-v2** | 384 | ~56 | Very fast | Free (local) | Prototyping, edge deployment |
| **BAAI/bge-large-en-v1.5** | 1024 | ~64 | Medium | Free (local) | High quality, self-hosted |
| **intfloat/e5-large-v2** | 1024 | ~62 | Medium | Free (local) | Instruction-tuned retrieval |
| **Cohere embed-v3** | 1024 | ~65 | Fast (API) | $0.10/1M tokens | Multilingual |
| **nomic-embed-text-v1.5** | 768 | ~62 | Fast | Free (local) | Good balance, Matryoshka |
| **mixedbread mxbai-embed-large** | 1024 | ~65 | Medium | Free (local) | Top open-source quality |

*MTEB = Massive Text Embedding Benchmark (higher is better, max ~70)*

### Matryoshka Embeddings

Matryoshka Representation Learning (MRL) trains embeddings such that the first N dimensions are useful on their own. You can truncate the vector and still get good results.

```
Full embedding (1024 dims):  [0.12, -0.34, 0.78, 0.45, ..., 0.05]
                              |___________________________|
                              First 256 dims are already
                              a good embedding!

Benefits:
  - Store 256-dim vectors instead of 1024-dim -> 4x less storage
  - Faster similarity search
  - Trade-off quality vs efficiency at query time
```

```python
from sentence_transformers import SentenceTransformer

# nomic-embed supports Matryoshka
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

embeddings = model.encode(["search query text"], normalize_embeddings=True)

# Use full 768 dims for best quality
full_vec = embeddings[0]          # (768,)

# Or truncate to 256 for speed/storage
truncated = embeddings[0][:256]   # (256,) -- still works well!

# OpenAI also supports this via the `dimensions` parameter:
# client.embeddings.create(model="text-embedding-3-small", input=text, dimensions=256)
```

### Fine-Tuning Embeddings for Domain-Specific Tasks

When your domain has specialized vocabulary (medical, legal, etc.), fine-tuning embeddings can dramatically improve retrieval.

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Prepare training pairs: (query, positive_passage)
train_examples = [
    InputExample(texts=["What is hypertension?",
                        "Hypertension is a condition where blood pressure is consistently elevated..."]),
    InputExample(texts=["Treatment for type 2 diabetes",
                        "Type 2 diabetes management includes metformin, lifestyle changes..."]),
    # ... hundreds to thousands of pairs
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="./fine-tuned-medical-embeddings",
)
```

### Embedding Best Practices

1. **Use the same model for indexing and querying** -- mixing models produces incompatible vectors
2. **Normalize embeddings** before storing (enables cosine similarity via dot product)
3. **Batch your embedding calls** to maximize throughput
4. **Consider dimensionality reduction** (Matryoshka or PCA) for large-scale deployments
5. **Prefix queries for instruction-tuned models** (e.g., E5 expects `"query: "` prefix)
6. **Benchmark on your data** -- MTEB scores are general; your domain may differ

---

## 4. Vector Search & Reranking

### Similarity Metrics

```
Given vectors A and B:

Cosine Similarity:
  cos(A, B) = (A . B) / (||A|| * ||B||)
  Range: [-1, 1]  (1 = identical, 0 = orthogonal, -1 = opposite)
  Best for: normalized embeddings, most common in RAG

Dot Product (Inner Product):
  dot(A, B) = sum(A_i * B_i)
  Range: (-inf, +inf)
  Note: equals cosine similarity when vectors are normalized
  Best for: when magnitude matters

Euclidean Distance (L2):
  L2(A, B) = sqrt(sum((A_i - B_i)^2))
  Range: [0, +inf)  (0 = identical)
  Best for: when absolute position in space matters
```

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b)

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

# With normalized vectors, cosine_similarity == dot_product
# This is why we normalize embeddings: dot product is faster to compute
```

### Approximate Nearest Neighbors (ANN) Algorithms

Exact nearest neighbor search is O(n) -- too slow for millions of vectors. ANN algorithms trade accuracy for speed.

#### HNSW (Hierarchical Navigable Small World)

The most popular ANN algorithm. Used by Qdrant, Weaviate, pgvector, and more.

```
Concept: Build a multi-layer graph where:
  - Bottom layer: all vectors connected to nearby neighbors
  - Upper layers: progressively sparser "highway" connections
  - Search: start at top (fast, coarse), descend to bottom (precise)

Layer 3 (sparse):    A ──────────────── D
                     │                  │
Layer 2:             A ──── B ──── C ── D
                     │      │     │     │
Layer 1:             A ─ B ─ C ─ D ─ E ─ F
                     │ │ │ │ │ │ │ │ │ │ │
Layer 0 (all nodes): A B C D E F G H I J K ...

Search for query Q:
  1. Start at top layer -> jump to nearest node (fast, few nodes)
  2. Descend to next layer -> refine (more nodes to check)
  3. Continue until bottom layer -> precise neighbors found

Key parameters:
  - M: max connections per node (higher = better recall, more memory)
  - ef_construction: search width during build (higher = better graph, slower build)
  - ef_search: search width during query (higher = better recall, slower search)
```

**Trade-offs:**
- Recall: 95-99%+ typical
- Memory: High (stores graph in RAM)
- Build time: Moderate
- Query time: O(log n) -- very fast

#### IVF (Inverted File Index)

Partition the vector space into clusters, then only search relevant clusters.

```
Build:
  1. Cluster all vectors into K centroids (using K-means)
  2. Assign each vector to its nearest centroid

Search:
  1. Find the nearest nprobe centroids to query
  2. Only search vectors in those clusters

┌────────────┐  ┌────────────┐  ┌────────────┐
│ Cluster 1  │  │ Cluster 2  │  │ Cluster 3  │
│  * * *     │  │    * *     │  │  * * * *   │
│   * *      │  │  * * * *   │  │    * *     │
│  * * *     │  │    * *     │  │  * * *     │
│    C1      │  │    C2 ←Q   │  │    C3      │
└────────────┘  └────────────┘  └────────────┘

Query Q is near C2 -> only search Cluster 2 (and maybe Cluster 1)
Skip Cluster 3 entirely -> much faster!

Key parameters:
  - nlist: number of clusters (more = faster search, lower recall)
  - nprobe: number of clusters to search (more = better recall, slower)
```

#### Product Quantization (PQ)

Compress vectors to reduce memory and speed up distance computation.

```
Original vector (768 floats = 3072 bytes):
  [0.12, -0.34, 0.78, 0.45, ..., 0.05]

Split into 96 sub-vectors of 8 dims each:
  [0.12, -0.34, ..., 0.23]  [0.78, 0.45, ..., 0.11]  ...

Quantize each sub-vector to nearest centroid ID (1 byte):
  [42]  [7]  [128]  ...  [33]

Compressed vector (96 bytes instead of 3072):
  [42, 7, 128, ..., 33]

32x compression! Distance computed via lookup tables.
```

**IVF + PQ are often combined (IVFPQ):** cluster first, then compress. This is how FAISS achieves billion-scale search.

### Hybrid Search (Dense + Sparse)

Combine vector similarity (semantic) with keyword matching (lexical) for better retrieval.

```
┌─────────────────────────────────────────────────┐
│                 Hybrid Search                    │
│                                                  │
│  Query: "Python asyncio event loop"              │
│         │                                        │
│         ├──> Dense Search (embeddings)            │
│         │    Finds: semantically similar docs     │
│         │    e.g., "concurrent programming in     │
│         │          Python using coroutines"       │
│         │                                        │
│         └──> Sparse Search (BM25 / TF-IDF)       │
│              Finds: exact keyword matches         │
│              e.g., "asyncio event loop docs"      │
│                                                  │
│         Results combined via RRF or weighted sum  │
└─────────────────────────────────────────────────┘
```

**Why hybrid works better than either alone:**
- Dense search captures meaning: "car" matches "automobile"
- Sparse search captures exact terms: "Python 3.12" matches "Python 3.12"
- Together, they cover both semantic and lexical relevance

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

# Qdrant natively supports hybrid search
results = client.query_points(
    collection_name="documents",
    prefetch=[
        # Dense vector search
        models.Prefetch(
            query=[0.12, -0.34, ...],  # query embedding
            using="dense",
            limit=100,
        ),
        # Sparse vector search (BM25-like)
        models.Prefetch(
            query=models.SparseVector(
                indices=[1, 42, 789],   # token IDs
                values=[0.5, 1.2, 0.3], # weights
            ),
            using="sparse",
            limit=100,
        ),
    ],
    query=models.FusionQuery(
        fusion=models.Fusion.RRF,  # Reciprocal Rank Fusion
    ),
    limit=10,
)
```

### Reranking

Reranking is a second-stage process that re-scores retrieved documents using a more powerful model.

```
Query -> Vector Search (top 100)  ->  Reranker (top 10)  ->  LLM Context
          Fast, approximate            Slow, precise
          Bi-encoder                   Cross-encoder

Why does reranking help?
  - Bi-encoders (embedding models) encode query and doc independently
    -> fast but miss fine-grained query-document interactions
  - Cross-encoders take (query, document) as a PAIR
    -> slow but much more accurate relevance scoring
```

#### Cross-Encoder Reranking

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

query = "How does authentication work?"
retrieved_docs = [
    "OAuth 2.0 provides delegated authorization...",
    "The database stores user credentials securely...",
    "JWT tokens contain encoded claims for auth...",
    "The server runs on port 8080...",
]

# Cross-encoder scores each (query, doc) pair
pairs = [(query, doc) for doc in retrieved_docs]
scores = reranker.predict(pairs)

# Sort by score (descending)
ranked = sorted(zip(scores, retrieved_docs), reverse=True)
for score, doc in ranked:
    print(f"  [{score:.3f}] {doc[:60]}...")
# [0.987] OAuth 2.0 provides delegated authorization...
# [0.923] JWT tokens contain encoded claims for auth...
# [0.451] The database stores user credentials securely...
# [0.012] The server runs on port 8080...
```

#### Cohere Rerank (API)

```python
import cohere

co = cohere.Client("your-api-key")

results = co.rerank(
    model="rerank-english-v3.0",
    query="How does authentication work?",
    documents=retrieved_docs,
    top_n=5,
    return_documents=True,
)

for result in results.results:
    print(f"  [{result.relevance_score:.3f}] {result.document.text[:60]}...")
```

#### ColBERT (Late Interaction)

ColBERT is a middle ground between bi-encoders and cross-encoders.

```
Bi-encoder:    query -> [single vec]  |  doc -> [single vec]  -> dot product
               Fast, but coarse

Cross-encoder: (query, doc) -> transformer -> score
               Accurate, but slow (can't pre-compute)

ColBERT:       query -> [vec_q1, vec_q2, ..., vec_qN]  (per-token vectors)
               doc   -> [vec_d1, vec_d2, ..., vec_dM]  (per-token vectors, pre-computed!)
               score = sum of max similarities (MaxSim)
               Accurate AND pre-computable!
```

```python
from ragatouille import RAGPretrainedModel

# ColBERT via RAGatouille
rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Index documents (pre-compute token embeddings)
rag.index(
    collection=documents,
    index_name="my_index",
)

# Search (uses late interaction for scoring)
results = rag.search(query="How does authentication work?", k=10)
```

### Reciprocal Rank Fusion (RRF)

Combines ranked lists from multiple retrievers without needing score normalization.

```
Formula: RRF_score(doc) = sum( 1 / (k + rank_i(doc)) ) for each ranker i

Example (k=60):
  Dense search ranks: [Doc_A(1), Doc_B(2), Doc_C(3), Doc_D(4)]
  BM25 search ranks:  [Doc_C(1), Doc_A(2), Doc_E(3), Doc_B(4)]

  RRF(Doc_A) = 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325  <- winner
  RRF(Doc_C) = 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
  RRF(Doc_B) = 1/(60+2) + 1/(60+4) = 0.0161 + 0.0156 = 0.0317
  RRF(Doc_E) = 0        + 1/(60+3) = 0.0159
  RRF(Doc_D) = 1/(60+4) + 0        = 0.0156

  Final ranking: [Doc_A, Doc_C, Doc_B, Doc_E, Doc_D]
```

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine multiple ranked lists using RRF."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Usage
dense_results = ["doc_a", "doc_b", "doc_c", "doc_d"]
sparse_results = ["doc_c", "doc_a", "doc_e", "doc_b"]

fused = reciprocal_rank_fusion([dense_results, sparse_results])
# [("doc_a", 0.0325), ("doc_c", 0.0323), ("doc_b", 0.0317), ...]
```

---

## 5. Vector Databases

### Comparison Table

| Feature | Qdrant | Pinecone | Weaviate | Milvus | ChromaDB | pgvector |
|---------|--------|----------|----------|--------|----------|----------|
| **Type** | Dedicated | Managed SaaS | Dedicated | Dedicated | Lightweight | PG extension |
| **Language** | Rust | Proprietary | Go | Go/C++ | Python | C |
| **Self-hosted** | Yes | No | Yes | Yes | Yes | Yes (PG) |
| **Cloud managed** | Yes | Yes (only) | Yes | Yes (Zilliz) | No | Yes (any PG) |
| **Hybrid search** | Native | No | Native | Yes | No | With pg_trgm |
| **Filtering** | Rich metadata | Metadata | GraphQL-like | Expr-based | Where clause | SQL |
| **Max vectors** | Billions | Billions | Billions | Billions | Millions | Millions |
| **ANN algorithm** | HNSW | Proprietary | HNSW | IVF, HNSW | HNSW | HNSW, IVFFlat |
| **Sparse vectors** | Yes | Yes | No | Yes | No | No |
| **Multi-tenancy** | Payload-based | Namespaces | Tenants | Partitions | Collections | Schemas |
| **Maturity** | Production | Production | Production | Production | Dev/small | Production |
| **Best for** | Production RAG | Serverless, fast start | GraphQL users | Large scale | Prototyping | Already use PG |

### When to Use Which

```
Decision tree:

Already using PostgreSQL?
  └─ Yes -> pgvector (add-on, no new infra)
  └─ No:
      Need managed service? No time for ops?
        └─ Yes -> Pinecone (fully managed, zero ops)
        └─ No:
            Need hybrid search (dense + sparse)?
              └─ Yes -> Qdrant or Milvus
              └─ No:
                  Scale < 100K vectors? Prototyping?
                    └─ Yes -> ChromaDB (simple, in-memory)
                    └─ No:
                        Production, high scale?
                          └─ Qdrant (Rust, fast, great DX)
                          └─ Milvus (battle-tested at scale)
```

### Qdrant Example (Production Favorite)

```python
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=1536,                    # embedding dimension
        distance=Distance.COSINE,
    ),
)

# Upsert documents with metadata
client.upsert(
    collection_name="documents",
    points=[
        models.PointStruct(
            id=1,
            vector=[0.12, -0.34, ...],  # embedding
            payload={
                "text": "OAuth 2.0 provides delegated authorization...",
                "source": "auth_guide.pdf",
                "page": 12,
                "category": "authentication",
                "created_at": "2024-01-15",
            },
        ),
        # ... more points
    ],
)

# Search with metadata filtering
results = client.query_points(
    collection_name="documents",
    query=[0.15, -0.30, ...],  # query embedding
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="category",
                match=models.MatchValue(value="authentication"),
            ),
        ],
    ),
    limit=10,
)
```

### pgvector Example (PostgreSQL Native)

```python
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect("postgresql://user:pass@localhost/mydb")
register_vector(conn)

cur = conn.cursor()

# Enable extension and create table
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id SERIAL PRIMARY KEY,
        content TEXT,
        source VARCHAR(255),
        embedding vector(1536),     -- pgvector column
        metadata JSONB
    );
""")

# Create HNSW index for fast search
cur.execute("""
    CREATE INDEX ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);
""")

# Insert document
cur.execute(
    "INSERT INTO documents (content, source, embedding) VALUES (%s, %s, %s)",
    ("OAuth 2.0 provides...", "auth_guide.pdf", embedding_vector),
)

# Search with SQL (cosine distance: <=> operator)
cur.execute("""
    SELECT content, source, 1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    WHERE metadata->>'category' = 'authentication'
    ORDER BY embedding <=> %s::vector
    LIMIT 10;
""", (query_vector, query_vector))

results = cur.fetchall()
conn.commit()
```

### ChromaDB Example (Prototyping)

```python
import chromadb

client = chromadb.Client()  # in-memory
# client = chromadb.PersistentClient(path="./chroma_data")  # persistent

collection = client.create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"},
)

# Add documents (ChromaDB can auto-embed if you configure an embedding function)
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=[
        "OAuth 2.0 provides delegated authorization...",
        "JWT tokens contain encoded claims...",
        "The database stores user credentials...",
    ],
    metadatas=[
        {"source": "auth_guide.pdf", "page": 12},
        {"source": "auth_guide.pdf", "page": 15},
        {"source": "db_design.pdf", "page": 3},
    ],
    embeddings=[[0.12, -0.34, ...], [0.45, 0.67, ...], [0.89, -0.12, ...]],
)

# Query
results = collection.query(
    query_embeddings=[[0.15, -0.30, ...]],
    n_results=5,
    where={"source": "auth_guide.pdf"},
)
```

### Metadata Filtering

Metadata filtering is essential for scoping search results (e.g., by user, tenant, date, document type).

```
Without filtering:
  Query "auth tokens" searches ALL 10M vectors -> slow, noisy

With filtering:
  Query "auth tokens" WHERE source="auth_guide.pdf" AND year>=2024
  Searches only ~1000 relevant vectors -> fast, precise
```

**Key consideration:** Pre-filtering vs post-filtering.
- **Pre-filtering:** Filter first, then ANN search on subset (Qdrant, Weaviate default)
- **Post-filtering:** ANN search first, then filter results (risk: fewer than K results)
- Most modern vector DBs do **pre-filtering** for correctness.

### Scalability Considerations

```
Small scale (< 1M vectors):
  - ChromaDB, pgvector, single-node Qdrant
  - Everything fits in RAM

Medium scale (1M - 100M vectors):
  - Qdrant, Weaviate, Milvus with proper indexing
  - Consider quantization (reduces memory 4-8x)
  - HNSW with tuned parameters

Large scale (100M+ vectors):
  - Distributed Qdrant / Milvus cluster
  - Sharding across nodes
  - IVF + PQ for memory efficiency
  - Consider tiered storage (hot/warm)

Key metrics to monitor:
  - Query latency (p50, p95, p99)
  - Recall@K (are you finding the right docs?)
  - Index build time
  - Memory usage per vector
  - QPS (queries per second)
```

---

## 6. Document Processing

### PDF Parsing

PDFs are the most common document format in enterprise RAG. Parsing quality varies enormously.

```python
# --- Option 1: PyPDF2 (simple text extraction) ---
from pypdf import PdfReader

reader = PdfReader("document.pdf")
for page in reader.pages:
    text = page.extract_text()
    print(text)

# --- Option 2: pdfplumber (better layout preservation, table support) ---
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()

        # Extract tables
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                print(row)

# --- Option 3: unstructured (best quality, handles complex layouts) ---
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="document.pdf",
    strategy="hi_res",           # Use OCR + layout detection
    infer_table_structure=True,  # Extract tables as HTML
    extract_images_in_pdf=True,  # Extract embedded images
)

for element in elements:
    print(f"Type: {type(element).__name__}")
    print(f"Text: {element.text[:100]}")
    print(f"Metadata: {element.metadata}")
    print()
```

### DOCX Processing

```python
from docx import Document

doc = Document("report.docx")

# Extract paragraphs with style info
for para in doc.paragraphs:
    style = para.style.name  # "Heading 1", "Normal", etc.
    text = para.text
    if style.startswith("Heading"):
        print(f"\n{'#' * int(style[-1])} {text}")
    else:
        print(text)

# Extract tables
for table in doc.tables:
    for row in table.rows:
        row_data = [cell.text for cell in row.cells]
        print(" | ".join(row_data))
```

### HTML Parsing

```python
from bs4 import BeautifulSoup
import requests

# Fetch and parse HTML
response = requests.get("https://docs.example.com/api")
soup = BeautifulSoup(response.text, "html.parser")

# Remove scripts, styles, navigation
for tag in soup(["script", "style", "nav", "footer", "header"]):
    tag.decompose()

# Extract structured content
content = []
for element in soup.find_all(["h1", "h2", "h3", "p", "li", "code", "pre"]):
    content.append({
        "tag": element.name,
        "text": element.get_text(strip=True),
    })

# Or just get clean text
clean_text = soup.get_text(separator="\n", strip=True)
```

### OCR for Scanned Documents

```python
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

# Convert scanned PDF pages to images
images = convert_from_path("scanned_document.pdf", dpi=300)

# OCR each page
for i, image in enumerate(images):
    text = pytesseract.image_to_string(image, lang="eng")
    print(f"--- Page {i + 1} ---")
    print(text)

# For better quality OCR, preprocess the image
import cv2
import numpy as np

def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Improve OCR accuracy with preprocessing."""
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(binary, h=30)
    return Image.fromarray(denoised)
```

### Table Extraction

```python
import pdfplumber
import pandas as pd

with pdfplumber.open("financial_report.pdf") as pdf:
    page = pdf.pages[0]
    tables = page.extract_tables()

    for table in tables:
        df = pd.DataFrame(table[1:], columns=table[0])
        print(df.to_markdown())

        # Convert table to text for embedding
        table_text = df.to_string(index=False)
        # Or create a structured description
        table_desc = f"Table with columns: {', '.join(df.columns)}. "
        table_desc += f"Contains {len(df)} rows. "
        for _, row in df.iterrows():
            table_desc += " | ".join(str(v) for v in row.values) + ". "
```

### Multimodal RAG

Handle images, charts, and diagrams within documents using vision models.

```python
import base64
from openai import OpenAI

client = OpenAI()

def describe_image_for_rag(image_path: str) -> str:
    """Use a vision model to generate text description of an image."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in detail for a knowledge base. "
                            "Include all text, data, relationships, and key information."
                        ),
                    },
                    {
                        "type": "image_url",
                        "url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content

# The description is then chunked and embedded like any other text
description = describe_image_for_rag("architecture_diagram.png")
# -> "The diagram shows a microservices architecture with 5 services:
#     Auth Service, User Service, Order Service, Payment Service, and
#     Notification Service. They communicate via an API Gateway..."
```

### Complete Document Processing Pipeline

```python
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ProcessedChunk:
    text: str
    metadata: dict
    source: str
    page: int | None = None

def process_document(file_path: str) -> list[ProcessedChunk]:
    """Route document to appropriate parser based on file type."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    parsers = {
        ".pdf": parse_pdf,
        ".docx": parse_docx,
        ".html": parse_html,
        ".htm": parse_html,
        ".md": parse_markdown,
        ".txt": parse_text,
    }

    parser = parsers.get(suffix)
    if not parser:
        raise ValueError(f"Unsupported file type: {suffix}")

    return parser(file_path)

def parse_pdf(file_path: str) -> list[ProcessedChunk]:
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(filename=file_path, strategy="fast")
    chunks = []
    for el in elements:
        chunks.append(ProcessedChunk(
            text=el.text,
            metadata={
                "element_type": type(el).__name__,
                "page_number": el.metadata.page_number,
            },
            source=file_path,
            page=el.metadata.page_number,
        ))
    return chunks

# ... similar for other formats
```

---

## 7. Advanced RAG Techniques

### Query Transformation

Raw user queries are often vague or poorly worded. Transforming them before retrieval significantly improves results.

#### HyDE (Hypothetical Document Embeddings)

Instead of embedding the short query, generate a hypothetical answer and embed that. The hypothetical answer is closer in embedding space to real relevant documents.

```
Original Query: "What causes rain?"
         |
         v
HyDE generates hypothetical answer:
"Rain is caused by the water cycle. Water evaporates from
 oceans, lakes, and rivers due to solar heat. The water
 vapor rises, cools in the atmosphere, condenses into
 clouds, and eventually falls as precipitation when
 droplets become heavy enough..."
         |
         v
Embed this hypothetical answer (closer to real docs in vector space)
         |
         v
Retrieve similar REAL documents from the vector store
         |
         v
Generate final answer using retrieved real documents
```

```python
from openai import OpenAI

client = OpenAI()

def hyde_retrieve(query: str, collection, embed_fn) -> list[str]:
    """HyDE: generate hypothetical doc, embed it, retrieve real docs."""

    # Step 1: Generate hypothetical document
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Given a question, write a detailed paragraph that would "
                    "answer it. Write as if you are writing a textbook passage."
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0.7,
    )
    hypothetical_doc = response.choices[0].message.content

    # Step 2: Embed the hypothetical document (not the original query!)
    hyp_embedding = embed_fn(hypothetical_doc)

    # Step 3: Retrieve real documents using this embedding
    results = collection.query(
        query_embeddings=[hyp_embedding],
        n_results=10,
    )

    return results["documents"][0]
```

**When HyDE helps:** Queries that are short, vague, or use different vocabulary than the source documents.
**When HyDE hurts:** When the hypothetical answer leads the retrieval astray (wrong topic).

#### Multi-Query Retrieval

Generate multiple reformulations of the query, retrieve for each, then combine results.

```python
from openai import OpenAI

client = OpenAI()

def multi_query_retrieve(query: str, collection, embed_fn, n_queries: int = 3):
    """Generate multiple query perspectives and combine results."""

    # Step 1: Generate alternative queries
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Generate {n_queries} different versions of the given "
                    "question to help retrieve relevant documents. "
                    "Return each version on a new line."
                ),
            },
            {"role": "user", "content": query},
        ],
    )

    queries = [query] + response.choices[0].message.content.strip().split("\n")

    # Step 2: Retrieve for each query
    all_docs = set()
    for q in queries:
        q_embedding = embed_fn(q)
        results = collection.query(query_embeddings=[q_embedding], n_results=5)
        for doc in results["documents"][0]:
            all_docs.add(doc)

    return list(all_docs)
```

#### Step-Back Prompting

For specific questions, first ask a more general question to get background context.

```python
def step_back_retrieve(query: str, collection, embed_fn):
    """Generate a broader question first for better context."""

    # Step 1: Generate step-back question
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Given a specific question, generate a more general "
                    "step-back question that would provide useful background "
                    "context. Example:\n"
                    "Specific: 'Why did GDP of Japan fall in Q3 2024?'\n"
                    "Step-back: 'What are the key economic factors affecting Japan?'"
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    step_back_query = response.choices[0].message.content

    # Step 2: Retrieve for both original and step-back queries
    original_docs = retrieve(query, collection, embed_fn)
    background_docs = retrieve(step_back_query, collection, embed_fn)

    return background_docs + original_docs  # background first for context
```

### Contextual Compression

After retrieval, compress the chunks to keep only the parts relevant to the query.

```python
from openai import OpenAI

client = OpenAI()

def compress_context(query: str, documents: list[str]) -> list[str]:
    """Extract only the relevant parts from each retrieved document."""
    compressed = []

    for doc in documents:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract only the parts of the following document that "
                        "are relevant to answering the question. If no part is "
                        "relevant, respond with 'NOT_RELEVANT'."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nDocument: {doc}",
                },
            ],
        )

        result = response.choices[0].message.content
        if result != "NOT_RELEVANT":
            compressed.append(result)

    return compressed
```

### Parent Document Retriever

Embed small chunks for precise retrieval, but pass the larger parent document to the LLM for full context.

```
┌──────────────────────────────────────────────────┐
│              Parent Document                      │
│  "Chapter 3: Authentication                       │
│                                                   │
│   OAuth 2.0 is an authorization framework that    │
│   enables applications to obtain limited access   │
│   to user accounts. It works by delegating user   │
│   authentication to the service that hosts the    │
│   user account and authorizing third-party        │
│   applications to access the user account.        │
│                                                   │
│   The OAuth 2.0 flow involves several steps:      │
│   1. The client requests authorization            │
│   2. The user grants authorization                │
│   3. The client receives an authorization code    │
│   4. The client exchanges the code for a token    │
│   ..."                                            │
│                                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Child 1  │ │ Child 2  │ │ Child 3  │ <- embed  │
│  │ (small)  │ │ (small)  │ │ (small)  │    these  │
│  └──────────┘ └──────────┘ └──────────┘          │
└──────────────────────────────────────────────────┘

Search matches Child 2 -> return Parent Document to LLM
```

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Small chunks for retrieval
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
# Large chunks for context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

vectorstore = Chroma(
    collection_name="children",
    embedding_function=OpenAIEmbeddings(),
)

store = InMemoryStore()  # stores parent documents

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Add documents -- auto-splits into parents and children
retriever.add_documents(documents)

# Query -- finds matching child, returns parent
results = retriever.invoke("How does OAuth 2.0 work?")
# Returns the full parent chunk (~2000 chars) even though the match
# was on a small child chunk (~200 chars)
```

### Self-RAG (Self-Reflective Retrieval)

The model decides whether retrieval is needed, evaluates retrieved documents, and checks its own answer.

```
Query
  |
  v
Should I retrieve?  --No-->  Generate directly (simple/conversational queries)
  |
  Yes
  v
Retrieve documents
  |
  v
For each document:
  Is it relevant?  --No-->  Discard
  |
  Yes
  v
Generate answer segment
  |
  v
Is the answer supported by the document?  --No-->  Retry / rephrase
  |
  Yes
  v
Is the answer useful?  --No-->  Generate alternative
  |
  Yes
  v
Return answer with citations
```

```python
def self_rag(query: str, retriever, llm) -> str:
    """Self-RAG: decide when to retrieve and verify answers."""

    # Step 1: Decide if retrieval is needed
    needs_retrieval = llm.predict(
        f"Does this query require external knowledge to answer accurately? "
        f"Query: '{query}'. Answer YES or NO."
    ).strip().upper()

    if needs_retrieval == "NO":
        return llm.predict(query)

    # Step 2: Retrieve and filter relevant documents
    docs = retriever.invoke(query)
    relevant_docs = []
    for doc in docs:
        is_relevant = llm.predict(
            f"Is this document relevant to the query '{query}'?\n"
            f"Document: {doc.page_content[:500]}\n"
            f"Answer YES or NO."
        ).strip().upper()
        if is_relevant == "YES":
            relevant_docs.append(doc)

    if not relevant_docs:
        return "I could not find relevant information to answer this question."

    # Step 3: Generate answer
    context = "\n\n".join(d.page_content for d in relevant_docs)
    answer = llm.predict(
        f"Based on the following context, answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}"
    )

    # Step 4: Verify answer is supported
    is_supported = llm.predict(
        f"Is the following answer fully supported by the context?\n"
        f"Context:\n{context}\n\nAnswer: {answer}\n"
        f"Respond SUPPORTED or NOT_SUPPORTED."
    ).strip().upper()

    if "NOT_SUPPORTED" in is_supported:
        answer += "\n\n[Note: This answer may not be fully supported by the sources.]"

    return answer
```

### Corrective RAG (CRAG)

If retrieval quality is poor, fall back to web search or other sources.

```
Query -> Retrieve from vector store
              |
              v
         Evaluate retrieval quality
        /         |          \
   Correct    Ambiguous    Incorrect
      |           |            |
      v           v            v
  Use docs    Refine +      Web search
  as-is       knowledge     (fallback)
              refinement
      \          |           /
       v         v          v
       Combine and generate answer
```

```python
def corrective_rag(query: str, retriever, web_search_fn, llm) -> str:
    """CRAG: assess retrieval quality and correct if needed."""

    docs = retriever.invoke(query)

    # Evaluate retrieval quality
    eval_prompt = (
        f"Rate the relevance of these documents to the query on a scale of "
        f"1-5.\nQuery: {query}\nDocuments:\n"
    )
    for i, doc in enumerate(docs):
        eval_prompt += f"\n{i+1}. {doc.page_content[:200]}..."

    score = float(llm.predict(eval_prompt + "\nReturn just the number.").strip())

    if score >= 4:
        # Correct: use retrieved docs directly
        context_docs = docs
    elif score >= 2:
        # Ambiguous: supplement with web search
        web_results = web_search_fn(query)
        context_docs = docs + web_results
    else:
        # Incorrect: rely on web search
        context_docs = web_search_fn(query)

    context = "\n\n".join(d.page_content for d in context_docs)
    return llm.predict(f"Context:\n{context}\n\nQuestion: {query}")
```

### Agentic RAG

An LLM agent decides which tools to use, when to retrieve, and how to combine information.

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI

# Create retriever tools for different document collections
auth_retriever_tool = create_retriever_tool(
    auth_retriever,
    name="search_auth_docs",
    description="Search authentication and authorization documentation",
)

api_retriever_tool = create_retriever_tool(
    api_retriever,
    name="search_api_docs",
    description="Search API reference documentation",
)

db_retriever_tool = create_retriever_tool(
    db_retriever,
    name="search_db_docs",
    description="Search database design and query documentation",
)

# Create agent that can choose which retriever to use
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [auth_retriever_tool, api_retriever_tool, db_retriever_tool]

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# The agent decides:
# - Which retriever(s) to call
# - Whether to call multiple retrievers
# - How to combine the results
result = agent_executor.invoke(
    {"input": "How does the API authenticate database connections?"}
)
# Agent might call search_auth_docs AND search_db_docs and combine results
```

### Graph RAG

Combine knowledge graphs with RAG for better reasoning over entity relationships.

```
Traditional RAG:
  Query -> find similar text chunks -> answer
  Problem: misses relationships between entities across chunks

Graph RAG:
  1. Build a knowledge graph from documents:
     (User) --[authenticates_via]--> (OAuth)
     (OAuth) --[issues]--> (Access Token)
     (Access Token) --[grants_access_to]--> (API)

  2. Query traverses the graph:
     "How does a user access the API?"
     -> User -> authenticates_via -> OAuth -> issues -> Token -> grants_access_to -> API

  3. Retrieved subgraph + text chunks -> LLM generates answer
```

```python
# Using LangChain + Neo4j for Graph RAG
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")

# Build graph from documents (using LLM to extract entities/relations)
from langchain_experimental.graph_transformers import LLMGraphTransformer

llm = ChatOpenAI(model="gpt-4o", temperature=0)
transformer = LLMGraphTransformer(llm=llm)

# Extract graph from documents
graph_documents = transformer.convert_to_graph_documents(documents)
graph.add_graph_documents(graph_documents)

# Query using natural language -> Cypher -> graph traversal
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
)

result = chain.invoke({"query": "How does a user access the API?"})
# Internally generates Cypher:
# MATCH (u:User)-[r1]->(auth)-[r2]->(token)-[r3]->(api:API) RETURN ...
```

### Multi-Hop RAG

For complex questions that require chaining multiple retrieval steps.

```
Question: "What is the capital of the country where the inventor of Python was born?"

Hop 1: "Who invented Python?"
  -> Retrieve: "Python was created by Guido van Rossum"

Hop 2: "Where was Guido van Rossum born?"
  -> Retrieve: "Guido van Rossum was born in the Netherlands"

Hop 3: "What is the capital of the Netherlands?"
  -> Retrieve: "Amsterdam is the capital of the Netherlands"

Final answer: "Amsterdam"
```

```python
def multi_hop_rag(query: str, retriever, llm, max_hops: int = 3) -> str:
    """Iteratively retrieve and reason for complex queries."""
    context_so_far = ""

    for hop in range(max_hops):
        # Decide what to search for next
        if hop == 0:
            search_query = query
        else:
            search_query = llm.predict(
                f"Original question: {query}\n"
                f"Information gathered so far: {context_so_far}\n"
                f"What specific piece of information do we still need? "
                f"Generate a search query to find it. "
                f"If we have enough information, respond with 'DONE'."
            )

            if "DONE" in search_query:
                break

        # Retrieve
        docs = retriever.invoke(search_query)
        new_info = "\n".join(d.page_content for d in docs[:3])
        context_so_far += f"\n\n[Hop {hop + 1}]: {new_info}"

    # Generate final answer
    return llm.predict(
        f"Based on the following information, answer the question.\n\n"
        f"Information:\n{context_so_far}\n\nQuestion: {query}"
    )
```

---

## 8. Evaluation

### Why RAG Evaluation is Hard

RAG has two components that can fail independently:
1. **Retrieval** -- did we find the right documents?
2. **Generation** -- did the LLM produce a correct answer from those documents?

You need metrics for both.

### Retrieval Metrics

```
Given:
  Query Q with known relevant documents: {D1, D3, D5}
  System retrieves (ranked): [D1, D2, D3, D4, D5]

Precision@K: What fraction of retrieved docs are relevant?
  Precision@3 = |{D1, D3} ∩ {D1, D2, D3}| / 3 = 2/3 = 0.667

Recall@K: What fraction of relevant docs were retrieved?
  Recall@3 = |{D1, D3} ∩ {D1, D2, D3}| / 3 = 2/3 = 0.667
  Recall@5 = |{D1, D3, D5} ∩ {D1, D2, D3, D4, D5}| / 3 = 3/3 = 1.0

MRR (Mean Reciprocal Rank): How high is the first relevant result?
  First relevant doc D1 is at rank 1 -> RR = 1/1 = 1.0

NDCG (Normalized Discounted Cumulative Gain):
  Accounts for graded relevance and position
  Higher-ranked relevant docs contribute more to the score
  NDCG@5 for our example ≈ 0.88
```

```python
import numpy as np

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-K retrieved that are relevant."""
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / k

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs found in top-K."""
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / len(relevant) if relevant else 0.0

def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved: list[str], relevance_scores: dict[str, int], k: int) -> float:
    """NDCG accounting for graded relevance."""
    dcg = sum(
        relevance_scores.get(doc, 0) / np.log2(i + 2)
        for i, doc in enumerate(retrieved[:k])
    )
    ideal = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0
```

### Generation Metrics

| Metric | What it Measures | How |
|--------|-----------------|-----|
| **Faithfulness** | Is the answer grounded in the retrieved context? | Check each claim against source docs |
| **Answer Relevancy** | Does the answer address the question? | Generate questions from the answer; compare to original |
| **Context Relevancy** | Are the retrieved docs relevant to the question? | Score each retrieved doc against the query |
| **Context Utilization** | How much of the context was used in the answer? | Check answer claims against context |

### RAGAS Framework

RAGAS (Retrieval Augmented Generation Assessment) is the standard framework for evaluating RAG pipelines.

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": [
        "How does OAuth 2.0 work?",
        "What is JWT?",
    ],
    "answer": [
        "OAuth 2.0 works by delegating user authentication...",
        "JWT is a JSON Web Token used for...",
    ],
    "contexts": [
        ["OAuth 2.0 is an authorization framework that enables..."],
        ["JSON Web Tokens (JWT) are an open standard (RFC 7519)..."],
    ],
    "ground_truth": [
        "OAuth 2.0 is an authorization framework that enables applications...",
        "JWT (JSON Web Token) is a compact, URL-safe means of representing claims...",
    ],
}

dataset = Dataset.from_dict(eval_data)

# Evaluate
results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,        # Is answer grounded in context?
        answer_relevancy,    # Does answer match the question?
        context_precision,   # Are retrieved docs relevant?
        context_recall,      # Did we retrieve all needed info?
    ],
)

print(results)
# {'faithfulness': 0.92, 'answer_relevancy': 0.88,
#  'context_precision': 0.85, 'context_recall': 0.90}
```

### LLM-as-Judge

Use a powerful LLM to evaluate answer quality.

```python
from openai import OpenAI

client = OpenAI()

def llm_judge(question: str, answer: str, context: str, reference: str) -> dict:
    """Use GPT-4o to evaluate RAG answer quality."""

    evaluation_prompt = f"""Evaluate the following RAG system output.

Question: {question}
Retrieved Context: {context}
System Answer: {answer}
Reference Answer: {reference}

Rate each dimension from 1-5 and explain:
1. Faithfulness: Is the answer supported by the context? (no hallucination)
2. Relevancy: Does the answer address the question?
3. Completeness: Does the answer cover all aspects of the question?
4. Conciseness: Is the answer appropriately concise?

Return as JSON: {{"faithfulness": score, "relevancy": score,
"completeness": score, "conciseness": score, "explanation": "..."}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": evaluation_prompt}],
        response_format={"type": "json_object"},
    )

    import json
    return json.loads(response.choices[0].message.content)

# Usage
scores = llm_judge(
    question="How does OAuth 2.0 work?",
    answer="OAuth 2.0 delegates authentication...",
    context="OAuth 2.0 is an authorization framework...",
    reference="OAuth 2.0 works by...",
)
print(scores)
# {'faithfulness': 5, 'relevancy': 4, 'completeness': 3, 'conciseness': 5,
#  'explanation': 'The answer is faithful to the context but misses...'}
```

### End-to-End Evaluation Pipeline

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class EvalCase:
    question: str
    expected_answer: str
    expected_sources: list[str]  # document IDs that should be retrieved

@dataclass
class EvalResult:
    question: str
    retrieved_docs: list[str]
    generated_answer: str
    precision_at_5: float
    recall_at_5: float
    mrr: float
    faithfulness: float
    relevancy: float

def evaluate_rag_pipeline(
    rag_fn: Callable,
    eval_cases: list[EvalCase],
) -> list[EvalResult]:
    """Evaluate a complete RAG pipeline."""
    results = []

    for case in eval_cases:
        # Run the RAG pipeline
        answer, retrieved_doc_ids = rag_fn(case.question)

        # Retrieval metrics
        relevant = set(case.expected_sources)
        p_at_5 = precision_at_k(retrieved_doc_ids, relevant, 5)
        r_at_5 = recall_at_k(retrieved_doc_ids, relevant, 5)
        rr = mrr(retrieved_doc_ids, relevant)

        # Generation metrics (via LLM judge)
        scores = llm_judge(
            question=case.question,
            answer=answer,
            context="...",
            reference=case.expected_answer,
        )

        results.append(EvalResult(
            question=case.question,
            retrieved_docs=retrieved_doc_ids,
            generated_answer=answer,
            precision_at_5=p_at_5,
            recall_at_5=r_at_5,
            mrr=rr,
            faithfulness=scores["faithfulness"] / 5.0,
            relevancy=scores["relevancy"] / 5.0,
        ))

    return results
```

---

## 9. Production Best Practices (2024-2025)

### Chunking

- **Use semantic or recursive chunking** -- avoid naive fixed-size splitting
- **Add 10-20% overlap** between chunks to preserve boundary context
- **Store metadata** with every chunk: source, page, section, timestamp
- **Experiment with chunk sizes**: start with 512 tokens, test 256 and 1024
- **Use parent document retriever** for the best of both worlds

### Retrieval

- **Always rerank** -- a cross-encoder or Cohere rerank model after initial retrieval dramatically improves precision
- **Use hybrid search** (BM25 + dense vectors) -- catches both lexical and semantic matches
- **Metadata filtering** -- scope searches by tenant, document type, date range
- **Set appropriate K** -- retrieve top-20 to top-100, then rerank to top-5 to top-10

### Embedding

- **Use the same model for indexing and querying** -- this is critical
- **Normalize embeddings** for cosine similarity
- **Consider Matryoshka embeddings** for storage efficiency
- **Batch embed** during indexing (up to 2048 texts per API call for OpenAI)

### System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Production RAG Architecture                        │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │  Client   │───>│  API Gateway  │───>│  RAG Service             │  │
│  │  App      │    │  (rate limit) │    │                          │  │
│  └──────────┘    └──────────────┘    │  1. Query transform       │  │
│                                      │  2. Embed query           │  │
│                                      │  3. Hybrid search         │  │
│  ┌──────────┐    ┌──────────────┐    │  4. Rerank                │  │
│  │  Doc      │───>│  Ingestion    │   │  5. Generate answer      │  │
│  │  Upload   │    │  Pipeline     │    └──────┬───────────────────┘  │
│  └──────────┘    │  (chunking,   │           │                      │
│                  │   embedding)  │    ┌──────┴───────────────────┐  │
│                  └──────┬───────┘    │  Vector DB (Qdrant)       │  │
│                         │            │  + Metadata Store          │  │
│                         └───────────>│  + BM25 Index              │  │
│                                      └──────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Supporting Infrastructure                                    │   │
│  │  - Redis (query cache)                                        │   │
│  │  - PostgreSQL (document registry, user data)                  │   │
│  │  - Object Storage (raw documents)                             │   │
│  │  - Monitoring (retrieval quality, latency, cost)              │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Caching

```python
import hashlib
import json
import redis

redis_client = redis.Redis(host="localhost", port=6379, db=0)

def cached_rag_query(query: str, ttl: int = 3600) -> str:
    """Cache RAG responses to avoid redundant LLM calls."""
    cache_key = f"rag:{hashlib.sha256(query.encode()).hexdigest()}"

    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)["answer"]

    # Run RAG pipeline
    answer = rag_pipeline(query)

    # Cache result
    redis_client.setex(
        cache_key,
        ttl,
        json.dumps({"query": query, "answer": answer}),
    )

    return answer
```

### Document Versioning & Incremental Indexing

```python
from datetime import datetime
from hashlib import sha256

def should_reindex(doc_path: str, doc_registry: dict) -> bool:
    """Check if a document needs re-indexing based on content hash."""
    with open(doc_path, "rb") as f:
        current_hash = sha256(f.read()).hexdigest()

    stored = doc_registry.get(doc_path)
    if not stored or stored["hash"] != current_hash:
        return True
    return False

def incremental_index(
    doc_paths: list[str],
    vector_store,
    doc_registry: dict,
):
    """Only re-index documents that have changed."""
    for path in doc_paths:
        if should_reindex(path, doc_registry):
            # Delete old vectors for this document
            vector_store.delete(
                filter={"source": path},
            )

            # Process and index new version
            chunks = process_document(path)
            embeddings = embed_chunks(chunks)
            vector_store.upsert(chunks, embeddings)

            # Update registry
            with open(path, "rb") as f:
                doc_registry[path] = {
                    "hash": sha256(f.read()).hexdigest(),
                    "indexed_at": datetime.utcnow().isoformat(),
                }
```

### Monitoring & Observability

Key metrics to track in production:

| Metric | What to Monitor | Alert Threshold |
|--------|----------------|-----------------|
| **Retrieval latency** | p50, p95, p99 | p95 > 500ms |
| **Generation latency** | Time to first token, total | p95 > 5s |
| **Retrieval relevance** | Avg reranker score of top-K | < 0.3 avg score |
| **Empty retrieval rate** | Queries with 0 relevant results | > 10% |
| **Hallucination rate** | Faithfulness score from eval | < 0.8 avg |
| **User feedback** | Thumbs up/down on answers | < 70% positive |
| **Token usage** | Tokens per query (cost) | Spike detection |
| **Index freshness** | Time since last document update | > 24h stale |

```python
import time
import logging
from dataclasses import dataclass

logger = logging.getLogger("rag_monitoring")

@dataclass
class RAGMetrics:
    query: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    num_retrieved: int
    avg_reranker_score: float
    total_tokens: int

def monitored_rag_pipeline(query: str) -> tuple[str, RAGMetrics]:
    """RAG pipeline with comprehensive monitoring."""

    # Retrieval
    t0 = time.perf_counter()
    docs, scores = retrieve_and_rerank(query)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    # Generation
    t1 = time.perf_counter()
    answer, usage = generate_answer(query, docs)
    generation_ms = (time.perf_counter() - t1) * 1000

    metrics = RAGMetrics(
        query=query,
        retrieval_latency_ms=retrieval_ms,
        generation_latency_ms=generation_ms,
        num_retrieved=len(docs),
        avg_reranker_score=sum(scores) / len(scores) if scores else 0,
        total_tokens=usage.total_tokens,
    )

    # Log and alert
    logger.info(f"RAG query completed", extra={"metrics": metrics.__dict__})

    if metrics.avg_reranker_score < 0.3:
        logger.warning(f"Low retrieval quality for query: {query}")

    if metrics.retrieval_latency_ms > 500:
        logger.warning(f"Slow retrieval: {metrics.retrieval_latency_ms:.0f}ms")

    return answer, metrics
```

### Common Production Pitfalls

1. **Chunk boundary issues** -- important info split across chunks; fix with overlap and parent doc retriever
2. **Stale index** -- documents updated but index not refreshed; fix with incremental indexing
3. **Over-retrieval** -- too many irrelevant chunks pollute context; fix with reranking
4. **Under-retrieval** -- missing relevant docs; fix with hybrid search and query expansion
5. **Context window overflow** -- too many chunks exceed LLM context; fix with contextual compression
6. **Embedding model mismatch** -- different models for indexing vs querying; always use the same model
7. **Missing metadata** -- cannot filter by tenant/date; always store rich metadata
8. **No evaluation** -- no way to know if quality is degrading; implement RAGAS or LLM-as-judge

---

## 10. Q&A Section

### Q1: What is RAG and when would you use it vs fine-tuning?

**A:** RAG (Retrieval-Augmented Generation) augments LLM responses by retrieving relevant documents from an external knowledge base at query time. The retrieved context is injected into the prompt so the model generates grounded, factual answers.

**Use RAG when:**
- Knowledge changes frequently (no retraining needed)
- You need source attribution and traceability
- You have large document collections to query
- You want to reduce hallucinations on factual questions

**Use fine-tuning when:**
- You need to change the model's behavior, tone, or output format
- Teaching specialized reasoning or a custom DSL
- Reducing prompt size by internalizing instructions
- The knowledge is stable and unlikely to change

**Use both when:**
- Fine-tune the model to better follow RAG instructions, then use RAG for knowledge

---

### Q2: Walk me through the complete RAG pipeline, indexing and query.

**A:**

**Indexing (offline):**
1. **Ingest** raw documents (PDF, DOCX, HTML, etc.)
2. **Parse** documents into structured text (handle tables, images, formatting)
3. **Chunk** text into passages (512 tokens with 50-token overlap, using recursive splitting)
4. **Embed** each chunk using an embedding model (e.g., `text-embedding-3-small`)
5. **Store** vectors + metadata in a vector database (e.g., Qdrant)

**Query (online):**
1. **Receive** user query
2. **Transform** query (optional: HyDE, multi-query, step-back)
3. **Embed** the query using the same embedding model
4. **Retrieve** top-K similar chunks (hybrid search: vector + BM25)
5. **Rerank** results using a cross-encoder
6. **Construct** prompt with retrieved context
7. **Generate** answer with the LLM
8. **Return** answer with source citations

---

### Q3: Explain different chunking strategies and when to use each.

**A:**

| Strategy | How | Best For |
|----------|-----|----------|
| **Fixed-size** | Split every N chars/tokens | Simple, predictable |
| **Recursive character** | Try paragraph -> sentence -> word boundaries | General purpose (best default) |
| **Semantic** | Split at topic boundaries (via embedding similarity) | High-quality, topic-coherent chunks |
| **Document-structure** | Split by headers, sections | Structured docs (Markdown, HTML) |
| **Code-aware** | Split by functions, classes (AST) | Code repositories |

**Key considerations:**
- Chunk size: 512 tokens is a good default; smaller for precision, larger for context
- Always use overlap (10-20% of chunk size) to prevent information loss at boundaries
- Match chunk size to your embedding model's training data (most trained on 256-512 token passages)

---

### Q4: How does reranking improve RAG and when should you use it?

**A:** Reranking uses a cross-encoder model to re-score retrieved documents by processing the (query, document) pair together through a transformer. This is more accurate than the initial bi-encoder retrieval because:

- **Bi-encoders** (embedding models) encode query and document independently -- fast but miss fine-grained interactions
- **Cross-encoders** process query and document together -- slower but much more precise

**Typical flow:**
1. Vector search retrieves top-100 candidates (fast, coarse)
2. Cross-encoder reranks to top-10 (slow, precise)
3. Top-10 go to the LLM

**When to use:** Always in production. The quality improvement is significant and the cost is low (reranking 100 docs takes ~50ms with a small cross-encoder). The only exception is extremely latency-sensitive applications where every millisecond counts.

---

### Q5: What is hybrid search and why does it outperform pure vector search?

**A:** Hybrid search combines dense vector search (semantic) with sparse search (lexical, typically BM25).

- **Dense search** captures meaning: "automobile" matches "car"
- **Sparse search** captures exact terms: "Python 3.12" matches "Python 3.12", "error code E1234" matches exactly

Neither alone is sufficient:
- Dense search can miss exact technical terms, product names, or error codes
- Sparse search cannot handle paraphrasing or synonyms

Results are combined using Reciprocal Rank Fusion (RRF) or weighted scoring. In benchmarks, hybrid search consistently outperforms either method alone by 5-15%.

---

### Q6: How do you evaluate RAG quality? What metrics do you use?

**A:** RAG evaluation has two dimensions:

**Retrieval metrics** (did we find the right documents?):
- **Precision@K**: fraction of top-K results that are relevant
- **Recall@K**: fraction of all relevant docs found in top-K
- **MRR**: reciprocal rank of the first relevant result
- **NDCG**: accounts for graded relevance and position

**Generation metrics** (did the LLM produce a good answer?):
- **Faithfulness**: is the answer grounded in the retrieved context (no hallucination)?
- **Answer Relevancy**: does the answer actually address the question?
- **Context Utilization**: how much of the retrieved context was used?

**Frameworks:**
- **RAGAS**: automated evaluation framework that uses LLMs to compute these metrics
- **LLM-as-Judge**: use GPT-4o to rate answers on multiple dimensions
- **Human evaluation**: gold standard but expensive and slow

In production, I track a combination of automated metrics (RAGAS) plus user feedback (thumbs up/down).

---

### Q7: What is HyDE and when does it help?

**A:** HyDE (Hypothetical Document Embeddings) is a query transformation technique. Instead of embedding the raw user query (which is typically short), it generates a hypothetical answer using the LLM, then embeds that hypothetical answer for retrieval.

**Why it helps:** The hypothetical answer is longer and uses vocabulary/structure similar to real documents, making it a better match in embedding space.

**Example:**
- Query: "What causes rain?" (short, vague)
- HyDE generates: "Rain is caused by the water cycle. Water evaporates from oceans and lakes due to solar heat..." (document-like)
- The hypothetical answer's embedding is closer to real documents about the water cycle

**When it helps:** Short or vague queries, queries using different vocabulary than source docs.
**When it hurts:** When the LLM generates a hypothetical answer about the wrong topic, leading retrieval astray. Also adds latency (one extra LLM call).

---

### Q8: How do you handle large documents that exceed context windows?

**A:** Several strategies:

1. **Chunking with overlap** -- split into manageable pieces, embed and retrieve only relevant chunks
2. **Parent document retriever** -- embed small chunks for precise matching, return larger parent chunks for context
3. **Map-reduce** -- process each chunk independently, then combine summaries
4. **Hierarchical indexing** -- create summaries at document/section level, then drill down to paragraphs
5. **Contextual compression** -- after retrieval, compress chunks to keep only query-relevant parts
6. **Iterative retrieval** -- start with a summary, ask follow-up questions, retrieve more specific chunks

For very long documents (100+ pages), I typically use hierarchical indexing: first retrieve relevant sections via section summaries, then retrieve specific chunks within those sections.

---

### Q9: Explain the HNSW algorithm at a high level. Why is it the most popular ANN algorithm?

**A:** HNSW (Hierarchical Navigable Small World) builds a multi-layer graph:

- **Bottom layer**: all vectors, each connected to its nearest neighbors
- **Upper layers**: progressively fewer nodes with longer-range connections (like highway shortcuts)

**Search process:**
1. Start at the topmost layer (few nodes, long-range connections)
2. Greedily navigate to the nearest node to the query
3. Drop to the next layer (more nodes, shorter connections)
4. Repeat until the bottom layer
5. Return the nearest nodes found

**Why it is popular:**
- **High recall** (95-99%+) with fast queries (O(log n))
- **No training** needed (unlike IVF which needs clustering)
- **Supports dynamic inserts** without rebuilding the index
- **Simple to tune** (just M and ef parameters)

**Trade-offs:** Higher memory usage than IVF+PQ (stores the graph in RAM). Not ideal for billion-scale on a single machine.

---

### Q10: What is the parent document retriever pattern?

**A:** Store documents at two granularities:
- **Small child chunks** (200-300 tokens): embedded and indexed for precise matching
- **Large parent chunks** (1000-2000 tokens): stored in a separate docstore

When a query matches a small child chunk, the system returns the larger parent chunk to the LLM. This gives you the best of both worlds:
- **Precise retrieval** (small chunks match specific queries well)
- **Rich context** (large chunks give the LLM enough information to generate good answers)

Each child chunk stores a reference to its parent, enabling the lookup.

---

### Q11: How do you handle document updates in a RAG system?

**A:** Incremental indexing strategy:

1. **Content hashing** -- compute SHA-256 of each document; only re-index if the hash changes
2. **Delete old vectors** -- when a document changes, delete all its vectors from the store (filter by source metadata)
3. **Re-chunk and re-embed** -- process the new version and insert fresh vectors
4. **Document registry** -- maintain a database tracking each document's hash, last indexed timestamp, and chunk IDs
5. **Versioning** -- optionally keep old versions searchable with a version filter

For real-time systems, use a change data capture (CDC) pipeline or file watcher to trigger re-indexing automatically.

---

### Q12: What is Graph RAG and when would you use it?

**A:** Graph RAG combines knowledge graphs with traditional vector retrieval. An LLM extracts entities and relationships from documents to build a knowledge graph (e.g., in Neo4j). At query time, relevant subgraphs are retrieved alongside vector search results.

**When to use Graph RAG:**
- Questions about relationships between entities ("How are X and Y connected?")
- Multi-hop reasoning that spans multiple documents
- Domains with rich entity relationships (medicine, law, organizational data)
- When traditional RAG misses cross-document connections

**Trade-offs:** More complex to build and maintain. Graph extraction is imperfect. Best combined with vector search (not as a replacement).

---

### Q13: How do you choose embedding model dimensions?

**A:** Consider this trade-off:

| Dimension | Storage | Search Speed | Quality |
|-----------|---------|-------------|---------|
| 256 | Very low | Very fast | Good (80-90% of max) |
| 512 | Low | Fast | Very good |
| 768-1024 | Medium | Medium | Excellent |
| 1536-3072 | High | Slower | Maximum |

**Decision framework:**
- **Prototype / small scale**: use full dimensions (1536 for OpenAI, 768 for open-source)
- **Large scale (10M+ vectors)**: use Matryoshka embeddings truncated to 256-512 dims
- **Latency-sensitive**: lower dimensions = faster distance computation
- **Always benchmark** on your specific data -- sometimes 256 dims capture 95% of the quality

---

### Q14: What is Corrective RAG (CRAG)?

**A:** CRAG adds a quality assessment step after retrieval. An evaluator (LLM or classifier) scores the retrieved documents:

- **Correct** (high confidence): use retrieved documents as-is
- **Ambiguous** (medium confidence): supplement with web search or additional retrieval
- **Incorrect** (low confidence): discard retrieved docs, fall back to web search

This prevents the LLM from generating answers based on irrelevant retrieved context, which is a common failure mode of naive RAG.

---

### Q15: What is Self-RAG?

**A:** Self-RAG trains or prompts the model to self-reflect at each step:

1. **Should I retrieve?** -- not all queries need retrieval (simple/conversational ones)
2. **Is this document relevant?** -- discard irrelevant retrieved docs
3. **Is my answer supported?** -- check if the generated answer is grounded in the context
4. **Is my answer useful?** -- evaluate if the answer actually addresses the question

This makes the RAG system adaptive: it retrieves only when needed and validates its own outputs. It is more token-expensive but produces higher quality answers.

---

### Q16: How does BM25 work and why is it still relevant?

**A:** BM25 (Best Matching 25) is a probabilistic ranking function based on term frequency and inverse document frequency:

```
BM25(Q, D) = sum over terms t in Q:
  IDF(t) * (f(t,D) * (k1 + 1)) / (f(t,D) + k1 * (1 - b + b * |D|/avgdl))

Where:
  f(t,D)  = term frequency of t in document D
  IDF(t)  = inverse document frequency (rare terms score higher)
  |D|     = document length
  avgdl   = average document length
  k1, b   = tuning parameters (typically k1=1.2, b=0.75)
```

**Why still relevant:** Dense embeddings struggle with exact matches (product names, error codes, acronyms). BM25 excels at exact lexical matching. Combined in hybrid search, they complement each other perfectly.

---

### Q17: How do you handle multi-tenancy in a RAG system?

**A:** Three approaches:

1. **Metadata filtering** (most common): Store a `tenant_id` field in each vector's metadata. Filter on `tenant_id` at query time. Supported by all vector DBs.

2. **Separate collections/namespaces**: Each tenant gets their own collection. Better isolation but more operational overhead.

3. **Separate instances**: Each tenant gets their own vector DB instance. Maximum isolation and security, highest cost.

For most cases, metadata filtering is sufficient and scalable. Use separate collections for strict data isolation requirements (healthcare, finance).

---

### Q18: What is Agentic RAG?

**A:** Agentic RAG uses an LLM agent that decides how to retrieve information rather than following a fixed pipeline:

- The agent can choose **which retriever** to call (auth docs vs API docs vs DB docs)
- It can decide **whether to retrieve** at all
- It can perform **multi-step retrieval** (retrieve, reason, retrieve more)
- It can **combine multiple sources** dynamically

This is more flexible than a fixed pipeline but harder to debug, more expensive (multiple LLM calls), and less predictable. Best for complex queries that span multiple knowledge domains.

---

### Q19: What are the key differences between Qdrant and pgvector?

**A:**

| Aspect | Qdrant | pgvector |
|--------|--------|----------|
| **Architecture** | Purpose-built vector DB (Rust) | PostgreSQL extension |
| **Performance** | Optimized for vector ops | Good but not specialized |
| **Hybrid search** | Native (dense + sparse) | Requires workaround (pg_trgm) |
| **Filtering** | Rich payload filtering (pre-filter) | SQL WHERE (post-filter with workarounds) |
| **Scale** | Billions of vectors, distributed | Millions (single node) |
| **Ops overhead** | New service to manage | Already have PostgreSQL |
| **Quantization** | Built-in (scalar, product) | Limited |

**Choose pgvector** if you already run PostgreSQL, have < 5M vectors, and want minimal infrastructure.
**Choose Qdrant** for production RAG at scale with hybrid search and rich filtering.

---

### Q20: How do you optimize RAG latency?

**A:** Several techniques at each stage:

**Retrieval:**
- Use HNSW with tuned `ef_search` (lower = faster, less recall)
- Reduce vector dimensions (Matryoshka to 256-512)
- Pre-filter by metadata to reduce search space
- Cache frequent query embeddings

**Reranking:**
- Use a small cross-encoder (MiniLM-L-6 instead of L-12)
- Limit reranking to top-20 (not top-100)
- Use ColBERT (pre-computed token embeddings) instead of cross-encoder

**Generation:**
- Stream LLM responses (time to first token matters)
- Use contextual compression to reduce prompt size
- Cache complete responses for repeated queries
- Use a faster model (GPT-4o-mini vs GPT-4o) when quality permits

**Infrastructure:**
- Keep vector DB in-memory
- Co-locate services to minimize network latency
- Use async/parallel processing where possible

---

### Q21: What is Reciprocal Rank Fusion (RRF) and why is it used?

**A:** RRF is a method for combining ranked results from multiple retrievers without needing to normalize their scores (which can be on completely different scales).

Formula: `RRF_score(doc) = sum(1 / (k + rank_i))` where k=60 is a constant and rank_i is the document's rank in retriever i.

**Why RRF:**
- Score-agnostic: works regardless of how different retrievers score documents
- Simple to implement
- Robust: performs well across many benchmarks
- Standard in hybrid search (combining BM25 ranks with vector search ranks)

---

### Q22: How do you handle tables and structured data in RAG?

**A:**

1. **Extract tables** using pdfplumber or unstructured
2. **Convert to text**: either markdown table format or natural language description
3. **Store table metadata**: column names, row count, source page
4. **Consider separate table index**: some queries are best answered by table lookup rather than text search

For complex tables, generate a natural language summary: "This table shows quarterly revenue from 2020-2024, with columns for quarter, revenue, expenses, and profit. Total 2024 revenue was $12.3M."

This summary embeds much better than raw table data.

---

### Q23: What are Matryoshka embeddings and why do they matter?

**A:** Matryoshka (nested doll) embeddings are trained so that the first N dimensions form a valid, useful embedding on their own. You can truncate the vector from 1024 dims to 256 dims and still get ~90-95% of the quality.

**Why they matter:**
- **4x less storage**: 256 floats vs 1024 floats per vector
- **4x faster search**: fewer dimensions = faster distance computation
- **Flexible trade-off**: choose quality vs efficiency at query time
- **No retraining**: just truncate the vectors

Supported by: OpenAI (`dimensions` parameter), nomic-embed, and several open-source models trained with MRL (Matryoshka Representation Learning).

---

### Q24: What is contextual retrieval (Anthropic's approach)?

**A:** Anthropic's contextual retrieval prepends each chunk with a short context that situates it within the full document. Before embedding, an LLM generates a brief explanation of what the chunk is about in the context of the whole document.

**Example:**
- Original chunk: "The company increased by 15% year over year."
- With context: "This chunk is from the Q3 2024 earnings report, specifically the revenue section. The company increased by 15% year over year."

The context-enriched chunk embeds much better because the embedding now captures that "the company" refers to the subject of the earnings report and "15%" refers to revenue growth. This reduces the out-of-context retrieval failures that plague naive chunking.

---

### Q25: Design a RAG system for a company with 10,000 internal documents. Walk through your architecture decisions.

**A:**

**Requirements analysis:**
- 10,000 docs = ~1-5M chunks (assuming avg 100-500 chunks per doc)
- Multi-tenant (departments), access control needed
- Mix of PDFs, DOCX, and web content
- Need to stay current (daily document updates)

**Architecture:**

1. **Document processing:** unstructured library for parsing all formats; recursive character splitter at 512 tokens with 50-token overlap; store metadata (source, department, author, date, permissions)

2. **Embedding:** OpenAI `text-embedding-3-small` at 1536 dims (or 512 with Matryoshka for cost savings); batch processing during ingestion

3. **Vector store:** Qdrant (production-grade, hybrid search, rich filtering); metadata filtering for department-based access control

4. **Retrieval:** Hybrid search (dense + BM25 via Qdrant sparse vectors); retrieve top-50, rerank with Cohere rerank-v3 to top-5

5. **Generation:** GPT-4o with retrieved context; system prompt enforces citation of sources; streaming responses

6. **Infrastructure:** Redis for caching frequent queries; PostgreSQL for document registry and user management; S3 for raw document storage; incremental indexing via content hashing

7. **Monitoring:** RAGAS evaluation on a weekly test set; user feedback (thumbs up/down); retrieval quality dashboard (avg reranker scores, empty retrieval rate)

8. **Iteration plan:** Start with this setup, evaluate with RAGAS, then add query transformation (HyDE or multi-query) if retrieval quality needs improvement.
