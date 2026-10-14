---
section: blog
series: embedding
title: 'Late Interaction & ColPali: Efficient Semantic Search'
description: >-
  Explore Late Interaction models like ColBERT and ColPali for efficient
  token-level semantic search balancing speed and accuracy.
lang: en
dir: ltr
author: kareem
publish: true
date: '2025-05-15'
tags:
  - blogging
  - embedding
  - minishlab
  - model2vec
  - arabic
socialImage: /assets/9cac6b407580862f4df87f77.webp
---

## Beyond Bi-Encoders: The Rise of Late Interaction

In the world of Information Retrieval (IR), we usually face a trade-off between speed and accuracy. 

### 1. Bi-Encoders (The Speed Kings)
Bi-encoders (like standard BERT embeddings) encode the query and the document independently into a single vector. Search is just a cosine similarity between these two points. It's incredibly fast (sub-millisecond) but loses fine-grained details because the entire document is compressed into one fixed-size vector.

### 2. Cross-Encoders (The Accuracy Masters)
Cross-encoders feed both the query and the document into the model simultaneously (Early Interaction). The model can attend to every word in the query relative to every word in the document. This is highly accurate but computationally expensive because you must run the model for every single query-document pair. You can't pre-compute embeddings.

### 3. Late Interaction: The Best of Both Worlds
Late Interaction models, pioneered by **ColBERT**, bridge this gap. Instead of one vector per document, they store a vector for **every single token** in the document. 

When a query comes in:
1. The query is encoded into token-level embeddings.
2. A **MaxSim** (Maximum Similarity) operation is performed: for each query token, we find the document token that matches it best.
3. We sum these maximum similarities to get the final score.

This allows the model to perform fine-grained matching (like a cross-encoder) while still allowing document embeddings to be pre-computed (like a bi-encoder).

### ColPali: Retrieval Without OCR

One of the most exciting recent developments is **ColPali**. Traditional PDF retrieval requires a complex pipeline: OCR the text, chunk it, and then embed it. This often fails on tables, charts, and complex layouts.

ColPali applies the Late Interaction principle to vision models (PaliGemma). It treats image patches of a PDF page as "tokens." Instead of reading text, it "looks" at the page and matches query tokens directly to visual features. 

**Key Benefits of ColPali:**
- **Layout Aware:** It understands that a caption belongs to a specific image.
- **OCR-Free:** No more messy text extraction from scanned documents.
- **Superior Retrieval:** It outperforms traditional text-based RAG on visually rich documents.

---

## Ecosystem and Tools

If you want to implement Late Interaction today, these are the projects to watch:
- **ColBERTv2:** The optimized version of the original late interaction model.
- **PyLate:** A flexible Python library for training and using late interaction models.
- **PLAID:** An extremely fast engine for searching ColBERT vectors.
- **Model2Vec:** While focused on static embeddings, it shows the trend towards more efficient representation learning.

Late interaction is transforming how we think about retrieval, moving us away from "one vector fits all" towards a more nuanced, token-aware future.

---

### Internal Resources

If you're interested in more technical deep dives or information on my research, check out these sections:

- [My Research Papers](/papers.md)
- [Open Source Contributions](https://github.com/abdelkareemkobo?tab=repositories)
- [Today I Learned: AI Engineering Notes](/til/index.md)
- [Arabic NLP Blog Posts](/blog/feed.md)

## Practical Example: Using ColBERT with PyLate

Here's how to use a late interaction model for retrieval in Python:

```python
from pylate import ColBERT, Indexes, retrieve

# Load a pre-trained ColBERT model
model = ColBERT("lightonai/colbertv2")

# Encode documents into token-level embeddings
documents = [
    "Late interaction models store a vector per token instead of one per document.",
    "Bi-encoders compress documents into a single vector for fast retrieval.",
    "Cross-encoders process query and document together for higher accuracy."
]

document_embeddings = model.encode(documents, convert_to_tensor=True)

# Build an index for fast retrieval
index = Indexes.FlatIndex()
index.add_documents(document_embeddings, documents=documents)

# Encode the query
query = "What is the difference between bi-encoders and cross-encoders?"
query_embedding = model.encode([query], convert_to_tensor=True)

# Retrieve top-k results
results = index.search(query_embedding, k=3)
for doc, score in results:
    print(f"{score:.3f}: {doc}")
```

### Performance Comparison

| Model Type | Speed | Accuracy | Use Case |
|---|---|---|---|
| Bi-Encoder | Fastest (~1ms/query) | Good | Initial retrieval, large-scale search |
| Cross-Encoder | Slowest (~100ms/pair) | Best | Re-ranking top results |
| Late Interaction | Fast (~10ms/query) | Very Good | High-accuracy retrieval at scale |

Late interaction gives you 90% of cross-encoder accuracy with 10% of the computational cost.

## FAQ: Late Interaction Models

### What is late interaction in NLP?
Late interaction is a retrieval technique where query and document tokens interact at query time rather than during encoding. Models like ColBERT store token-level embeddings for documents and compute similarity using MaxSim operations when a query arrives.

### Is ColBERT better than standard embeddings?
For retrieval tasks requiring fine-grained matching, yes. ColBERT outperforms bi-encoders on most benchmarks while being much faster than cross-encoders. However, it requires more storage since you store one vector per token instead of one per document.

### What is ColPali?
ColPali applies late interaction to vision-language models for document retrieval. Instead of extracting text from PDFs via OCR, it processes page images directly and matches query tokens to visual patches. This handles tables, charts, and complex layouts better than text-based retrieval.

### How much storage do late interaction models need?
Storage is higher than bi-encoders because you store vectors for every token. A 512-token document needs 512 vectors instead of 1. Compression techniques like PLAID indexing and quantization reduce this overhead significantly.

### When should I use late interaction vs. bi-encoders?
Use late interaction when retrieval quality is critical and you can afford the extra storage. Use bi-encoders for very large-scale search where speed and storage efficiency matter most. A common hybrid approach: bi-encoder for initial retrieval, late interaction for re-ranking.

## Related Posts

- [BM25 Benchmarking](/blog/posts/nlp/embedding_world/sparse_embedding/bm25_benchmark_full.md) — Comparing sparse retrieval methods
- [MTEB Benchmark Explained](/blog/posts/mteb_encoding/mteb_massive_text_embedding_benchmark.md) — How embedding models are evaluated
- [Tiny-GTE Transformer Model](/blog/posts/mteb_encoding/tiny-gte_transformer_model.md) — Efficient transformer architecture for embeddings
- [HyperRun + ColGrep](/blog/posts/nlp/embedding_world/hyperrun.md) — Self-hosted AI docs search using late interaction
