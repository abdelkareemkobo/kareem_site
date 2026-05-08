---
title: "tiny-gte: Efficient Transformer for Semantic Search"
description: "tiny-gte is a 45MB distilled transformer from gte-small that produces 384D embeddings for efficient semantic search and vector DBs."
date: 2023-10-21
draft: false
tags:
  - nlp
  - transformers
  - embedding
  - txtai
  - vector_db
  - architecture
  - Idea_Forge
  - publish
authors:
  - kareem
image: images/til.jpg
---

## What is tiny-gte?

The `tiny-gte` model is a specialized [sentence-transformers](https://www.sbert.net/) model designed for extreme efficiency without sacrificing too much accuracy. Distilled from the `thenlper/gte-small` parent, it’s only about 45 MB in size while still outputting 384‑dimensional vectors — a compact footprint that answers questions about **gte-small model size mb** and **gte-small embedding dimension 384**. It maps sentences and paragraphs into this dense vector space, making it perfect for tasks like clustering, semantic search, and Retrieval-Augmented Generation (RAG).

The model is a distilled version of `thenlper/gte-small`. Through distillation, it manages to maintain comparable performance (only slightly lower on benchmarks like MTEB) while being roughly half the size of its parent model.

## Model Details

If you are building production-grade AI systems, the size and latency of your embedding model matter. Here is why `tiny-gte` stands out:

- **Ultra-Small Footprint:** It weighs in at around **~45MB**. To put that in perspective, the popular `all-MiniLM-L6-v2` is nearly double the size at ~80MB.
- **Dimensionality:** It produces **384D** embeddings, which is the "sweet spot" for many vector databases, balancing search precision with storage costs.
- **Architecture:** Based on the BERT architecture but optimized through distillation.
- **Parent Model:** Distilled from `thenlper/gte-small`, inheriting its robust understanding of semantic relationships.

## Why Size Matters: The Benefits of Small Models

In the world of LLMs, we often hear that "bigger is better." However, for embedding models used in search pipelines, smaller models offer several critical advantages:

1. **Lower Latency:** Smaller models require fewer FLOPs, meaning faster inference times. This is crucial for real-time search applications.
2. **Reduced Hosting Costs:** You can run `tiny-gte` on cheaper hardware, even on CPU-only instances, without a significant performance bottleneck.
3. **Edge Deployment:** At 45MB, this model can easily be deployed on mobile devices or in browser-based applications using Transformers.js.
4. **Memory Efficiency:** You can fit more instances of the model in memory, allowing for higher throughput in multi-tenant systems.

## Use Cases for tiny-gte

- **Real-time Document Retrieval:** Quickly finding relevant context for an LLM prompt in a RAG pipeline.
- **Mobile AI Applications:** Enabling semantic search within offline mobile apps where storage space is limited.
- **Large-scale Clustering:** Processing millions of documents where the computational cost of larger models would be prohibitive.
- **Edge Search:** Using `tiny-gte` with libraries like `txtai` or `fastembed` for local, private search on your own machine.

## Performance Comparison: tiny-gte vs. gte-small

On the Massive Text Embedding Benchmark (MTEB), `tiny-gte` performs impressively well given its size. While `gte-small` might lead by a few percentage points in specific retrieval tasks, `tiny-gte` often provides better "value per megabyte." If your application can tolerate a 1-2% drop in accuracy in exchange for 2x faster inference, `tiny-gte` is the clear winner.

## References

- [TaylorAI/gte-tiny on Hugging Face](https://huggingface.co/TaylorAI/gte-tiny)
- [MTEB (Massive Text Embedding Benchmark) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Prithiviraj Damodaran's Note on distilled models](https://www.linkedin.com/posts/prithivirajdamodaran_%3F%3F%3F%3F-%3F%3F%3F%3F-%3F%3F%3F%3F%3F-%3F%3F%3F%3F-activity-7120279840569597952-iwc-)

---

### Internal Resources

If you're looking for more technical deep dives or information on my research, check out these sections:

- [My Research Papers]((../../../papers.html)
- [Open Source Contributions]((../../../oss/opensource.html)
- [Today I Learned: AI Engineering Notes]((../../../til/index.html)
- [Arabic NLP Blog Posts]((../../feed.html)
