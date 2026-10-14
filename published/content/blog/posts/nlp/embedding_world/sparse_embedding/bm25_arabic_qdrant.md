---
section: blog
series: embedding
title: BM25 Explained Part 2 | Qdrant hybrid search for real estate
description: >-
  Step-by-step guide to building a bilingual hybrid search system using Qdrant
  and BM25. Learn how to optimize real estate search with n-grams and RRF
  fusion.
lang: en
dir: ltr
author: kareem
publish: true
date: '2025-12-19'
tags:
  - blogging
  - embedding
  - qdrant
  - sparse
socialImage: /assets/4f6c3d8a4dd655aaaf454094.webp
---

## Building Hybrid Search for Real Estate with Qdrant and BM25

<img src="/assets/4f6c3d8a4dd655aaaf454094.webp" alt="xbites mena real esate ai" class="img-fluid figure-img">

### The Problem

When building a real estate search system, I ran into a frustrating issue: dense vector search kept returning the wrong locations. A query for "6th Settlement Apartment" would return properties in "5th Settlement" or "New Cairo" instead.

I was using Gemini's embedding model, which performs well for both English and Arabic. But even strong embeddings struggle with domain-specific data where names are nearly identical:

- El Patio Vera
- El Patio Solo
- El Patio Casa

These project names fall outside the embedding model's training distribution. 

The semantic similarity between them is too high for dense search to distinguish.

My property chunks contain full descriptions like "Villa in El Patio Vera, 3 bedrooms, 1,000,000 EGP". Semantic search handles most queries well, but some searches are fundamentally lexical. I needed both approaches working together.

---

## When Dense Search Fails

My dataset contained 12,877 real estate properties across Egypt with location names like:

- 6th Settlement, 5th Settlement
- 6th of October, 5th of October
- Sheikh Zayed, New Zayed
- El Alamein, North Coast

When a user searched for "6th Settlement Apartment less than 10 million", the dense vector results looked like this:

```
5th Settlement, New Cairo - 9.5M
5th Settlement, New Cairo - 9.3M
El Alamein, North Coast - 5.6M
5th Settlement, New Cairo - 9.4M
El Alamein, North Coast - 5.7M
```

Zero results from 6th Settlement. The embedding model treated "6th" and "5th" as semantically similar because they are both ordinal numbers. The word "Settlement" matched in both cases, pushing the semantic similarity even higher. This is exactly where lexical matching would help.

---

## Why Hybrid Search Works

Dense vectors and BM25 have complementary strengths.

Dense vectors understand meaning. They know that "apartment", "flat", and "unit" refer to similar things. They handle typos and variations gracefully. But they struggle with exact matches where surface-level differences matter, like distinguishing "6th" from "5th".

BM25 sparse vectors match tokens exactly. The token "6th" will never match "5th". This works across languages without additional configuration. The downside is that BM25 has no semantic understanding. It cannot recognize that "apartment" and "flat" mean the same thing.

By combining both approaches, you get semantic understanding when you need it and exact matching when that matters more.

The architecture looks like this:

```
User Query: "6th Settlement Apartment"
                    |
         +----------+----------+
         |                     |
   Dense Search          BM25 Search
   (100 candidates)      (30 candidates)
         |                     |
         +----------+----------+
                    |
              RRF Fusion
                    |
             Final Results
```

Reciprocal Rank Fusion combines the rankings from both approaches, giving weight to results that appear highly ranked in either or both lists.

---

## Challenge 1: Reducing Token Noise

The first problem I encountered was tokenization noise. Raw property chunks contained everything: URLs, metadata IDs, field names, and numeric values.

A typical chunk looked like:

```
"Palm Hills, Palm Hills New Cairo, Apartment, 15.3M EGP,
154 sqm, metadata_id_123, https://example.com/file.pdf, ..."
```

Tokenizing this produced over 13,000 unique tokens across the corpus. Most of these were useless for search: URL fragments, random IDs, and field names that would never appear in user queries.

The solution was to extract only the fields that users would actually search for:

- Developer name
- Project name
- Unit type
- Location and sublocation

After filtering, a chunk became:

```
"Palm Hills Palm Hills New Cairo Apartment 5th Settlement New Cairo"
```

This reduced the vocabulary from 13,000 tokens to 1,941, an 86% reduction. The BM25 index became faster and more accurate because every remaining token was meaningful.

---

## Challenge 2: Handling Stopwords

Common words like "of", "in", and "the" added noise to the BM25 scores. Since this was a bilingual system supporting Arabic and English, I needed stopwords for both languages:

```python
STOPWORDS = {
    # English
    'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or',
    # Arabic
    'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'التي', 'الذي'
}
```

Removing these ensured that queries like "apartment in 6th Settlement" focused on the meaningful tokens rather than matching every document containing "in".

---

## Challenge 3: N-gram Tokenization

With basic tokenization, similar location names still overlapped too much:

```
"6th Settlement" → ['6th', 'settlement']
"5th Settlement" → ['5th', 'settlement']
```

Both share the token "settlement", so BM25 would give partial credit to 5th Settlement results even when the user explicitly asked for 6th Settlement.

The solution was to add bigrams, two-word phrases joined with an underscore:

```python
def tokenize_ngram(text: str) -> List[str]:
    words = [w for w in text.split() if w not in STOPWORDS]
    tokens = words.copy()
    
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]}_{words[i+1]}")
    
    return tokens
```

Now the tokenization produces:

```
"6th Settlement" → ['6th', 'settlement', '6th_settlement']
"5th Settlement" → ['5th', 'settlement', '5th_settlement']
```

The bigram "6th_settlement" is unique and will only match documents containing that exact phrase. This dramatically improved precision for location-specific queries.

---

## Challenge 4: Data Imbalance

Even with proper tokenization, some brands dominated the results unfairly. My dataset had:

- Palm Hills: 718 properties
- El Patio: 49 properties

When searching for "El Patio apartment", BM25 would often return Palm Hills results first because the sheer frequency of Palm Hills in the corpus gave it higher term frequency scores.

The solution was to adjust the balance between dense and sparse search in the hybrid approach. By retrieving more candidates from dense search (100) and fewer from BM25 (30), I gave semantic understanding more influence in the final ranking. This allowed the dense embeddings, which correctly understood "El Patio" as a distinct entity, to override BM25's frequency bias.



## RRF (Reciprocal Rank Fusion) Explained

RRF is a method for combining rankings from multiple search systems.

c How It Works

Instead of averaging scores (which can be misleading when different systems use different scales), RRF uses **rank positions**.

**Formula:**
```
RRF_score = Σ (1 / (k + rank))
```

Where:
- `k` = constant (usually 60)
- `rank` = position in that ranking (1st, 2nd, 3rd...)

### Example

Query: "6th Settlement Apartment"

**Dense Search Rankings:**
1. Result A (6th Settlement) → score = 1/(60+1) = 0.0164
2. Result B (New Cairo) → score = 1/(60+2) = 0.0161
3. Result C (5th Settlement) → score = 1/(60+3) = 0.0159

**BM25 Rankings:**
1. Result A (6th Settlement) → score = 1/(60+1) = 0.0164
2. Result D (6th October) → score = 1/(60+2) = 0.0161
3. Result C (5th Settlement) → score = 1/(60+3) = 0.0159

**Combined RRF Scores:**
- Result A: 0.0164 + 0.0164 = **0.0328** (appears in both, ranked 1st)
- Result C: 0.0159 + 0.0159 = **0.0318** (appears in both)
- Result B: 0.0161 (only in dense)
- Result D: 0.0161 (only in BM25)

Result A wins because it ranked highly in **both** systems!

---

## Results

After implementing hybrid search with these optimizations, the same query that previously failed now worked correctly.

Query: "6th Settlement Apartment less than 10 million"

Before (Dense Only):
```
5th Settlement - 9.5M
5th Settlement - 9.3M
El Alamein - 5.6M
```

After (Hybrid with BM25):
```
6th Settlement - 8.0M
6th Settlement - 9.5M
6th Settlement - 7.2M
```

---

## Key Takeaways

1. Filter your tokens aggressively. Remove everything that users would never search for.

2. Use n-grams for multi-word entities. Bigrams turn "6th Settlement" into a unique, matchable token.

3. Handle stopwords in all supported languages. A bilingual system needs bilingual stopword lists.

4. Balance dense and sparse weights based on your data. If one brand dominates your corpus, give more weight to semantic search.

5. Hybrid search is not always necessary. If pure semantic search works for your use case, the added complexity of BM25 may not be worth it. Use hybrid when exact matches matter and when you have domain-specific vocabulary that embeddings handle poorly.



```python
# Step 1: Install dependencies
# pip install qdrant-client rank-bm25

import re
from typing import List
import pickle
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, models

# Step 2: Define tokenizer with stopwords and bigrams
STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "and",
    "or",
    "في",
    "من",
    "إلى",
    "على",
    "عن",
    "مع",
}


def tokenize_ngram(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = [w for w in text.split() if w not in STOPWORDS]
    tokens = words.copy()
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]}_{words[i + 1]}")
    return tokens
```

```python
# Step 3: Build BM25 index from your filtered chunks
filtered_chunks = ["palm hills apartment 5th settlement new cairo", ...]  # Your data
tokenized_corpus = [tokenize_ngram(chunk) for chunk in filtered_chunks]
bm25 = BM25Okapi(tokenized_corpus)

# Save for later use
with open("bm25_index.pkl", "wb") as f:
    pickle.dump(bm25, f)

# Step 4: Create hybrid collection in Qdrant
client = QdrantClient(url="your-url", api_key="your-key")

client.create_collection(
    collection_name="hybrid_collection",
    vectors_config={
        "dense": models.VectorParams(size=3072, distance=models.Distance.COSINE)
    },
    sparse_vectors_config={"text-sparse": models.SparseVectorParams()},
)
```

```python
# Step 5: Convert text to sparse vector using BM25 scores
def text_to_sparse_vector(text: str, bm25_index) -> models.SparseVector:
    tokens = tokenize_ngram(text)
    scores = bm25_index.get_scores(tokens)

    indices = []
    values = []
    for idx, score in enumerate(scores):
        if score > 0:
            indices.append(idx)
            values.append(float(score))

    return models.SparseVector(indices=indices, values=values)


# Step 6: Upload documents with both vectors
for i, (chunk, dense_embedding, metadata) in enumerate(
    zip(filtered_chunks, embeddings, metadata_list)
):
    client.upsert(
        collection_name="hybrid_collection",
        points=[
            models.PointStruct(
                id=i,
                payload=metadata,
                vector={
                    "dense": dense_embedding,
                    "text-sparse": text_to_sparse_vector(chunk, bm25),
                },
            )
        ],
    )
```

```python
# Step 7: Hybrid search with RRF fusion
def hybrid_search(query_text: str, query_dense_vector: list, limit: int = 10):
    query_sparse = text_to_sparse_vector(query_text, bm25)

    results = client.query_points(
        collection_name="hybrid_collection",
        prefetch=[
            models.Prefetch(query=query_dense_vector, using="dense", limit=100),
            models.Prefetch(query=query_sparse, using="text-sparse", limit=30),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    )
    return results.points


# Usage
results = hybrid_search("6th Settlement Apartment", your_query_embedding)
```

### References

1. [Xbites Real Estate AI](https://www.xbites.io/) 
2. [Qdrand Hybrid search](https://qdrant.tech/articles/hybrid-search/)
3. [Bm25 part 1](/blog/posts/nlp/embedding_world/sparse_embedding/bm25_from_scratch.md)
4. [Bm25 part 3](/blog/posts/nlp/embedding_world/sparse_embedding/bm25_benchmark_full.md)



### Related Posts

- [Diversifying Search Results with Pyversity and Qdrant](/blog/posts/minishlab/pyversity_qdrant.md) — Using Qdrant for search diversification in RAG
