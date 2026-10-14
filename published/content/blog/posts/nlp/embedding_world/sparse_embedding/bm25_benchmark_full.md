---
section: blog
series: embedding
title: 'BM25 Part 3: Benchmarking Python BM25 Libraries'
description: >-
  Benchmarking BM25 implementations: bm25-sparse, rankbm25, numpy vectorization,
  scipy, and plain Python. Find the fastest BM25 for your search use case.
lang: en
dir: ltr
author: kareem
publish: true
date: '2025-12-20'
tags:
  - blogging
  - embedding
  - qdrant
socialImage: /assets/3ab4fcaf08eb27a74f281da0.webp
---

```python
import marimo as mo
```

```python
from collections import Counter
from math import log
from scipy.sparse import csr_matrix
import numpy as np


def tokenize(text):
    return text.lower().split()
```

### Dataset Benchmark

I will continue exploring the bm25 and how to use with a huggingface dataset with multiple implementations and compare their speed 

The methods i will use: 

1. Normal Python Calculations
2. Spicy sparse matrix with python loops
3. Spicy sparse matrix with Numpy Vectorization
4. Rank_bm25
5. bm25-sparse
6. Polars optimization

If there is any mistake please tell me!

it's first time for me to create a benchmark for anything

```python
from datasets import load_dataset
import time
import random

# Load dataset
dataset = load_dataset("rotten_tomatoes", split="train")
corpus = [item["text"] for item in dataset]
print(f"Corpus size: {len(corpus)} reviews")


def generate_random_queries(corpus, num_queries=1000):
    queries = []
    for _ in range(num_queries):
        doc = random.choice(corpus)
        words = tokenize(doc)
        num_words = random.randint(2, 4)
        if len(words) >= num_words:
            query_words = random.sample(words, num_words)
        else:
            query_words = words
        queries.append(" ".join(query_words))
    return queries


test_queries = generate_random_queries(corpus, 1000)
print(f"Generated {len(test_queries)} test queries")
```

```text
Corpus size: 8530 reviews
Generated 1000 test queries
```

```python
def benchmark_python_loops():
    print("=== 1. NORMAL PYTHON (LOOPS) ===")
    start = time.time()

    # Tokenize all documents
    all_docs_terms_py = []
    for doc in corpus:
        tokens = tokenize(doc)
        all_docs_terms_py.append(Counter(tokens))

    # Compute DF
    df_py = {}
    for doc_tf in all_docs_terms_py:
        for term in doc_tf.keys():
            df_py[term] = df_py.get(term, 0) + 1

    # Average doc length
    total_len = sum(sum(doc_tf.values()) for doc_tf in all_docs_terms_py)
    avg_doc_len_py = total_len / len(corpus)

    # IDF function
    def compute_idf_py(term, df_dict, N):
        if term not in df_dict:
            return 0
        return np.log((N - df_dict[term] + 0.5) / (df_dict[term] + 0.5))

    # BM25 function
    def bm25_score_py(term, doc_tf, idf_val, doc_len, avg_len, k1=1.5, b=0.75):
        tf = doc_tf.get(term, 0)
        if tf == 0:
            return 0
        return idf_val * tf * (k1 + 1) / (tf + k1 * (1 - b + b * (doc_len / avg_len)))

    index_time = time.time() - start

    # Query
    start_query = time.time()
    for query in test_queries:
        query_tokens = tokenize(query)
        doc_scores = [0] * len(corpus)
        for token in query_tokens:
            if token in df_py:
                idf_val = compute_idf_py(token, df_py, len(corpus))
                for doc_idx, doc_tf in enumerate(all_docs_terms_py):
                    doc_len = sum(doc_tf.values())
                    doc_scores[doc_idx] += bm25_score_py(
                        token, doc_tf, idf_val, doc_len, avg_doc_len_py
                    )
        top_k = sorted(
            range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True
        )[:10]

    query_time = time.time() - start_query
    qps = len(test_queries) / query_time

    print(f"Indexing: {index_time:.4f}s")
    print(f"Query time: {query_time:.4f}s")
    print(f"QPS: {qps:.2f}\n")

    return index_time, query_time, qps


benchmark_python_loops()
```

```text
=== 1. NORMAL PYTHON (LOOPS) ===
Indexing: 0.0801s
Query time: 23.6460s
QPS: 42.29
```

```text
(0.08005261421203613, 23.645965099334717, 42.29051323551751)
```

## Why Sparse Matrices?

Right now, to search for "python programming", you'd need to:
1. Loop through each document
2. Calculate BM25 score for "python"
3. Calculate BM25 score for "programming"
4. Add them up

That's slow for large datasets!

**The key insight:** We can pre-compute ALL BM25 scores for ALL words in ALL documents and store them in a matrix. Then searching becomes just looking up rows and adding them.

Here's what the matrix looks like conceptually:

```
               Doc0   Doc1   Doc2
python         -0.51  -0.56   0.00
programming    -1.95  -2.14  -1.79
java            0.00   0.00   2.20
```

```python
def benchmark_sparse_python_loops():
    print("=== 2. SPARSE MATRIX WITH PYTHON LOOPS ===")
    start = time.time()

    # Tokenize
    all_docs_terms_sp = []
    for doc in corpus:
        tokens = tokenize(doc)
        all_docs_terms_sp.append(Counter(tokens))

    # Get all words and create word_to_idx
    all_words_sp = list(set(word for doc in all_docs_terms_sp for word in doc))
    word_to_idx_sp = {word: idx for idx, word in enumerate(all_words_sp)}

    # Compute DF
    df_sp = {}
    for doc_tf in all_docs_terms_sp:
        for term in doc_tf.keys():
            df_sp[term] = df_sp.get(term, 0) + 1

    # Average doc length
    avg_doc_len_sp = sum(sum(doc_tf.values()) for doc_tf in all_docs_terms_sp) / len(
        corpus
    )

    # Build sparse matrix
    rows_idx = []
    cols_idx = []
    data = []

    for word_idx, word in enumerate(all_words_sp):
        idf_val = np.log((len(corpus) - df_sp[word] + 0.5) / (df_sp[word] + 0.5))
        for doc_idx, doc_tf in enumerate(all_docs_terms_sp):
            doc_len = sum(doc_tf.values())
            tf = doc_tf.get(word, 0)
            if tf != 0:
                score = (
                    idf_val
                    * tf
                    * 2.5
                    / (tf + 1.5 * (1 - 0.75 + 0.75 * (doc_len / avg_doc_len_sp)))
                )
                rows_idx.append(word_idx)
                cols_idx.append(doc_idx)
                data.append(score)

    sparse_matrix_sp = csr_matrix(
        (data, (rows_idx, cols_idx)), shape=(len(all_words_sp), len(corpus))
    )

    index_time = time.time() - start

    # Query
    start_query = time.time()
    for query in test_queries:
        query_indices = [
            word_to_idx_sp[word] for word in tokenize(query) if word in word_to_idx_sp
        ]
        if query_indices:
            doc_scores = np.array(
                sparse_matrix_sp[query_indices, :].sum(axis=0)
            ).flatten()
            top_k = np.argsort(doc_scores)[-10:][::-1]

    query_time = time.time() - start_query
    qps = len(test_queries) / query_time

    print(f"Indexing: {index_time:.4f}s")
    print(f"Query time: {query_time:.4f}s")
    print(f"QPS: {qps:.2f}\n")

    return index_time, query_time, qps


benchmark_sparse_python_loops()
```

```text
=== 2. SPARSE MATRIX WITH PYTHON LOOPS ===
Indexing: 77.9146s
Query time: 0.4709s
QPS: 2123.64
```

```text
(77.91458225250244, 0.4708900451660156, 2123.6380133019015)
```

## Performance Optimization with Numpy

My code has python loops everywhere.
Python loops are very slow.
The goal here it to use Numpy Vectorization: doing operations on entiry arrays at once instead of looping :
```
for word_1 in all_words:
for doc_idx, doc_1 in enumerate(all_docs_terms):
    idf_python_1 = compute_idf(word_1, 3, df)
    doc_len_1 = sum(all_docs_terms[doc_idx].values())
    score_1 = bm25_score(word_1, doc_1, idf_python_1, doc_len_1, avg_doc_len)
```
This has nested loops => very slow for large datasets.
Key Insight: Instead of computing one BM25 score at a time, we can compute all scores at once using matrix opertions.

```python
def benchmark_sparse_numpy():
    print("=== 3. SPARSE MATRIX WITH NUMPY VECTORIZATION ===")
    start = time.time()

    # Tokenize
    all_docs_terms_np = []
    for doc in corpus:
        tokens = tokenize(doc)
        all_docs_terms_np.append(Counter(tokens))

    all_words_np = list(set(word for doc in all_docs_terms_np for word in doc))
    word_to_idx_np = {word: idx for idx, word in enumerate(all_words_np)}

    # Build TF matrix
    rows = []
    cols = []
    tf_data = []

    for doc_idx, doc_tf in enumerate(all_docs_terms_np):
        for word, tf in doc_tf.items():
            rows.append(word_to_idx_np[word])
            cols.append(doc_idx)
            tf_data.append(tf)

    tf_matrix_np = csr_matrix(
        (tf_data, (rows, cols)), shape=(len(all_words_np), len(corpus))
    )

    # Compute DF array
    df_array_np = np.array((tf_matrix_np > 0).sum(axis=1)).flatten()

    # Document lengths
    doc_lengths_np = np.array(tf_matrix_np.sum(axis=0)).flatten()
    avg_doc_len_np = doc_lengths_np.mean()

    # Compute IDF array
    N = len(corpus)
    idf_array_np = np.log((N - df_array_np + 0.5) / (df_array_np + 0.5))

    # Vectorized BM25
    k1 = 1.5
    b = 0.75
    tf_dense = tf_matrix_np.toarray()

    numerator = tf_dense * (k1 + 1)
    length_norm = 1 - b + b * (doc_lengths_np / avg_doc_len_np)
    denominator = tf_dense + k1 * length_norm

    idf_column = idf_array_np.reshape(-1, 1)
    bm25_scores = idf_column * (numerator / (denominator + 1e-10))

    bm25_matrix_np = csr_matrix(bm25_scores)

    index_time = time.time() - start

    # Query
    start_query = time.time()
    for query in test_queries:
        query_indices = [
            word_to_idx_np[word] for word in tokenize(query) if word in word_to_idx_np
        ]
        if query_indices:
            doc_scores = np.array(
                bm25_matrix_np[query_indices, :].sum(axis=0)
            ).flatten()
            top_k = np.argsort(doc_scores)[-10:][::-1]

    query_time = time.time() - start_query
    qps = len(test_queries) / query_time

    print(f"Indexing: {index_time:.4f}s")
    print(f"Query time: {query_time:.4f}s")
    print(f"QPS: {qps:.2f}\n")

    return index_time, query_time, qps


benchmark_sparse_numpy()
```

```text
=== 3. SPARSE MATRIX WITH NUMPY VECTORIZATION ===
Indexing: 6.9885s
Query time: 0.5250s
QPS: 1904.79
```

```text
(6.988537311553955, 0.5249927043914795, 1904.7883744577036)
```

```python
def benchmark_rank_bm25():
    from rank_bm25 import BM25Okapi

    print("=== 4. RANK-BM25 ===")

    tokenized_corpus_rank = [tokenize(doc) for doc in corpus]

    start = time.time()
    bm25_rank = BM25Okapi(tokenized_corpus_rank)
    index_time = time.time() - start

    start_query = time.time()
    for query in test_queries:
        query_tokens = tokenize(query)
        scores = bm25_rank.get_scores(query_tokens)
        top_k = np.argsort(scores)[-10:][::-1]

    query_time = time.time() - start_query
    qps = len(test_queries) / query_time

    print(f"Indexing: {index_time:.4f}s")
    print(f"Query time: {query_time:.4f}s")
    print(f"QPS: {qps:.2f}\n")

    return index_time, query_time, qps


benchmark_rank_bm25()
```

```text
=== 4. RANK-BM25 ===
Indexing: 0.0813s
Query time: 5.0695s
QPS: 197.26
```

```text
(0.08132147789001465, 5.069519281387329, 197.2573619892297)
```

```python
def benchmark_bm25s():
    import bm25s

    print("=== 5. BM25S ===")

    start = time.time()
    retriever_bm25s = bm25s.BM25()
    corpus_tokens_bm25s = bm25s.tokenize(corpus, show_progress=False)
    retriever_bm25s.index(corpus_tokens_bm25s, show_progress=False)
    index_time = time.time() - start

    start_query = time.time()
    for query in test_queries:
        query_tokens_bm25s = bm25s.tokenize(query, show_progress=False)
        results, scores = retriever_bm25s.retrieve(
            query_tokens_bm25s, k=10, show_progress=False
        )

    query_time = time.time() - start_query
    qps = len(test_queries) / query_time

    print(f"Indexing: {index_time:.4f}s")
    print(f"Query time: {query_time:.4f}s")
    print(f"QPS: {qps:.2f}\n")

    return index_time, query_time, qps


benchmark_bm25s()
```

```text
=== 5. BM25S ===
Indexing: 0.5329s
Query time: 0.7049s
QPS: 1418.74
```

```text
(0.5328564643859863, 0.7048521041870117, 1418.7373408687158)
```

```python
def benchmark_polars():
    import polars as pl

    print("=== 6. POLARS OPTIMIZATION ===")
    start = time.time()

    k1 = 1.5
    b = 0.75
    N = len(corpus)

    # Create LazyFrame
    lf = pl.LazyFrame({"doc_id": range(N), "text": corpus})

    # Tokenize and explode
    lf_tokens = (
        lf.with_columns(
            pl.col("text").str.to_lowercase().str.split(" ").alias("tokens")
        )
        .explode("tokens")
        .filter(pl.col("tokens").str.len_chars() > 0)
    )

    # Build vocab
    vocab_df = (
        lf_tokens.select(pl.col("tokens").unique()).collect().with_row_index("word_idx")
    )
    vocab_size = len(vocab_df)

    # Join to map tokens to indices
    lf_tokens = lf_tokens.join(vocab_df.lazy(), on="tokens", how="left")

    # Compute TF
    lf_tf = lf_tokens.group_by(["doc_id", "word_idx"]).agg(pl.len().alias("tf"))

    # Compute document lengths
    lf_doc_lens = lf_tf.group_by("doc_id").agg(pl.col("tf").sum().alias("doc_len"))
    avg_doc_len_pl = lf_doc_lens.select(pl.col("doc_len").mean()).collect().item()

    # Compute DF and IDF
    lf_idf = (
        lf_tf.group_by("word_idx")
        .agg(pl.col("doc_id").n_unique().alias("df"))
        .with_columns(
            ((pl.lit(N) - pl.col("df") + 0.5) / (pl.col("df") + 0.5)).log().alias("idf")
        )
    )

    # Calculate BM25
    df_bm25_pl = (
        lf_tf.join(lf_doc_lens, on="doc_id", how="left")
        .join(lf_idf, on="word_idx", how="left")
        .with_columns(
            (
                pl.col("idf")
                * pl.col("tf")
                * (k1 + 1)
                / (
                    pl.col("tf")
                    + k1 * (1 - b + b * (pl.col("doc_len") / avg_doc_len_pl))
                )
            ).alias("bm25_score")
        )
        .select(["doc_id", "word_idx", "bm25_score"])
        .collect()
    )

    # Build sparse matrix
    rows_pl = df_bm25_pl["word_idx"].to_numpy()
    cols_pl = df_bm25_pl["doc_id"].to_numpy()
    data_pl = df_bm25_pl["bm25_score"].to_numpy()
    word_to_idx_pl = dict(zip(vocab_df["tokens"], vocab_df["word_idx"]))
    bm25_matrix_pl = csr_matrix((data_pl, (rows_pl, cols_pl)), shape=(vocab_size, N))

    index_time = time.time() - start

    # Query
    start_query = time.time()
    for query in test_queries:
        query_indices = [
            word_to_idx_pl[word] for word in tokenize(query) if word in word_to_idx_pl
        ]
        if query_indices:
            doc_scores = np.array(
                bm25_matrix_pl[query_indices, :].sum(axis=0)
            ).flatten()
            top_k = np.argsort(doc_scores)[-10:][::-1]

    query_time = time.time() - start_query
    qps = len(test_queries) / query_time

    print(f"Indexing: {index_time:.4f}s")
    print(f"Query time: {query_time:.4f}s")
    print(f"QPS: {qps:.2f}\n")

    return index_time, query_time, qps


benchmark_polars()
```

```text
=== 6. POLARS OPTIMIZATION ===
Indexing: 0.1580s
Query time: 0.6143s
QPS: 1627.94
```

```text
(0.1579906940460205, 0.6142714023590088, 1627.9449053946898)
```

## 🏆 BM25 Benchmark Results (8,530 documents, 1000 queries)

| Implementation | Indexing (s) | Query (s) | QPS |
|----------------|-------------|-----------|-----|
| **1. Python Loops** | 0.0801 | 23.6460 | 42.29 |
| **2. Sparse + Python Loops** | 77.9146 | 0.4709 | 2,123.64 |
| **3. Sparse + NumPy** | 6.9885 | 0.5250 | 1,904.79 |
| **4. rank-bm25** | 0.0813 | 5.0695 | 197.26 |
| **5. bm25s** | 0.5329 | 0.7049 | 1,418.74 |
| **6. Polars** | 0.1580 | 0.6143 | 1,627.94 |
---

**Key Findings:**


| Category | Winner | Value |
|----------|--------|-------|
| **Fastest Indexing** | Python Loops | 0.0801s |
| **Fastest Query** | Sparse + Python Loops | 0.4709s |
| **Highest QPS** | Sparse + Python Loops | 2,123.64 |
| **Best Balance** | **Polars** | 0.1580s index, 1,627.94 QPS |

---

### References

1. [bm25 part 1](/blog/posts/nlp/embedding_world/sparse_embedding/bm25_from_scratch.md)
2. [bm25 part 2](/blog/posts/nlp/embedding_world/sparse_embedding/bm25_arabic_qdrant.md)
