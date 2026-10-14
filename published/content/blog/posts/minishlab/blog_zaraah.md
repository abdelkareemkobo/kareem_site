---
section: blog
title: 'Bojji & Zarra Arabic Embeddings: MTEB vs all-MiniLM-L6-v2'
description: >-
  Analyze Bojji and Zarra model2vec embeddings for Arabic, compare MTEB average
  score with all-MiniLM-L6-v2 and other static models.
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

## Arabic Embedding Models

This blog post introduces the **Bojji and Zarra** family of static embedding models, designed for Arabic language tasks and built using the **model2vec** distillation technique from MinishLab. We compare performance to popular baselines such as **all-MiniLM-L6-v2** using MTEB average score – a query that’s currently rising in search trends.

These models distill knowledge from larger transformer models, such as SBERT, into compact, efficient embeddings. 

This approach balances performance with speed and resource efficiency.

Below, I explore the Bojji models and  their relationship to **Potion** models, their strengths and limitations, and their applications in Arabic embedding tasks.

### What are Potion Models?

**Potion models** combine innovative techniques to create high-performing, compact static embeddings. 

I liken them to Bojji from *Ousama Ranking* small in size but capable of competing with giants like Jina AI and BGE models.

<img src="/assets/4495c469616e57979248d4b7.webp" alt="Bojji Embedding" class="img-fluid figure-img">

Key features of Potion models include:

- **Superior Performance**: They outperform traditional static embeddings like GloVe and FastText across various tasks, matching the performance of models like **all-MiniLM-L6-v2** in English.

- **Compact Size**: With approximately 2–4 million parameters, they are ~55 times smaller than GloVe, with model sizes ranging from 8 MB to 30 MB.

- **Efficiency**: Designed for CPU execution and browser-based applications, they are ideal for edge devices and low-resource environments.

- **MTEB Performance**: They achieve an average MTEB score above 50%, making them highly competitive for their size.

### What is the model2vec Distillation Method?

The **model2vec** distillation method addresses the challenge of creating fast, compact sentence transformers. 


It transforms large sentence transformer models into static embeddings that are up to **500x faster** and **15x smaller**, with only a minor performance trade-off.

Unlike traditional methods like GloVe, model2vec captures knowledge from large sentence transformers, producing uncontextualized word vectors. 

While this sacrifices some contextual nuance, it offers significant advantages in:

- **Speed**: Up to 500x faster inference.

- **Size**: Models reduced by up to 50x, ranging from 8 MB to 30 MB.

- **Versatility**: Sufficient word representations for most NLP applications.

For more details, refer to the [MinishLab blog](https://minishlab.github.io/) and [GitHub repository](https://github.com/MinishLab/model2vec).

## Jina Embeddings v3 for Arabic

The **jina-embeddings-v3** model is currently the top-performing open-source, zero-shot embedding model for Arabic on the MTEB leaderboard. 

It excels across various tasks and has been validated in production for Arabic applications.

However, its large size and high memory requirements make it computationally expensive and slow compared to other embedding models. 

To address this, I used model2vec to create a compact Arabic version, the **Zarra** and **bojji** with with another base model and different method, which retains strong performance while being significantly smaller and faster.

## Bojji and Zarra

- [Bojji HuggingFace](https://huggingface.co/NAMAA-Space/bojji)

- [Zarra HuggingFace](https://huggingface.co/NAMAA-Space/zarra)

The **Zarra** models are the first static embedding models for Arabic trained with **tokenlearn** on the Arabic subset of the [C4 dataset](https://huggingface.co/datasets/allenai/c4). 

They are optimized for Arabic-specific tasks and come in multiple sizes:

All variants support **float32** and **int8** quantization without performance loss, making them highly efficient for resource-constrained environments.

## Bojji Model vs Competitors

To evaluate Bojji’s performance, I compared it against several multilingual and Arabic-specific sentence transformer models using MTEB tasks tailored for Arabic.

```python
import json
import os
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text
import pandas as pd
dirs = {
    "muffakir_embedding": "/home/ai/kobo/bert_world/static_embedding/results/mohamed2811/Muffakir_Embedding/mohamed2811__Muffakir_Embedding/no_revision_available",
    "get_multilingual_base": "/home/ai/kobo/bert_world/static_embedding/results/Alibaba-NLP/gte-multilingual-base/Alibaba-NLP__gte-multilingual-base/ca1791e0bcc104f6db161f27de1340241b13c5a4",
    "arabic_retrieval_v1.0": "/home/ai/kobo/bert_world/static_embedding/results/omarelshehy/Arabic-Retrieval-v1.0/omarelshehy__Arabic-Retrieval-v1.0/no_revision_available",
    "arabic_sts_matryoshka": "/home/ai/kobo/bert_world/static_embedding/results/omarelshehy/Arabic-STS-Matryoshka/omarelshehy__Arabic-STS-Matryoshka/no_revision_available",
    "arabic_triplet_matryoshka_v2": "/home/ai/kobo/bert_world/static_embedding/results/Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2/Omartificial-Intelligence-Space__Arabic-Triplet-Matryoshka-V2/ed357f222f0b6ea6670d2c9b5a1cb93950d34200",
    "gate_arabert-v1": "/home/ai/kobo/bert_world/static_embedding/results/Omartificial-Intelligence-Space/GATE-AraBert-v1/Omartificial-Intelligence-Space__GATE-AraBert-v1/no_revision_available",
    "all_minilm_l6_v2": "/home/ai/kobo/bert_world/static_embedding/results/sentence-transformers/all-MiniLM-L6-v2/sentence-transformers__all-MiniLM-L6-v2/8b3219a92973c328a8e22fadcfa821b5dc75636a",
    "paraphrase-multilingual-MiniLM-L12-v2": "/home/ai/kobo/bert_world/static_embedding/results/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2/bf3bf13ab40c3157080a7ab344c831b9ad18b5eb",
    "Arabic-MiniLM-L12-v2-all-nli-triplet": "/home/ai/kobo/bert_world/static_embedding/results/Omartificial-Intelligence-Space/Arabic-MiniLM-L12-v2-all-nli-triplet/Omartificial-Intelligence-Space__Arabic-MiniLM-L12-v2-all-nli-triplet/6916465c43b984e955aa6dc72851474f0128f428",
    "silma_ai_embedding_sts_v0.1": "/home/ai/kobo/bert_world/static_embedding/results/silma-ai/silma-embedding-sts-v0.1/silma-ai__silma-embedding-sts-v0.1/no_revision_available",
    "zarra": "/home/ai/kobo/bert_world/static_embedding/results/jina_zaraaah_256_arabic/no_model_name_available/no_revision_available",
    # "jina_zaraah_256_arabic_int8": "/home/ai/kobo/bert_world/static_embedding/results/jina_zaraaah_256_arabic_int8/no_model_name_available/no_revision_available",
    # "zarra": "/home/ai/kobo/bert_world/static_embedding/results/jina_zaraaah_32_arabic/no_model_name_available/no_revision_available",
    # "jina_zaraah_32_int8_arabic": "/home/ai/kobo/bert_world/static_embedding/results/jina_zaraaah_32_arabic_int8/no_model_name_available/no_revision_available",
    # "jina_zaraah_64_int8_arabic": "/home/ai/kobo/bert_world/static_embedding/results/jina_zaraaah_64_int8_arabic/no_model_name_available/no_revision_available",
    # "jina_zaraah_64_arabic": "/home/ai/kobo/bert_world/static_embedding/results/jina_zaraaah_64_arabic/no_model_name_available/no_revision_available",
    # "jina_zaraah_16_arabic_int8": "/home/ai/kobo/bert_world/static_embedding/results/jina_zaraaah_16_arabic_int8/no_model_name_available/no_revision_available",
    # "jina_zaraah_16_arabic": "/home/ai/kobo/bert_world/static_embedding/results/jina_zaraaah_16_arabic/no_model_name_available/no_revision_available",
    # "jina_zarrah_4_arabic_int8": "/home/ai/kobo/bert_world/static_embedding/results/jina_zaraaah_4_arabic_int8/no_model_name_available/no_revision_available",
    # "jina_zarrah_4_arabic": "/home/ai/kobo/bert_world/static_embedding/results/jina_zaraaah_4_arabic/no_model_name_available/no_revision_available",
    # "jina_zarrah_8_arabic_int8": "/home/ai/kobo/bert_world/static_embedding/results/jina_zaraaah_8_arabic_int8/no_model_name_available/no_revision_available",
    # "jina_zarrah_8_arabic": "/home/ai/kobo/bert_world/static_embedding/results/jina_zaraaah_8_arabic/no_model_name_available/no_revision_available",
    # "jina_zarrah_2_arabic": "/home/ai/kobo/bert_world/static_embedding/results/jina_zaraaah_2_arabic/no_model_name_available/no_revision_available",
    # "jina_zarrah_2_arabic_int8": "/home/ai/kobo/bert_world/static_embedding/results/jina_zaraaah_2_arabic_int8/no_model_name_available/no_revision_available",
    # "jina_zarrah_256_superbpe": "/home/ai/kobo/bert_world/static_embedding/results/jinaai/jina-embeddings-v3_distilled_superbpe_256/no_model_name_available/no_revision_available",
    "potion-multilingual-128M": "/home/ai/kobo/bert_world/static_embedding/results/minishlab/potion-multilingual-128M/minishlab__potion-multilingual-128M/38ebd7f10f71e67fa8db898290f92b82e9cfff2a",
    "bojji":"/home/ai/kobo/bert_world/static_embedding/tokenlearn_new/results/nomic_zaraah_256/no_model_name_available/no_revision_available",
    # "king-bojji":"/home/ai/kobo/bert_world/static_embedding/tokenlearn_new/results/nomic_zaraah_512/no_model_name_available/no_revision_available",
    # "bojji":"/home/ai/kobo/bert_world/static_embedding/tokenlearn_new/results/bojji/no_model_name_available/no_revision_available",
    # "king-bojji":"/home/ai/kobo/bert_world/static_embedding/tokenlearn_new/results/king-bojji/no_model_name_available/no_revision_available"
    # "jina_zarrah_256_superbe_int8":"",
    # "jina_zarrah_384_superbe":"/home/ai/kobo/bert_world/static_embedding/results/zaraah_jinav3_v02_385D/no_model_name_available/no_revision_available"
    }

result_files = [
    "STS17.json",
    "STS22.v2.json",
    "MLQARetrieval.json",
    "MassiveIntentClassification.json",
    "MultiHateClassification.json",
    "MIRACLRetrievalHardNegatives.json",
    "XNLI.json",
]
```

```python
all_data = []

for model_name, model_path in dirs.items():
    model_results = {"model_name": model_name}
    for file_name in result_files:
        json_path = os.path.join(model_path, file_name)
        task_name = file_name.replace(".json", "")  # Use filename as task identifier

        main_score_val = None
        eval_time_val = None

        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                eval_time_val = data.get("evaluation_time")

                # Extract main_score. It can be under 'test', 'validation', or 'dev'.
                # MTEB usually prioritizes 'test', then 'dev' (for retrieval), then 'validation'.
                scores_section = data.get("scores", {})

                score_entry = None
                if "test" in scores_section and scores_section["test"]:
                    score_entry = scores_section["test"][0]
                elif (
                    "dev" in scores_section and scores_section["dev"]
                ):  # For MIRACL style
                    score_entry = scores_section["dev"][0]
                elif (
                    "validation" in scores_section and scores_section["validation"]
                ):  # For XNLI style if test is missing
                    score_entry = scores_section["validation"][0]

                if score_entry:
                    main_score_val = score_entry.get("main_score")

            except json.JSONDecodeError:
                print(f"Error decoding JSON for: {json_path}")
            except Exception as e:
                print(f"An error occurred while processing {json_path}: {e}")
        else:
            print(f"File not found: {json_path}")  # Helpful for debugging

        model_results[f"{task_name}_main_score"] = main_score_val
        # model_results[f"{task_name}_evaluation_time"] = eval_time_val

    all_data.append(model_results)

# Create DataFrame from the collected data
df = pd.DataFrame(all_data)

# Set model_name as index
df.set_index("model_name", inplace=True)

# Calculate the average of main_score columns
score_columns = [col for col in df.columns if col.endswith("_main_score")]
df["Average_main_score"] = df[score_columns].mean(axis=1)

# Create a MultiIndex for columns for better organization
if not df.empty:
    df.columns = pd.MultiIndex.from_tuples(
        [tuple(col.rsplit("_", 1)) if col != "Average_main_score" else ("Average", "main_score") for col in df.columns],
        names=["Task", "Metric"]
    )
    # Sort columns for consistent order: Task Name, then Metric (evaluation_time, main_score)
    df = df.sort_index(axis=1, level=[1, 0])

# Sort DataFrame by Average_main_score in descending order
df = df.sort_values(("Average", "main_score"), ascending=False)

# Create a rich table
console = Console()
table = Table(title="Model Evaluation Summary")

# Add columns to the table with abbreviated names
table.add_column("Model", style="cyan", no_wrap=True)
# Create short aliases for task names (first 5 characters or less if shorter)
task_aliases = {task: task[:5] for task, _ in df.columns}
task_aliases["Average"] = "Avg"  # Short alias for Average column
for task, metric in df.columns:
    table.add_column(task_aliases[task], justify="right")

# Add rows to the table
for model_name, row in df.iterrows():
    row_data = [model_name]
    for value in row:
        row_data.append(f"{value:.4f}" if isinstance(value, (int, float)) and not pd.isna(value) else "-")
    table.add_row(*row_data)

# Display the table
console.print(table)
```

I filtered the most related MTEB tasks that supports Arabic-script only the evalution script is in the references blow. 
We can say that the average score for the Zarra are very low compared to the other models, but didn't let the Average score fool you!
Average is affected with the outliers so, if one task is the low the final answer with be low also. 

But from the first look, we can see the peformance is similar to the Arabic versions of MiniLM-L12 in Average and if you looked at the Sentence similarity for STS22 it's score are very good compared to static-embedding model. 

### Understanding MTEB Tasks for Arabic

The Massive Text Embedding Benchmark (MTEB) evaluates embedding models across various tasks. Here’s a breakdown of the tasks used :

- MIRACLRetrievalHardNegatives: Measures retrieval accuracy for hard negative examples, critical for search and question-answering systems. Zarra’s lower score here reflects its static embedding nature, which sacrifices some contextual nuance.

- MLQARetrieval: Tests retrieval performance on multilingual question-answering datasets, where Zarra performs comparably to MiniLM models.

- STS17 & STS22: Evaluates semantic textual similarity, where Zarra excels, particularly in STS22, with scores rivaling larger models.

- XNLI: Assesses natural language inference, where Zarra’s performance is competitive despite its compact size.

These tasks highlight Zarra’s strengths in semantic similarity and efficiency, making it ideal for applications like chatbots and lightweight search systems

You can see the peformance for every task in MTEB here


```python


# Initialize rich Console
console = Console()

# Filter for score columns only
score_columns = df.columns[df.columns.get_level_values('Metric') == 'score']

# List of model name patterns to bold
bold_names = ['zarra', "bojji",]

# Iterate over each score column
for col in score_columns:
    # Sort DataFrame by the current column (descending)
    df_sorted = df.sort_values(by=col, ascending=False)
    
    # Create a rich Table
    table = Table(title=f"Sorted by {col[0]} (Score)", title_style="bold magenta", show_lines=True)
    
    # Add columns: Model Name and the specific score column
    table.add_column("Model Name", style="cyan", no_wrap=True)
    table.add_column(f"{col[0]}", justify="right", style="green")
    
    # Add rows: Model Name and the value for the sorted column
    for idx, row in df_sorted.iterrows():
        value = row[col]
        # Color-code based on value
        color_style = "green" if value > 0.6 else "yellow" if value > 0.3 else "red"
        # Apply bold if model name contains any bold_names pattern
        row_style = f"bold {color_style}" if any(name.lower() in str(idx).lower() for name in bold_names) else color_style
        table.add_row(
            Text(str(idx), style="bold green" if any(name.lower() in str(idx).lower() for name in bold_names) else "cyan"),
            Text(f"{value:.3f}", style=row_style)
        )
    
    # Print the table
    console.print(table)
    console.print("\n")  # Add spacing between tables
```

### Bojji vs. all-MiniLM

The Bojji model outperform the **all-MiniLM** family from SBERT in Arabic tasks while being significantly faster and capable of running on CPU. 

This makes  Bojji an excellent lightweight alternative for applications requiring efficient Arabic embeddings.

Also the models performance is most of the time better than the **potion-multilingual-128M** which indicates that techniques are working with any language not just English. 

### Arabic RAG Leaderboard

To complement MTEB evaluations, I tested Zarra on the **Arabic-RAG Leaderboard**, which provides a robust benchmark for Arabic-specific tasks. Zarra ranks 37 out of 45 models with an average score of **36.84**. 

This is impressive, as Zarra is the smallest model in the leaderboard, highlighting its efficiency and competitive performance in resource-constrained settings.

#### What about Bojji perfomrance in Arabic RAG Leaderboard?

Actually the first model  with Zarra and not bojji is the second release of Zarra with different updates but..i know that if i didn't write the blog as it as, i will not publish it again!

Bojji will be test on the Arabic Rag soon. 

## Speed comprsion 


```python
import time
import csv
import torch
from model2vec import StaticModel
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table
import numpy as np

# Initialize rich console
console = Console()

# Check for GPU availability
if not torch.cuda.is_available():
    console.print("[red]GPU not available. Falling back to CPU for all models.[/red]")

# Define devices for each model
cpu_device = torch.device("cpu")
gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define models with their respective devices
models = {
    "zarra": {"model": StaticModel.from_pretrained("NAMAA-Space/zarra"), "device": cpu_device},
    # "jina_zaraah_32": {"model": StaticModel.from_pretrained("Abdelkareem/zaraah_jina_v3_32D"), "device": cpu_device},
    "bojji":{"model":StaticModel.from_pretrained("Abdelkareem/bojji"),"device":cpu_device},
    # "king-bojji":{"model":StaticModel.from_pretrained("Abdelkareem/king-bojji"),"device":cpu_device},
    "potion-multilingual-128M": {"model": StaticModel.from_pretrained("minishlab/potion-multilingual-128M"), "device": cpu_device},
    "paraphrase-multilingual-MiniLM-L12-v2": {"model": SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=gpu_device), "device": gpu_device},
    "silma_ai_embedding_sts_v0.1": {"model": SentenceTransformer("silma-ai/silma-embedding-sts-v0.1", device=gpu_device), "device": gpu_device},
    "muffakir_embedding": {"model": SentenceTransformer("mohamed2811/Muffakir_Embedding", device=gpu_device), "device": gpu_device},
    "get_multilingual_base": {"model": SentenceTransformer("Alibaba-NLP/gte-multilingual-base", device=gpu_device, trust_remote_code=True), "device": gpu_device},
    "arabic_retrieval_v1.0": {"model": SentenceTransformer("omarelshehy/Arabic-Retrieval-v1.0", device=gpu_device), "device": gpu_device},
    "arabic_triplet_matryoshka_v2": {"model": SentenceTransformer("Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2", device=gpu_device), "device": gpu_device},
}

# Dataset: Synthetic multilingual sentences
sentences = (
    ["This is a short sentence."] * 3000 +
    ["هذه جملة قصيرة."] * 3000 +  # Arabic: "This is a short sentence."
    ["Este es un texto largo " + "word " * 100] * 4000  # Long Spanish sentences
)
batch_size = 32

# Prepare results storage
results = []

# Benchmark each model
for name, config in models.items():
    model = config["model"]
    device = config["device"]
    
    if model is None:
        console.print(f"[yellow]Skipping {name}: Model not loaded[/yellow]")
        results.append({"Model": name, "Speed (sentences/second)": "N/A", "Device": str(device)})
        continue

    # For SentenceTransformer models, ensure device is set explicitly
    if isinstance(model, SentenceTransformer):
        model.to(device)
    
    start_time = time.time()
    embeddings = model.encode(sentences, batch_size=batch_size, show_progress_bar=False, device=device if isinstance(model, StaticModel) else None)
    elapsed_time = time.time() - start_time
    speed = len(sentences) / elapsed_time
    results.append({"Model": name, "Speed (sentences/second)": f"{speed:.2f}", "Device": str(device)})
    console.print(f"Completed {name} on {device}: {speed:.2f} sentences/second")

# Save results to CSV
with open("benchmark_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Model", "Speed (sentences/second)", "Device"])
    writer.writeheader()
    writer.writerows(results)

# Display results using rich
table = Table(title="Model Benchmark Results")
table.add_column("Model", style="cyan")
table.add_column("Speed (sentences/second)", style="magenta")
table.add_column("Device", style="green")

for result in results:
    table.add_row(result["Model"], result["Speed (sentences/second)"], result["Device"])

console.print(table)
# console.print("[green]Results saved to benchmark_results.csv[/green]")
```

```text
Some weights of the model checkpoint at Alibaba-NLP/gte-multilingual-base were not used when initializing NewModel: ['classifier.bias', 'classifier.weight']
- This IS expected if you are initializing NewModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing NewModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
```

```python
from huggingface_hub import get_safetensors_metadata

# metadata = get_safetensors_metadata("minishlab/potion-multilingual-128M")
# metadata
# metadata.files_metadata["model.safetensors"].metadata
# "minishlab/potion-multilingual-128M"
model_id = "NAMAA-Space/zarra"
metadata = get_safetensors_metadata(model_id)

get_safetensors_metadata(model_id)
```

```text
SafetensorsRepoMetadata(metadata=None, sharded=False, weight_map={'embeddings': 'model.safetensors'}, files_metadata={'model.safetensors': SafetensorsFileMetadata(metadata={}, tensors={'embeddings': TensorInfo(dtype='F32', shape=[249999, 256], data_offsets=(0, 255998976), parameter_count=63999744)}, parameter_count={'F32': 63999744})}, parameter_count={'F32': 63999744})
```

```python
import csv
from huggingface_hub import get_safetensors_metadata
from rich.console import Console
from rich.table import Table
import uuid

# Initialize rich console
console = Console()

# Define models
models = {
    "zarra": "NAMAA-Space/zarra",
        # "king-bojji": "Abdelkareem/king-bojji",
    "bojji": "Abdelkareem/bojji",
    "potion-multilingual-128M": "minishlab/potion-multilingual-128M",
    "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "silma_ai_embedding_sts_v0.1": "silma-ai/silma-embedding-sts-v0.1",
    "muffakir_embedding": "mohamed2811/Muffakir_Embedding",
    "arabic_retrieval_v1.0": "omarelshehy/Arabic-Retrieval-v1.0",
    "arabic_triplet_matryoshka_v2": "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
    "get_multilingual_base": "Alibaba-NLP/gte-multilingual-base",

}

def get_model_info(model_id):
    try:
        metadata = get_safetensors_metadata(model_id)
        
        # Extract total parameter count
        total_params = sum(metadata.parameter_count.get(dtype, 0) for dtype in metadata.parameter_count)
        num_parameters = round(total_params / 1e6, 2)  # Convert to millions

        # Calculate size in MB
        size_bytes = 0
        for file, file_metadata in metadata.files_metadata.items():
            for tensor_info in file_metadata.tensors.values():
                # Estimate size based on dtype and parameter count
                dtype_size = 4  # Default to 4 bytes (F32)
                if tensor_info.dtype in ['F16', 'BF16']:
                    dtype_size = 2
                elif tensor_info.dtype in ['F64']:
                    dtype_size = 8
                size_bytes += tensor_info.parameter_count * dtype_size
        size_mb = round(size_bytes / (1024 ** 2), 2)  # Convert to MB

        return num_parameters, size_mb
    except Exception as e:
        console.print(f"[yellow]Error: Could not fetch model information for {model_id}. {str(e)}[/yellow]")
        return 0, 0

# Fetch model information
def fetch_model_information(model_name):
    try:
        return get_model_info(model_name)
    except Exception as e:
        console.print(f"[red]Error: Could not fetch model information for {model_name}. {str(e)}[/red]")
        return 0, 0

# Collect results
results = []
for name, path in models.items():
    num_parameters, size_mb = fetch_model_information(path)
    results.append({
        "Model": name,
        "Parameters (M)": f"{num_parameters:.2f}" if num_parameters else "N/A",
        "Size (MB)": f"{size_mb:.2f}" if size_mb else "N/A",
    })

# Calculate relative size and "less than largest" factor
max_size = max(float(result["Size (MB)"]) for result in results if result["Size (MB)"] != "N/A") if any(result["Size (MB)"] != "N/A" for result in results) else 1
for result in results:
    size_mb = float(result["Size (MB)"]) if result["Size (MB)"] != "N/A" else 0
    result["Relative to Largest (%)"] = f"{(size_mb / max_size * 100):.2f}" if size_mb else "N/A"
    result["Less than Largest (x)"] = f"{(max_size / size_mb):.2f}" if size_mb else "N/A"

# Save results to CSV
try:
    with open("model_info_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Parameters (M)", "Size (MB)", "Relative to Largest (%)", "Less than Largest (x)"])
        writer.writeheader()
        writer.writerows(results)
    console.print("[green]Results saved to model_info_results.csv[/green]")
except IOError as e:
    console.print(f"[red]Failed to save CSV: {e}[/red]")

# Display results using rich
table = Table(title="Model Information Results")
table.add_column("Model", style="cyan", justify="left")
table.add_column("Parameters (M)", style="yellow", justify="right")
table.add_column("Size (MB)", style="green", justify="right")
table.add_column("Relative to Largest (%)", style="magenta", justify="right")
table.add_column("Less than Largest (x)", style="blue", justify="right")

for result in results:
    table.add_row(
        result["Model"],
        result["Parameters (M)"],
        result["Size (MB)"],
        result["Relative to Largest (%)"],
        result["Less than Largest (x)"]
    )

console.print(table)
```

It's very clear the main advantage of the static embedding models here: 

They can process large number of samples on cpu which make them very useful for : 
1. Clustring
2. Build classification pipelines on top of them 
3. Use them in edge-devices 
4. base models for chunking algorithms and dudplications..more to come soon ! ❤️

### What’s Next for Bojji?

It's just the start with initial tests, there is more to explore from the base models, datasets and the new features from minishlab which will try to narrow the gab between model2vec and sentence-transformers. 

<img src="/assets/14bbe81f9f456b5c2b60dae8.webp" alt="Zarra and Bojji" class="img-fluid figure-img">

Also thanks a lot for the minishlab team for their continous help to debug and update the models with me! 

### Bojji model references 

1. [Arabic Leaderboard](https://huggingface.co/blog/Navid-AI/arabic-rag-leaderboard)
2. [MTEB](https://huggingface.co/spaces/mteb/leaderboard) 
3. [Minishlab](https://minishlab.github.io/)
4. [NAMAA-Space colleciton]()/blog/posts/minishlab/blog_zaraah.md



<iframe src="/assets/01eb47d96c2df863e81f9c3c.htm" title="Saved notebook visualizations" width="100%" height="640" loading="lazy"></iframe>
