---
title: My Dream Job Working at Tarteel & muaalem El Quran  | Part 2
description: Sharing my journey trying to join tarteel ai  and contribute to Islamic open source projects
date: 2025-12-26
author: kareem
draft: false
categories:
  - blogging
  - tarteel
  - speech_recognition
  - machine_learning
  - quran
  - job
image: images/tarteel_ai.jpg
---

## The Journey Continues | Tarteel AI career

Last year, I wrote about [my dream to work at Tarteel](https://kareemai.com/blog/posts/speech_recognition/my_dream_job_at_tarteel.html). 

It was a heartfelt post about my passion for Quranic technology. No one from Tarteel reached out  :) which was expected. 

Feeling passionate isn't enough; you must be a strong engineer and demonstrate your skills so companies see the value in hiring you.

**Talking is easy. Acting is hard.**

Some people messaged me on LinkedIn saying they shared the same feelings. Others mentioned they knew people at Tarteel and offered to help. 

But honestly, I didn't pursue this dream as actively as I should have.

Instead, I joined **xbites**, a real estate AI company where I learned invaluable lessons about building agentic systems, creating production-ready models, and meeting business requirements. 

We had hard times and a lot of fun.

This is my last week at xbites — I'm moving to another opportunity. But what about Tarteel?

---

## muaalem El Quran: A New Chapter

I'm building a habit of daily TIL (Today I Learned) posts to improve my learning and share knowledge with others. This blog is part of that journey.

I'm excited to share my work on **muaalem El Quran** , an open-source Quranic recitation model developed by **Abdalla Amal**. 

It's designed as an alternative to Tarteel for the developer community, enabling anyone to build their own solutions on top of it.

We have an ambitious roadmap ahead. I joined the team as an open-source contributor focusing on:

- AI model optimization

- Inference acceleration  

- Deployment infrastructure

---

## El muaalem Before Ramadan

We have two months before Ramadan and several deliverables for both business partners and open-source projects like **Quran Foundation**.

### The Challenge: How Should We Serve the Model?

I approached this systematically:

1. **Model Optimization** — Make inference faster

2. **Preprocessing Optimization** — Efficient audio processing

3. **Serving & Batching** — Production-ready API

---

## Model Optimization (Wav2Vec2Bert)

### Starting Point

| Metric | Value |
|--------|-------|
| Model | `Wav2Vec2BertForMultilevelCTC` |
| Architecture | Custom audio model with 11 output heads |
| Parameters | 605 million |
| Baseline RTF | 0.0303 |
| Test GPU | NVIDIA GTX 1660 Ti (6GB VRAM) |

> **What is RTF?** Real-Time Factor = Processing Time ÷ Audio Duration. An RTF of 0.03 means 1 second of audio processes in 30ms. Lower is better.

The model has **11 output heads** predicting different Tajweed attributes:

- Phonemes (Arabic sounds)

- Ghonna (nasalization)

- Qalqala (echoing sounds)

- Tafkheem/Tarqeeq (heavy/light letters)

- And 7 more...

---

### Optimization 1: PyTorch torch.compile

PyTorch 2.0 introduced `torch.compile()`  a one-line optimization that can sometimes achieve 2x speedups through graph optimization and kernel fusion.

```python
model = torch.compile(model)
```

**Result:** RTF 0.0303 — **No improvement**

The GTX 1660 Ti lacks support, limiting the benefits of `torch.compile`'s optimizations.

---

### Optimization 2: ONNX Runtime

ONNX (Open Neural Network Exchange) allows models to run on optimized inference engines. We exported the PyTorch model to ONNX format and tested different execution providers.

**CUDA Execution Provider:**
```python
session = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider"]
)
```

**Result:** RTF 0.0263 — **1.15x faster**

**Graph Optimizations:**
We applied ONNX Runtime's graph optimizer which fused 97 SkipLayerNormalization operations.

**Result:** RTF 0.0258 — **1.17x faster**

**FP16 Quantization:**
We attempted FP16 (half precision) to reduce memory and increase speed.

**Result:** RTF 0.1170 — **4x SLOWER!**

Without Tensor Cores, FP16 operations fell back to CPU, dramatically hurting performance.

---

### Optimization 3: TensorRT

NVIDIA TensorRT is a high-performance inference optimizer. It analyzes the network and applies:

- Layer fusion

- Kernel auto-tuning

- Precision calibration (FP16/INT8)

- Memory optimization

```python
trt_options = {
    "trt_fp16_enable": True,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": "trt_cache",
    "trt_cuda_graph_enable": True,
}

session = ort.InferenceSession(
    "model.onnx",
    providers=[
        ("TensorrtExecutionProvider", trt_options),
        "CUDAExecutionProvider"
    ]
)
```

**Key options explained:**

- `trt_fp16_enable`: Use half precision where beneficial

- `trt_engine_cache_enable`: Cache the optimized engine (first run is slow, subsequent runs are fast)

- `trt_cuda_graph_enable`: Reduce CPU overhead between kernel launches

**Result:** RTF 0.0178 — **1.70x faster! 🚀**

---

### Optimization 4: CTranslate2

CTranslate2 is a C++ inference engine optimized for transformer models. It supports INT8 quantization which compresses weights to 8-bit integers.

And the model here is the base part after trying to implement a Class wrapper so I will be able to convert the modified architecture with Ctranslate2 because their automatic script failed to convert it. 

```python
# Convert model
converter = TransformersConverter("wav2vec2bert_base")
converter.convert("ct2_model", quantization="int8")

# Load and run
encoder = ctranslate2.models.Wav2Vec2Bert(
    "ct2_model", 
    device="cuda", 
    compute_type="int8"
)
```

**Result:** RTF 0.0189 — **1.60x faster**

---

### Final Results Summary

| Runtime | RTF | Speedup | Notes |
|---------|-----|---------|-------|
| PyTorch (baseline) | 0.0303 | 1.00x | Reference |
| PyTorch + torch.compile | 0.0303 | 1.00x | No gain on GTX 1660 Ti |
| ONNX CUDA | 0.0263 | 1.15x | Easy win |
| ONNX Optimized | 0.0258 | 1.17x | Graph fusion |
| ONNX FP16 | 0.1170 | 0.26x | ❌ Avoid without Tensor Cores |
| CTranslate2 INT8 | 0.0189 | 1.60x | Great for deployment |
| **TensorRT FP16** | **0.0178** | **1.70x** | **Best result** |

---

## Serving the Model

With optimization complete, we needed a production-ready API. We chose **LitServe**  a lightweight Python framework that integrates seamlessly with our TensorRT-optimized model.

### Server Architecture

```python
class TensorRTAPI(ls.LitAPI):
    def setup(self, device):
        # Load ONNX model with TensorRT provider
        # Initialize tokenizers for all 11 output heads
        
    def decode_request(self, request):
        # Receive audio file upload
        # Load and preprocess with librosa
        # Extract features with HuggingFace processor
        
    def predict(self, input_features):
        # Run TensorRT inference
        # Return logits for all heads
        
    def encode_response(self, outputs):
        # CTC decode each head
        # Return JSON with all predictions
```

### API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -F "audio=@recitation.wav"
```

**Response:**
```json
{
  "phonemes": "ءِنَلَاهَبِكُلِشَيءِنعَلِۦمُ",
  "ghonna": "[لا غنة][مغن][لا غنة]",
  "qalqla": "[لا قلقلة]",
  ...
}
```

---

## The Dependency Hell with ONNX 

*Okay let me be real with you.*

I spent more time fighting package versions than actually optimizing the model. Not joking.

So you install `librosa` because you need audio stuff. It pulls in `numba`. Numba wants `numpy<=2.0`. Fine whatever.

Then you need `onnxruntime-gpu`. Guess what? It wants `numpy>=2.1`. 

Everything breaks. You Google. Stack Overflow says "just downgrade". You downgrade. Now `onnxruntime` breaks. Great.

And don't get me started on `ctranslate2` — the moment you install it, it downgrades like 5 packages you didn't even know you needed.

**The TensorRT disaster:**

I ran `uv pip install tensorrt`. Waited. And waited. **12 hours later** — still downloading. I gave up and went to sleep.

Next day I found out you need NVIDIA's special index:
```bash
pip install tensorrt --extra-index-url https://pypi.nvidia.com
```

Took 30 mins. WHY IS THIS NOT THE DEFAULT.

**Honestly?** Just use different virtual environments. Don't be like me trying to fit everything in one place. Your sanity is worth more.
## Key Learnings

1. **Know your hardware**: GTX 1660 Ti lacks Tensor Cores, so FP16 without TensorRT hurts performance

2. **TensorRT is powerful**: Even on consumer GPUs, it provides significant speedups

3. **Cache your engines**: TensorRT engine building is slow; always enable caching

4. **INT8 is viable**: CTranslate2's INT8 quantization offers good speedups with minimal accuracy loss

5. **Profile first**: Don't assume — measure each optimization's actual impact

There is more to come, and for the other part of optimization I am still exploring... Thanks!

### References

1. [Tensorrt](https://developer.nvidia.com/tensorrt)
2. [optimum](https://huggingface.co/docs/optimum/index)
3. [muaalem](https://github.com/obadx/quran-muaalem)