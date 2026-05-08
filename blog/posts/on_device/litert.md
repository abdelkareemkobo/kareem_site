---
title: "LiteRT and Qualcomm AI Hub: On-Device ML Without the Cloud"
description: A practical guide for ML engineers on converting PyTorch models to LiteRT, benchmarking on real Android devices via ADB, and when to use Qualcomm AI Hub for NPU compilation.
author: kareem
date: 2026-04-25
draft: false
featured: false
categories:
  - blogging
  - blog/review/tool
  - on_device
  - mobile
  - optimization
image: images/litert.jpeg
---

In 2024, Google rebranded TensorFlow Lite as **LiteRT**.

it came with a new API, broader hardware support, and tighter integration with Qualcomm's AI ecosystem.

This post covers what [LiteRT](https://ai.google.dev/edge/litert/next/overview) is, when to use [Qualcomm AI Hub](https://aihub.qualcomm.com/resources) alongside it, and how to benchmark a model on a real Android device without needing a cloud account or writing any app code.

## Why On-Device Inference?

Running models on-device instead of a remote server has several practical advantages:

- **Privacy**: user data never leaves the device
- **Latency**: no network round-trip, response is immediate
- **Offline**: works without internet connectivity
- **Cost**: no cloud inference bill

These benefits come with a tradeoff: constrained compute, memory, and power budget on the device.

## What is LiteRT?

LiteRT is Google's on-device ML runtime for Android, iOS, embedded Linux, and microcontrollers.

It is the direct successor to TensorFlow Lite, with the same `.tflite` [flatbuffer format](https://flatbuffers.dev/).

The core runtime supports three execution backends:

- **CPU** via XNNPACK (default, broad op coverage)
- **GPU** via OpenGL ES / Vulkan delegates
- **NPU** via vendor-specific delegates (Qualcomm, MediaTek, etc.)

Models are converted to `.tflite` format from PyTorch, TensorFlow, or JAX using the `litert-torch` or `ai-edge-torch` Python libraries.

### LiteRT vs LiteRT-LM

| Package               | What it does                                                  |
| --------------------- | ------------------------------------------------------------- |
| **LiteRT**            | The core runtime — runs `.tflite` models on Android/iOS/Linux |
| **LiteRT-LM**         | Specialized runtime for LLMs on-device (Gemma, Llama, etc.)   |
| **litert-torch**      | Python library to **convert** PyTorch models → `.tflite`      |
| **ai-edge-torch**     | Deprecated predecessor to `litert-torch`                      |
| **ai-edge-quantizer** | Quantizes `.tflite` models (int8, fp16)                       |

## Converting a PyTorch Model to LiteRT

The conversion path from PyTorch to `.tflite` uses the `litert-torch` library:

```
import torch
import litert_torch

# Your model must be an nn.Module returning tensors (not dicts)
model.eval()
sample_inputs = (torch.randn(1, 438, 160),)

tflite_model = litert_torch.convert(model, sample_inputs)
tflite_model.export("model.tflite")
```

Key requirements:

- Model must use `torch.export` compatible ops
- Inputs and outputs must be tensors or tuples of tensors (no dicts)

## Benchmarking on a Real Device Without Writing Any App Code

One underappreciated feature of LiteRT is that you can benchmark a `.tflite` model directly on a real Android device using ADB, with no app development required.

First, install the benchmark APK:

```
wget https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model.apk
adb install -r -d -g android_aarch64_benchmark_model.apk
```

Push your model and run:

```
adb push model.tflite /data/local/tmp/
adb shell am start -S \
  -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity \
  --es args '"--graph=/data/local/tmp/model.tflite --num_threads=4"'
```

Read results:

```
adb logcat | grep "Inference timings"
```

This gives you real latency numbers on your target hardware in minutes.

## When You Need Qualcomm AI Hub

The ADB approach runs your model on the CPU delegate by default. To unlock the Qualcomm NPU (Hexagon DSP), you need to compile the model specifically for that chip. This is where Qualcomm AI Hub comes in.

AI Hub has three products:

- **Models**: pre-optimized models ready to download and deploy
- **Apps**: sample Android app code to bundle models
- **Workbench**: compile and profile your custom model on 50+ hosted Qualcomm devices

The Workbench workflow in Python:

```
import qai_hub

model = qai_hub.upload_model("model.tflite")
job = qai_hub.submit_compile_job(
    model,
    device=qai_hub.Device("Snapdragon 8 Gen 2"),
    options="--target_runtime tflite"
)
compiled_model = job.download_target_model()
```

### Final Thougts

This is part of my journey exploring on-device models development.
I finished the optimization of voice models on my tablet.
I will share the results in the next post.
For now you can read more about :

1. [My dream job at Tarteel part 1](https://kareemai.com/blog/posts/speech_recognition/my_dream_job_at_tarteel.html)
2. [Muaalem El Quran part 2](https://kareemai.com/blog/posts/speech_recognition/muaalm_quran_chance.html) : the app i was optimizing.d
