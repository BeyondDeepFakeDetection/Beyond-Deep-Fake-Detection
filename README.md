# Beyond Deep Fake Detection (2025)

This repository contains the official implementation of our paper:

> **Beyond Deep Fake Detection: Framework for Semantic Calibration of deceptive media distributions.**

## 🔗 Links

- 📝 Paper: [2025 PDF Link] <!-- update once available -->
- 🤗 Hugging Face: [https://huggingface.co/BeyondDeepFakeDetection](https://huggingface.co/BeyondDeepFakeDetection)
- 📊 Datasets: [https://huggingface.co/BeyondDeepFakeDetection](https://huggingface.co/BeyondDeepFakeDetection)
- 📦 Fine-tuned Models: [https://huggingface.co/BeyondDeepFakeDetection](https://huggingface.co/BeyondDeepFakeDetection)
- 📁 Code: [https://github.com/BeyondDeepFakeDetection/Beyond-Deep-Fake-Detection](https://github.com/BeyondDeepFakeDetection/Beyond-Deep-Fake-Detection)

---

## 🧠 Overview

**Beyond Deepfake Detection** addresses a key limitation in current deepfake detection systems: the overreliance on low-level visual artifacts that may disappear as generative models improve. Instead of merely asking *"Is this media fake?"*, our work shifts the focus to a more meaningful question: **"Is the semantic content deceptive?"**

We formally define the notion of **deception** in online media and introduce **semantic calibration** — a framework that detects media samples which distort semantic expectations, even when they are synthetically flawless. Our pipeline operates by:

- Captioning media content into text using image-to-text and audio-to-text models
- Computing acceptance probabilities for semantic consistency using fine-tuned large language models (LLMs)
- Rejection sampling over the semantic space to detect content that is surprising or overrepresented under a calibrated distribution

This approach is:

- **Modality-agnostic**: applicable to any media that can be converted to text  
- **Explainable**: decisions are based on interpretable semantic tokens  
- **Robust to future generative advances**: does not rely on generative artifacts

We release:
- Code to reproduce all experiments from the paper.
- Pretrained GPT-2 models fine-tuned on general and real distributions.
- Semantically distorted datasets simulating distribution shifts.
- Evaluation script for kl-divergence and deception reduction metrics.
---

## 📦 Installation

We recommend using a virtual environment:

```bash
git clone https://github.com/BeyondDeepFakeDetection/.git
cd Beyond-Deep-Fake-Detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ✅ Quick Start
We will navigate through an example of running the full pipeline and reproducing results from the paper.

### 1. Select dataset of choice:

### 2. Load general model (fine tuned on the chosen dataset)

### 3. Load real model with split of your choice (fine tuned on the chosen dataset)
