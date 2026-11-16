---
title: Industrial Asset Health AI
emoji: ⚙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: 3.12
---

# Industrial Asset Health AI — Hugging Face ZeroGPU

A Gradio demonstration of fine-tuned Qwen2.5-7B-Instruct on the FailureSensorIQ benchmark.

## Before uploading

1. Create a **public model repository** containing your *trained final* QLoRA adapter files (`adapter_model.safetensors`, `adapter_config.json`, tokenizer assets if desired). Upload model artifacts **only**; no training checkpoints. Review dataset/model license terms before redistributing files. A private model repository would require a read token configured in the Space and compatible token handling.
2. Create a Hugging Face Space with the **Gradio** SDK. Choose **ZeroGPU** hardware in Space Settings > Hardware. Currently, free personal accounts in good standing (verified email, 30+ day-old account) can create up to two ZeroGPU Spaces.
3. Add a Space **Variable** named `ADAPTER_REPO` containing the model repository identifier (for example, `your-username/failuresensoriq-qlora-adapter`). The provided identifier is an example, not an existing repository.
4. Upload `README.md`, `app.py`, `requirements.txt`, and `data/examples.jsonl` to the root of the Space.
5. Wait for the build to finish, then test a manually entered question and an example.

## Precision choice

The original paper used **4-bit NF4** QLoRA; for deployment simplicity, this Space defaults to **float16** base-model loading with the same LoRA adapter. It needs more memory, but the default ZeroGPU `large` size has 48GB GPU RAM. Since precision differs, do **not** claim identical predictions or the exact original 88.01% test accuracy for this live serving configuration without rerunning the evaluation.

To attempt exact original 4-bit NF4 inference instead, add `bitsandbytes` to `requirements.txt` and set Space variable `MODEL_DTYPE=nf4`. This loading path has not been tested on ZeroGPU and may fail under its CUDA emulation. For a production-quality replication, verify model responses against the saved test predictions before making any equivalence claims.

## Important limitations

* ZeroGPU supports **Gradio**, not Streamlit. This Space can be a standalone employer-facing demo; you can also link to it from your Streamlit app.
* On ZeroGPU, GPUs are shared and allocated to decorated functions; queues and per-visitor daily usage quotas apply. The `@spaces.GPU(duration=60)` value may need tuning after testing real inference time.
* Model initialization and first build require downloading the 7B base model, so initial loading can take significant time. Compatible CUDA-backed package versions matter.
* The app includes 20 original held-out test questions only; benchmark results are clearly marked historical. It does not pretend to run retrieval live.
* No private API key should be included in repository files.
* Research prototype only; do not use for live safety or maintenance decisions without independent engineering validation.
