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

# FailureSensorIQ: QLoRA vs hybrid retrieval comparison

This Gradio Space compares **live** predictions from the fine-tuned Qwen2.5-7B-Instruct + QLoRA adapter and **live** option-wise hybrid retrieval (TF-IDF + MiniLM). It is a research demonstration, not a validated safety or maintenance system.

## Uploading the update

Upload/replace the following files in the *root* of your **existing** Hugging Face Space:

- `app.py` — replaces the previous file.
- `retrieval_runtime.py` — new file, next to `app.py`.
- `requirements.txt` — replaces the previous file (adds CPU retrieval libraries).
- `data/retrieval_train.jsonl` — new file; **keep inside data/**.

You may retain your existing `data/examples.jsonl`, `data/main_test_results_percent.csv`, and `data/robustness_results_percent.csv` unchanged; identical copies are included in this package. `README.md` is optional to update but recommended. Do **not** upload these as an extra nested `industrial_asset_health_comparison/` directory.

Keep Space variable `ADAPTER_REPO` set to your Hugging Face model repository. Leave `MODEL_DTYPE=float16` as before if your prior app works. The FP16 deployment is not guaranteed to reproduce original NF4 results.

## How to use

1. Go to **Try a held-out example**, choose an example, then **Load example into question form**. The app moves to **Ask a diagnostic question** and sets the task metadata used by retrieval.
2. Click **Compare QLoRA and hybrid retrieval**. Retrieval runs on CPU (first call builds its index), followed by QLoRA inference via ZeroGPU. The two answers, ground-truth comparisons, elapsed times, method agreement, and per-option hybrid scores appear side by side.
3. Click **Download comparison CSV** to export the result.
4. For a custom question, manually fill in the question, 2–5 options, and select the accurate **Question type for hybrid retrieval**.

## Methodological details

- Retrieval memory comprises only the **2,133 original training questions** reconstructed by `train_test_split(test_size=0.20, random_state=42, stratify=asset_name)` from the original `all.jsonl`; the **534 held-out test questions** are excluded. The exported train subset was checked against the 534 saved QLoRA prediction IDs.
- Each training option becomes a positive or negative retrieval memory record.
- Lexical TF-IDF: unigrams + bigrams, sublinear TF and L2 normalization.
- Semantic: `sentence-transformers/all-MiniLM-L6-v2` normalized embeddings, CPU inference.
- Option-level hybrid similarity: `0.10 * minmax(TF-IDF) + 0.90 * minmax(semantic)`.
- Option score: mean of top-2 positive similarities minus `0.75` times mean of top-2 negative similarities.
- Scores are *relative retrieval scores*, **not calibrated probabilities**.
- Historical test accuracies: QLoRA 88.01%, hybrid retrieval 73.41%. These are benchmark results, not a claim about live-deployed performance.
- The hybrid model's `question_type` is automatically inferred from the selected task for custom questions; held-out examples use the original task metadata. Different task tagging can affect retrieval accuracy.

## Practical limitations

- The first retrieval request downloads the MiniLM model and encodes the training memory on CPU, which can take some time. Later retrieval requests reuse the cached index within the same Space process. A Space restart clears this in-memory index.
- Large QLoRA base weights still load at Space startup, as in the already working ZeroGPU deployment.
- GPU inference is subject to ZeroGPU queues and quotas, and the live model uses FP16 unless explicitly reconfigured.
- Avoid publishing private company maintenance records or confidential questions to a public demo.
- Public Space source includes the original training-split dataset subset. Verify the dataset's redistribution license if distributing publicly.
