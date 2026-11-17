# Industrial Asset Health AI

**Failure–sensor reasoning with a fine-tuned language model and hybrid retrieval**

[**Live interactive demo**](https://huggingface.co/spaces/dseshukumar/industrial-asset-health-ai)

> **Research prototype:** This is not a validated industrial maintenance, diagnosis, or safety system. Engineering decisions require independent verification.

## Overview

Industrial operators and maintenance engineers often need to connect equipment failure modes with relevant sensor measurements. This application evaluates multiple-choice diagnostic questions about industrial assets and compares two independently developed approaches:

1. **Fine-tuned language model:** Qwen2.5-7B-Instruct adapted using QLoRA on FailureSensorIQ questions.
2. **Option-wise hybrid retrieval:** TF-IDF and MiniLM semantic search over labeled option-level records, combining supporting and contradicting evidence.

The public interface is built in **Gradio**, hosted on **Hugging Face Spaces with ZeroGPU**.

## Try the app

1. Open the [live Space](https://huggingface.co/spaces/dseshukumar/industrial-asset-health-ai).
2. Enter a diagnostic question and **2–5 possible answers**, or select **Try a held-out example**.
3. Click **Compare QLoRA and hybrid retrieval**.
4. Review each method's selected answer and inference time. For held-out examples, compare against the known answer.
5. Export a comparison CSV if desired.

> Initial hybrid retrieval initialization may download the MiniLM encoder and build an in-memory index. ZeroGPU predictions may be queued or subject to quota limits.

## Screenshots

Add screenshots captured from the **current live version** of the app:

- `screenshots/question-form.png`: question entry and answer choices
- `screenshots/comparison.png`: actual side-by-side predictions and timing, with no private information
- `screenshots/benchmark.png`: benchmark tab with historical results

Once captured, uncomment these Markdown lines:

<!-- ![Question entry](screenshots/question-form.png) -->
<!-- ![Side-by-side predictions](screenshots/comparison.png) -->
<!-- ![Research benchmark](screenshots/benchmark.png) -->

## Architecture

```text
Browser (Gradio / Hugging Face Space)
             |
     Question + 2–5 choices
             |
       +-----+-----+
       |           |
  Hybrid retrieval   QLoRA inference
  (CPU)              (ZeroGPU)
  TF-IDF + MiniLM    Qwen2.5-7B + adapter
       |           |
       +-----+-----+
             |
      Side-by-side answers
      Agreement, timing, CSV
```

## Method details

| Component | Implementation |
|---|---|
| Base LLM | Qwen2.5-7B-Instruct |
| Fine-tuning | QLoRA; the Space loads the fine-tuned adapter with the base model |
| Deployed LLM inference | FP16 (original benchmark used 4-bit NF4) |
| Retrieval | Option-level TF-IDF + `all-MiniLM-L6-v2` semantic similarity |
| Retrieval mixture | 0.10 lexical + 0.90 semantic, individually normalized per query |
| Support / contradiction | Top 2 matching correct / incorrect option records |
| Option score | Mean support score − 0.75 × mean contradiction score |
| Retrieval data | 2,133 training questions; held-out test questions excluded |
| Web framework | Gradio |
| Hosting | Hugging Face Spaces / ZeroGPU |

Hybrid retrieval scores are **relative similarity scores, not calibrated probabilities**.

## Research evaluation

| Method | Original held-out test accuracy |
|---|---:|
| QLoRA fine-tuned Qwen2.5-7B | **88.01%** |
| Hybrid retrieval | **73.41%** |
| Zero-shot Qwen2.5-7B | 55.43% |
| Few-shot Qwen2.5-7B | 54.49% |

These figures come from an **offline benchmark of 534 held-out questions**, not from a full evaluation of the deployed FP16 app.

The research also found pronounced performance drops under perturbed question formats and answer choices. This underscores the need for careful validation before any practical maintenance application.

## Code organization

```text
app.py                  # Gradio interface and model inference
retrieval_runtime.py    # Hybrid option-wise retrieval
requirements.txt        # Python dependencies
README.md               # Project documentation
data/
  examples.jsonl
  retrieval_train.jsonl
  main_test_results_percent.csv
  robustness_results_percent.csv
screenshots/             # Captured from the public demo (add manually)
```

The adapter weights are hosted separately in a Hugging Face model repository; the base model is downloaded from its upstream repository.

## Reproducing the deployment

1. Create a **Gradio** Hugging Face Space with appropriate ZeroGPU hardware access.
2. Upload `app.py`, `retrieval_runtime.py`, `requirements.txt`, and the `data/` directory while preserving relative paths.
3. In **Settings → Variables**, set `ADAPTER_REPO` to the model repository containing your fine-tuned LoRA adapter.
4. Keep `MODEL_DTYPE=float16` to use the tested startup path. The application also includes an NF4 option whose compatibility must be independently established.
5. Restart the Space, open the app, load a held-out example, and compare both methods.

Do not commit API tokens or private credentials. Review licensing/redistribution rights for the model adapter and FailureSensorIQ-derived data before republishing them.

## Important limitations

- This is a **research demonstration**, not an industrially validated fault-diagnosis system or autonomous maintenance recommendation engine.
- Performance may degrade on out-of-distribution assets, unfamiliar operating conditions, changes in answer choices, and question paraphrases.
- Multiple-choice predictions are not verified causal explanations; the LLM was trained to return a letter.
- The public Space runs the LLM with FP16 rather than the original NF4 evaluation configuration; results may differ.
- ZeroGPU shares GPUs and has queue and quota restrictions; response times vary.
- Retrieval depends on the original training split and question/task metadata, which may not transfer directly to arbitrary plant records.
- Do not submit confidential industrial records to the public app.

## Credits

Research implementation based on the **FailureSensorIQ** benchmark and the manuscript *Retrieval-based and Fine-tuned LLM Approaches for Industrial Asset Health Monitoring and Decision Support* (Seshu K. Damarla and Xiuli Zhu).

Please check the benchmark dataset's original license and attribution requirements before releasing or reusing derived data publicly.
