## About Industrial Asset Health AI

Industrial Asset Health AI is a research prototype for answering multiple-choice diagnostic questions about failure modes, sensor measurements, and industrial assets. It compares (1) Qwen2.5-7B-Instruct fine-tuned through QLoRA and (2) option-wise hybrid retrieval combining TF-IDF and MiniLM semantic search with supporting and contradicting examples.

Users can type a question with 2–5 choices, load a held-out example, compare the methods side by side, and export a CSV. On the original 534-question held-out test set, QLoRA achieved **88.01%** accuracy and hybrid retrieval **73.41%**.

**Limitations:** Research-only prototype; not validated for plant maintenance or safety decisions. Original QLoRA benchmark used 4-bit NF4 while the Space currently uses FP16. Performance can degrade on reworded or out-of-distribution questions. QLoRA outputs an option label, not validated causal reasoning. ZeroGPU has usage quotas and variable latency. Do not enter confidential industrial information.
