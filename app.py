"""FailureSensorIQ ZeroGPU research demo.

Set ADAPTER_REPO as a Hugging Face Space variable (your public HF model repository).
MODEL_DTYPE defaults to float16 for ZeroGPU compatibility; change to nf4 for
closer reproduction of original 4-bit QLoRA evaluation, if supported.
"""
import csv
import io
import json
import os
import re
from pathlib import Path

import spaces  # import before torch/CUDA initialization
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).resolve().parent
BASE = 'Qwen/Qwen2.5-7B-Instruct'
ADAPTER = os.environ.get('ADAPTER_REPO', '').strip()
DTYPE = os.environ.get('MODEL_DTYPE', 'float16').strip().lower()
SYSTEM = ('You are an industrial asset health monitoring assistant. '
          'Answer multiple-choice questions about assets, failure modes, and sensors. '
          'Return only one option label. Do not explain.')

if not ADAPTER:
    raise RuntimeError('Set ADAPTER_REPO in your Space Settings > Variables to your Hugging Face adapter repository.')

# ZeroGPU guidance: initialize onto CUDA at module scope, not inside a GPU handler.
# Float16 is default to avoid potential bitsandbytes/CUDA emulation incompatibilities.
# This has greater memory use but fits within the default 48GB ZeroGPU GPU size;
# output equivalence to original NF4 evaluation is NOT guaranteed.
tokenizer = AutoTokenizer.from_pretrained(BASE)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# ZeroGPU startup emulates CUDA for .to("cuda") calls at module scope.
# Standard FP16 loading with device_map="cuda" can bypass that emulation.
if DTYPE == 'nf4':
    # Quantized loading path; support depends on ZeroGPU/bitsandbytes versions.
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=quant, device_map='cuda',
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
else:
    # Plain FP16: assemble model and adapter on CPU, then hand off to
    # ZeroGPU's CUDA-emulating model packer via explicit .to('cuda').
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
    model = model.to('cuda')
model.eval()

with open(ROOT / 'data/examples.jsonl', encoding='utf-8') as f:
    examples = [json.loads(line) for line in f if line.strip()]

ASSETS = ['Electric motor', 'Pump', 'Compressor', 'Steam turbine',
          'Aero gas turbine', 'Industrial gas turbine', 'Electric generator',
          'Fan', 'Power transformer', 'Reciprocating internal combustion engine']
LABELS = 'ABCDE'


def format_question(question, choices):
    options = '\n'.join(f'{LABELS[i]}. {value}' for i, value in enumerate(choices))
    user = ('Select the single best answer option for the following industrial failure-sensor question.\n'
            'Return only the option label, for example A.\n\n'
            f'Question:\n{question}\n\nOptions:\n{options}\n\nAnswer:')
    return tokenizer.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}],
        tokenize=False, add_generation_prompt=True,
    )


def parse_label(raw, choices):
    valid = LABELS[:len(choices)]
    text = raw.strip().upper()
    match = re.fullmatch(rf'([{valid}])\s*[.):]?', text)
    if match:
        return match.group(1)
    # Explicitly signal invalid outputs rather than manufacture a predicted label.
    return None


@spaces.GPU(duration=60)
def predict(asset, question, n_choices, a, b, c, d, e, expected):