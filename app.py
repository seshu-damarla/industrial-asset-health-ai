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
    # Plain FP16: explicitly load adapter on CPU because PEFT can infer
    # a fake CUDA device under ZeroGPU before a physical GPU is allocated.
    # Assemble model and adapter on CPU, then hand off to
    # ZeroGPU's CUDA-emulating model packer via explicit .to('cuda').
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, ADAPTER, torch_device='cpu')
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
    n = int(n_choices)
    choices = [a, b, c, d, e][:n]
    question = (question or '').strip()
    choices = [str(s or '').strip() for s in choices]
    if not question or any(not choice for choice in choices):
        return 'Enter a question and fill in every answer choice.', '', '', ''
    prompt = format_question(question, choices)
    batch = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=768).to('cuda')
    with torch.inference_mode():
        output = model.generate(**batch, max_new_tokens=8, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(output[0, batch['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    label = parse_label(raw, choices)
    if label is None:
        result = f'Invalid/ambiguous model response: {raw!r}'
        correctness = 'Cannot score prediction'
    else:
        result = f'{label}. {choices[LABELS.index(label)]}'
        correctness = (('Correct' if label == expected else 'Incorrect')
                       if expected in LABELS[:n] else 'No ground truth supplied')
    record = {'asset': asset, 'question': question, 'options': choices,
              'predicted_label': label, 'predicted_answer': choices[LABELS.index(label)] if label else None,
              'ground_truth': expected or None, 'comparison': correctness,
              'raw_model_output': raw, 'inference_mode': DTYPE}
    return result, correctness, raw, json.dumps(record, ensure_ascii=False)


def select_example(idx):
    if idx is None or str(idx).strip() == '':
        raise gr.Error('Please select a held-out example first.')
    row = examples[int(idx)]
    opts = row['options']
    label = next((row['option_ids'][i] for i, val in enumerate(row['correct']) if val), '')
    return (row['asset_name'].replace('_', ' ').capitalize(), row['question'], str(len(opts)),
            *[opts[i] if i < len(opts) else '' for i in range(5)], label, gr.Tabs(selected='ask'))


def export_csv(log):
    if not log:
        return None
    d = json.loads(log)
    path = ROOT / 'prediction_export.csv'
    # Write one record, not a public server-side audit log.
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(d))
        writer.writeheader()
        writer.writerow(d)
    return str(path)


with gr.Blocks(title='Industrial Asset Health AI') as demo:
    gr.Markdown('# Industrial Asset Health AI\n'
                '### Failure–sensor multiple-choice reasoning with a fine-tuned Qwen2.5-7B model\n'
                'Research prototype. Not a validated industrial maintenance or safety system.')
    with gr.Tabs(selected='ask') as main_tabs:
      with gr.Tab('Ask a diagnostic question', id='ask'):
        asset = gr.Dropdown(choices=ASSETS, value='Electric motor', label='Asset')
        question = gr.Textbox(lines=3, label='Diagnostic question')
        n_choices = gr.Dropdown(choices=['2', '3', '4', '5'], value='5', label='Number of choices')
        inputs = [gr.Textbox(label=f'Option {x}') for x in LABELS]
        expected = gr.Textbox(label='Ground truth label (optional)', visible=False)
        with gr.Row():
            run = gr.Button('Analyze question', variant='primary')
            clear = gr.Button('Clear')
        result = gr.Textbox(label='Fine-tuned model prediction')
        correctness = gr.Textbox(label='Benchmark comparison (when example loaded)')
        raw = gr.Textbox(label='Raw model output')
        record = gr.State('')
        export = gr.Button('Export prediction as CSV')
        outfile = gr.File(label='CSV download')
        run.click(fn=predict, inputs=[asset,question,n_choices,*inputs,expected],
                  outputs=[result,correctness,raw,record])
        export.click(fn=export_csv, inputs=record, outputs=outfile)
        clear.click(fn=lambda: ('', *(['']*5), '', '', '', ''),
                    inputs=None, outputs=[question,*inputs,result,correctness,raw,record])
      with gr.Tab('Try a held-out example', id='examples'):
        gr.Markdown('Select a sample question to populate the Ask tab. Ground truth is shown only for the original example.')
        example_index = gr.Dropdown(
            choices=[(f"#{i + 1} — {r['asset_name']} — {r['question'][:72]}", str(i)) for i,r in enumerate(examples)],
            value='0', label='Sample example')
        load = gr.Button('Load example into question form')
        load.click(fn=select_example, inputs=example_index,
                   outputs=[asset,question,n_choices,*inputs,expected,main_tabs])
      with gr.Tab('Research benchmark', id='benchmark'):
        gr.Markdown('### Published results on the original held-out test set (534 questions)')
        gr.Markdown('| Approach | Test accuracy |\n|---|---:|\n'
                    '| QLoRA fine-tuned Qwen2.5-7B | **88.01%** |\n'
                    '| Hybrid retrieval | **73.41%** |\n'
                    '| Zero-shot Qwen2.5-7B | 55.43% |\n'
                    '| Few-shot Qwen2.5-7B | 54.49% |\n\n'
                    'These are historical benchmark measurements, **not** live runtime performance. '
                    'This initial ZeroGPU app runs only the fine-tuned LLM; the original hybrid retrieval '
                    'implementation remains available in your local Streamlit application.\n\n'
                    '**Robustness caution:** QLoRA accuracy was 43.07% on the simple perturbed set '
                    'and 40.82% on the complex perturbed set.')
    gr.Markdown(f'Inference precision: `{DTYPE}`. '
                'The original benchmark used 4-bit NF4. Float16 inference may produce different predictions.')

if __name__ == '__main__':
    demo.queue(max_size=12, default_concurrency_limit=1).launch()
