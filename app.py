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
import time
from pathlib import Path

import spaces  # import before torch/CUDA initialization
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from retrieval_runtime import get_index

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
                       if expected and expected in LABELS[:n] else 'No ground truth supplied')
    record = {'asset': asset, 'question': question, 'options': choices,
              'predicted_label': label, 'predicted_answer': choices[LABELS.index(label)] if label else None,
              'ground_truth': expected or None, 'comparison': correctness,
              'raw_model_output': raw, 'inference_mode': DTYPE}
    return result, correctness, raw, json.dumps(record, ensure_ascii=False)


# Run CPU retrieval first, then allocate ZeroGPU for QLoRA.
def compare_retrieval(asset, question, n_choices, a, b, c, d, e, expected, task):
    n=int(n_choices)
    choices=[str(x or '').strip() for x in [a,b,c,d,e][:n]]
    question=(question or '').strip()
    if not question or any(not choice for choice in choices):
        raise gr.Error('Enter the question and all answer choices before comparing.')
    start=time.perf_counter()
    try:
        retrieval=get_index().predict({
            'asset_name':str(asset).strip().lower(),'question':question,
            'options':choices,'relevancy':task,
            'question_type':'mcp1_negative' if task.startswith('irrelevant_') else 'mcp1_positive',
        })
    except Exception as exc:
        raise gr.Error(f'Retrieval initialization or prediction failed: {exc}') from exc
    elapsed=time.perf_counter()-start
    label=retrieval['label']
    status=('Correct' if label==expected else 'Incorrect') if expected and expected in 'ABCDE'[:n] else 'No ground truth'
    record={'asset':asset,'question':question,'options':choices,'task':task,
            'ground_truth':expected if expected and expected in 'ABCDE'[:n] else None,
            'hybrid_label':label,'hybrid_answer':retrieval['answer'],
            'hybrid_correctness':status,'hybrid_elapsed_seconds':round(elapsed,3),
            'hybrid_option_scores':retrieval['details'],'hybrid_score_margin':retrieval['margin']}
    return (f"{label}. {retrieval['answer']}",status,f'{elapsed:.2f} s',
            [[i['option'],i['score'],i['support'],i['contradiction']] for i in retrieval['details']],
            json.dumps(record,ensure_ascii=False))


@spaces.GPU(duration=60)
def compare_qlora(asset, question, n_choices, a, b, c, d, e, expected, record_json):
    n=int(n_choices)
    choices=[str(x or '').strip() for x in [a,b,c,d,e][:n]]
    question=(question or '').strip()
    start=time.perf_counter()
    prompt=format_question(question,choices)
    batch=tokenizer(prompt,return_tensors='pt',truncation=True,max_length=768).to('cuda')
    with torch.inference_mode():
        output=model.generate(**batch,max_new_tokens=8,do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    elapsed=time.perf_counter()-start
    raw=tokenizer.decode(output[0,batch['input_ids'].shape[1]:],skip_special_tokens=True).strip()
    label=parse_label(raw,choices)
    record=json.loads(record_json)
    status=(('Correct' if label==expected else 'Incorrect') if expected and expected in 'ABCDE'[:n] and label else 'No ground truth / invalid')
    record.update({'qlora_label':label,'qlora_answer':choices['ABCDE'.index(label)] if label else None,
                   'qlora_raw_output':raw,'qlora_elapsed_seconds':round(elapsed,3),
                   'qlora_correctness':status,'inference_precision':DTYPE})
    agree='Agree' if label is not None and label==record['hybrid_label'] else 'Disagree' if label is not None else 'Invalid LLM output'
    record['method_agreement']=agree
    result=f"{label}. {record['qlora_answer']}" if label else f'Invalid model output: {raw!r}'
    return result,status,f'{elapsed:.2f} s',agree,json.dumps(record,ensure_ascii=False)


def export_comparison(record_json):
    if not record_json: return None
    import tempfile
    rec=json.loads(record_json)
    with tempfile.NamedTemporaryFile(mode='w',suffix='.csv',prefix='asset_ai_comparison_',
                                     delete=False,newline='',encoding='utf-8') as f:
        data={k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in rec.items()}
        writer=csv.DictWriter(f,fieldnames=list(data));writer.writeheader();writer.writerow(data)
        return f.name


def select_example(idx):
    if idx is None or str(idx).strip() == '':
        raise gr.Error('Please select a held-out example first.')
    row = examples[int(idx)]
    opts = row['options']
    label = next((row['option_ids'][i] for i, val in enumerate(row['correct']) if val), '')
    return (row['asset_name'].replace('_', ' ').capitalize(), row['question'], str(len(opts)),
            *[opts[i] if i < len(opts) else '' for i in range(5)], label, row['relevancy'], gr.Tabs(selected='ask'))


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
        task = gr.Dropdown(
            choices=[('Sensor relevant to failure','relevant_sensors_for_failure_mode'),
                     ('Failure mode relevant to sensor','relevant_failure_modes_for_sensor'),
                     ('Sensor irrelevant to failure','irrelevant_sensors_for_failure_mode'),
                     ('Failure mode irrelevant to sensor','irrelevant_failure_modes_for_sensor')],
            value='relevant_sensors_for_failure_mode',label='Question type for hybrid retrieval',
            info='Held-out examples set this automatically. For manual questions choose the appropriate task.')
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
        gr.Markdown('### Compare both research methods\nThe hybrid model runs on CPU first; QLoRA then runs on ZeroGPU. The first retrieval request builds a CPU index and may take longer.')
        compare = gr.Button('Compare QLoRA and hybrid retrieval', variant='primary')
        with gr.Row():
            with gr.Column():
                gr.Markdown('#### Hybrid retrieval (CPU)')
                hybrid_answer=gr.Textbox(label='Predicted answer',interactive=False)
                hybrid_correct=gr.Textbox(label='Ground-truth comparison',interactive=False)
                hybrid_time=gr.Textbox(label='Elapsed time',interactive=False)
            with gr.Column():
                gr.Markdown('#### Fine-tuned QLoRA (ZeroGPU)')
                qlora_answer=gr.Textbox(label='Predicted answer',interactive=False)
                qlora_correct=gr.Textbox(label='Ground-truth comparison',interactive=False)
                qlora_time=gr.Textbox(label='Elapsed time',interactive=False)
        agreement=gr.Textbox(label='Method agreement',interactive=False)
        gr.Markdown('Hybrid retrieval per-option scores (not probabilities)')
        option_scores=gr.Dataframe(headers=['Option','Final score','Support','Contradiction'],
                                   datatype=['str','number','number','number'],interactive=False)
        comparison_log=gr.State('')
        comparison_export=gr.Button('Download comparison CSV')
        comparison_download=gr.File(label='Comparison CSV')
        comparison_event=compare.click(
            fn=compare_retrieval,
            inputs=[asset,question,n_choices,*inputs,expected,task],
            outputs=[hybrid_answer,hybrid_correct,hybrid_time,option_scores,comparison_log],
        )
        comparison_event.success(
            fn=compare_qlora,
            inputs=[asset,question,n_choices,*inputs,expected,comparison_log],
            outputs=[qlora_answer,qlora_correct,qlora_time,agreement,comparison_log],
        )
        comparison_export.click(fn=export_comparison,inputs=comparison_log,outputs=comparison_download)
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
                   outputs=[asset,question,n_choices,*inputs,expected,task,main_tabs])
      with gr.Tab('Research benchmark', id='benchmark'):
        gr.Markdown('### Published results on the original held-out test set (534 questions)')
        gr.Markdown('| Approach | Test accuracy |\n|---|---:|\n'
                    '| QLoRA fine-tuned Qwen2.5-7B | **88.01%** |\n'
                    '| Hybrid retrieval | **73.41%** |\n'
                    '| Zero-shot Qwen2.5-7B | 55.43% |\n'
                    '| Few-shot Qwen2.5-7B | 54.49% |\n\n'
                    'These are historical benchmark measurements, **not** live runtime performance. '
                    'Both approaches can now be compared live in the Ask tab.\n\n'
                    '**Robustness caution:** QLoRA accuracy was 43.07% on the simple perturbed set '
                    'and 40.82% on the complex perturbed set.')
    
    with gr.Tab("About This Project"):
        gr.Markdown("""
## Industrial Asset Health AI

### Project Overview

Industrial Asset Health AI is a research-based web application
for answering multiple-choice diagnostic questions related to
industrial equipment failures and sensor measurements.

The application compares two approaches:

**1. Fine-tuned Large Language Model (QLoRA)**

- Base model: Qwen2.5-7B-Instruct
- Fine-tuning: Quantized Low-Rank Adaptation (QLoRA)
- Task: Select the most relevant answer for an industrial
  failure-sensor diagnostic question

**2. Hybrid Retrieval**

- Combines TF-IDF and semantic similarity (MiniLM)
- Evaluates each candidate answer independently
- Uses supporting and contradicting examples from training data
- Selects the answer with the highest retrieval score

### Research Performance

Evaluated on 534 held-out questions from FailureSensorIQ.

| Method | Test Accuracy |
|---|---:|
| Fine-tuned Qwen2.5-7B (QLoRA) | 88.01% |
| Hybrid Retrieval | 73.41% |

These are the original research benchmark results,
not live measurements from the deployed application.

### Application Features

- Enter diagnostic questions with 2–5 answer choices
- Load held-out examples from FailureSensorIQ
- Compare QLoRA and hybrid retrieval predictions
- View predicted answers and inference times
- Download comparison results as CSV

### Technologies

Python, PyTorch, Hugging Face Transformers, PEFT,
QLoRA, Sentence Transformers, Scikit-learn,
Gradio, and Hugging Face ZeroGPU.

### Limitations

- This is a research prototype, not a validated industrial
  fault-diagnosis or safety system.
- Performance can decrease when questions are paraphrased,
  answer options are shuffled, or new distractors are introduced.
- The model provides an answer choice, not a verified
  causal explanation.
- The deployed LLM uses FP16 inference, while the original
  research benchmark used 4-bit NF4 inference.
- GPU inference is subject to ZeroGPU queues and usage limits.

**Important:** Predictions should be verified by qualified
engineers before being used for operational decisions.

### Live Application

[Industrial Asset Health AI on Hugging Face](
https://huggingface.co/spaces/dseshukumar/industrial-asset-health-ai
)
        """)

    gr.Markdown(f'Inference precision: `{DTYPE}`. '
                'The original benchmark used 4-bit NF4. Float16 inference may produce different predictions.')

if __name__ == '__main__':
    demo.queue(max_size=12, default_concurrency_limit=1).launch()
