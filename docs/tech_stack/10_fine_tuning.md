# 10. Fine-Tuning: LoRA, QLoRA, Hugging Face & Dataset Preparation

> Interview preparation guide for Backend AI Engineers.
> Covers theory, visual diagrams, practical Python code, and 25+ Q&A.

---

## Table of Contents

1. [Fine-Tuning Overview](#1-fine-tuning-overview)
2. [Full Fine-Tuning](#2-full-fine-tuning)
3. [LoRA (Low-Rank Adaptation)](#3-lora-low-rank-adaptation)
4. [QLoRA (Quantized LoRA)](#4-qlora-quantized-lora)
5. [Hugging Face Ecosystem](#5-hugging-face-ecosystem)
6. [Dataset Preparation](#6-dataset-preparation)
7. [Training Process](#7-training-process)
8. [Evaluation](#8-evaluation)
9. [Model Merging & Deployment](#9-model-merging--deployment)
10. [Advanced Topics](#10-advanced-topics)
11. [Q&A Section (25 Questions)](#11-qa-section)

---

## 1. Fine-Tuning Overview

### 1.1 What Is Fine-Tuning?

Fine-tuning is the process of taking a **pre-trained** language model and continuing its training
on a smaller, domain-specific or task-specific dataset so the model adapts its behavior to that
particular domain or task.

```
Pre-trained Model                        Fine-tuned Model
(trillions of tokens,                    (thousands-millions of examples,
 general knowledge)                       specialized knowledge)
┌──────────────────┐                     ┌──────────────────┐
│  Knows language,  │   + task-specific   │  Retains general  │
│  grammar, facts,  │ ──── dataset ────►  │  knowledge AND    │
│  reasoning        │      (fine-tune)    │  excels at task   │
└──────────────────┘                     └──────────────────┘
```

### 1.2 Fine-Tuning vs Prompt Engineering vs RAG

| Aspect | Prompt Engineering | RAG | Fine-Tuning |
|---|---|---|---|
| **What changes** | Only the input prompt | Retrieval pipeline + prompt | Model weights |
| **Training needed** | No | No (but needs index) | Yes |
| **Data needed** | Few examples (0-shot, few-shot) | A document corpus | Hundreds to thousands of examples |
| **Cost** | Low (API calls) | Medium (embeddings + vector DB) | High (GPU compute) |
| **Latency at inference** | Low | Medium (retrieval step) | Low |
| **Knowledge freshness** | Static (model cutoff) | Dynamic (update docs) | Static (retrain needed) |
| **Best for** | General tasks, quick iteration | Knowledge-grounded Q&A | Style/format control, specialized reasoning |

### 1.3 Decision Tree: When to Use What

```
                         Your Task
                            │
                            ▼
              ┌─────────────────────────────┐
              │ Can prompt engineering       │
              │ (zero-shot / few-shot)       │
              │ solve it adequately?         │
              └──────────┬──────────────────┘
                    Yes  │           No
                    │    │            │
                    ▼    │            ▼
              Use Prompting    ┌────────────────────────┐
                               │ Do you need up-to-date  │
                               │ or external knowledge?   │
                               └──────┬─────────────────┘
                                 Yes  │          No
                                 │    │           │
                                 ▼    │           ▼
                             Use RAG      ┌─────────────────────┐
                                          │ Do you have enough   │
                                          │ labeled data         │
                                          │ (500+ examples)?     │
                                          └──────┬──────────────┘
                                            Yes  │         No
                                            │    │          │
                                            ▼    │          ▼
                                       Fine-Tune    Use RAG or collect
                                                    more data first
```

### 1.4 When to Fine-Tune

**Good reasons to fine-tune:**
- You need a specific output **format** or **style** consistently
- You need the model to follow complex domain-specific **instructions**
- You have a well-defined task with hundreds/thousands of quality examples
- You need to reduce latency (replace multi-step prompts with a single call)
- You need to reduce cost (smaller fine-tuned model can replace larger prompted model)
- You need specialized vocabulary or terminology

**When NOT to fine-tune:**
- A good prompt already achieves your goal
- Your data is fewer than ~100 high-quality examples
- You need factual grounding over a large corpus (use RAG)
- Your knowledge changes frequently (RAG is better)
- You lack the compute budget or infrastructure

### 1.5 Types of Fine-Tuning

```
Fine-Tuning Approaches
│
├── Full Fine-Tuning
│   └── Update ALL model parameters
│       (expensive, risk of catastrophic forgetting)
│
├── Parameter-Efficient Fine-Tuning (PEFT)
│   ├── LoRA / QLoRA          ← most popular
│   ├── Prefix Tuning
│   ├── Prompt Tuning (soft prompts)
│   ├── (IA)^3
│   └── AdaLoRA
│
└── Adapter-Based
    ├── Bottleneck Adapters
    └── Parallel Adapters
```

### 1.6 Cost and Compute Considerations

| Model Size | Full FT (fp32) VRAM | Full FT (fp16) VRAM | LoRA (fp16) VRAM | QLoRA (4-bit) VRAM |
|---|---|---|---|---|
| 1.3B | ~5 GB | ~3 GB | ~3 GB | ~1.5 GB |
| 7B | ~28 GB | ~14 GB | ~14 GB | ~4 GB |
| 13B | ~52 GB | ~26 GB | ~26 GB | ~8 GB |
| 70B | ~280 GB | ~140 GB | ~140 GB | ~36 GB |

> **Rule of thumb for training VRAM**: approximately 4x model size in fp32, 2x in fp16 (because
> of optimizer states, gradients, and activations in addition to the weights themselves).

---

## 2. Full Fine-Tuning

### 2.1 How It Works

Full fine-tuning updates **every parameter** in the model. For a 7B parameter model, that means
7 billion floating-point numbers are adjusted during backpropagation.

```
Full Fine-Tuning:
┌──────────────────────────────────────┐
│  ALL Parameters Updated              │
│  ████████████████████████████████████ │  ← Embedding layers
│  ████████████████████████████████████ │  ← Attention layers (Q, K, V, O)
│  ████████████████████████████████████ │  ← Feed-forward layers
│  ████████████████████████████████████ │  ← Layer norms
│  ████████████████████████████████████ │  ← Output head
└──────────────────────────────────────┘
   Total: 7B params = ~28 GB VRAM (fp32)
                       ~14 GB VRAM (fp16)
   + Optimizer states: ~2x model size (Adam)
   + Gradients:        ~1x model size
   + Activations:      variable
   ─────────────────────────────────────
   Total training:     ~56-84 GB for 7B in fp32
```

### 2.2 Catastrophic Forgetting

When you fine-tune on a narrow dataset, the model may **forget** its general capabilities.

```
Before Fine-Tuning:          After Aggressive Fine-Tuning:
┌──────────────────┐         ┌──────────────────┐
│ Math:     ████░  │         │ Math:     █░░░░  │  ← degraded
│ Code:     ████░  │         │ Code:     ██░░░  │  ← degraded
│ Writing:  ████░  │   ──►   │ Writing:  █░░░░  │  ← degraded
│ Medical:  ██░░░  │         │ Medical:  █████  │  ← improved
│ General:  ████░  │         │ General:  ██░░░  │  ← degraded
└──────────────────┘         └──────────────────┘
```

**Mitigation strategies:**
- Mix general data with domain data during fine-tuning
- Use lower learning rates
- Train for fewer epochs (1-3 is typical)
- Use PEFT methods (LoRA) which freeze most weights
- Regularization techniques (weight decay, dropout)

### 2.3 When Full Fine-Tuning Makes Sense

- You have **abundant** compute (multi-GPU cluster)
- You have a **large** high-quality dataset (tens of thousands+ examples)
- You need **maximum** task performance and can afford the compute
- You plan to distill the model afterward
- You are creating a foundational model for a specific domain

### 2.4 Code Example: Full Fine-Tuning

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# --- Load model and tokenizer ---
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # fp16 to halve memory
    device_map="auto",          # spread across available GPUs
)

# --- Load and tokenize dataset ---
dataset = load_dataset("json", data_files="train.jsonl", split="train")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=2048,
        padding="max_length",
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# --- Training ---
training_args = TrainingArguments(
    output_dir="./full-ft-output",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # effective batch size = 2 * 8 = 16
    learning_rate=2e-5,              # lower LR for full fine-tuning
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    report_to="wandb",               # optional: Weights & Biases logging
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
model.save_pretrained("./full-ft-final")
tokenizer.save_pretrained("./full-ft-final")
```

---

## 3. LoRA (Low-Rank Adaptation)

### 3.1 Core Idea

LoRA (Hu et al., 2021) is based on the insight that weight updates during fine-tuning have
**low intrinsic rank**. Instead of updating a full weight matrix `W`, LoRA decomposes the update
into two smaller matrices.

**Key principle:** Freeze the original pre-trained weights and inject small, trainable
low-rank matrices alongside them.

### 3.2 Mathematical Explanation

For a pre-trained weight matrix `W_0` of dimensions `(d x d)`:

```
Standard fine-tuning:
    W = W_0 + DeltaW          where DeltaW is (d x d)

LoRA fine-tuning:
    W = W_0 + DeltaW
    W = W_0 + B * A            where A is (d x r) and B is (r x d), r << d

The forward pass becomes:
    h = W_0 * x + (B * A) * x
    h = W_0 * x + B * (A * x)   ← compute A*x first (cheaper)
```

### 3.3 Parameter Savings

```
Example: d = 4096, r = 8

Original weight matrix W:
    Parameters = d x d = 4096 x 4096 = 16,777,216  (16.7M)

LoRA matrices A and B:
    A: d x r = 4096 x 8 =  32,768
    B: r x d = 8 x 4096 =  32,768
    Total:                  65,536   (65K)

Ratio: 65,536 / 16,777,216 = 0.39%  (< 0.4% of original!)

Across an entire 7B model (applying LoRA to attention layers):
    Trainable params:  ~4-20M   (depending on rank and target modules)
    Frozen params:     ~7,000M
    Trainable %:       ~0.1 - 0.3%
```

### 3.4 Visual Architecture

```
                    LoRA Applied to One Linear Layer
                    ─────────────────────────────────

                         x (input)
                         │
                ┌────────┴────────┐
                │                 │
                ▼                 ▼
     ┌──────────────────┐   ┌──────────┐
     │   W_0 (frozen)   │   │  A (down) │  d --> r  (trainable)
     │   d x d           │   │  d x r    │
     │   (original       │   └────┬─────┘
     │    weights)       │        │
     └────────┬─────────┘        ▼
              │             ┌──────────┐
              │             │  B (up)   │  r --> d  (trainable)
              │             │  r x d    │
              │             └────┬─────┘
              │                  │
              │    scaling:      │
              │    alpha / r     │
              │                  │
              └───────┬──────────┘
                      │ (element-wise add)
                      ▼
                   h (output)

    h = W_0 * x  +  (alpha/r) * B * A * x
```

### 3.5 Key Hyperparameters

| Parameter | Description | Typical Values | Effect |
|---|---|---|---|
| **r (rank)** | Rank of decomposition matrices | 4, 8, 16, 32, 64 | Higher = more capacity, more params |
| **alpha** | Scaling factor for LoRA updates | Usually 2*r (e.g., 16, 32, 64) | Controls magnitude of the update |
| **target_modules** | Which layers get LoRA | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj | More modules = more capacity |
| **dropout** | Dropout on LoRA layers | 0.0 - 0.1 | Regularization |

**Scaling behavior:**
```
Effective scaling = alpha / r

If r=8, alpha=16:   scaling = 16/8 = 2.0
If r=16, alpha=32:  scaling = 32/16 = 2.0   (same effective scaling)
If r=8, alpha=8:    scaling = 8/8 = 1.0      (standard, no extra scaling)

Common practice: set alpha = 2 * r, so effective scaling = 2.0
```

### 3.6 Which Layers to Target

```
Transformer Block (typical LLaMA-style):
├── Self-Attention
│   ├── q_proj  ← commonly targeted
│   ├── k_proj  ← commonly targeted
│   ├── v_proj  ← commonly targeted
│   └── o_proj  ← commonly targeted
│
├── MLP / Feed-Forward
│   ├── gate_proj  ← sometimes targeted
│   ├── up_proj    ← sometimes targeted
│   └── down_proj  ← sometimes targeted
│
└── Layer Norms (usually NOT targeted by LoRA)

Typical configurations:
  Minimal:   ["q_proj", "v_proj"]                    ← fast, lightweight
  Standard:  ["q_proj", "k_proj", "v_proj", "o_proj"] ← good balance
  Full:      all linear layers                        ← max capacity
```

**Guidelines:**
- Start with `q_proj` and `v_proj` (original LoRA paper)
- Add `k_proj` and `o_proj` for more capacity
- Add MLP layers (`gate_proj`, `up_proj`, `down_proj`) for complex tasks
- More target modules = more trainable parameters = more VRAM

### 3.7 Code Example: LoRA with PEFT

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# --- Load base model ---
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# --- Define LoRA configuration ---
lora_config = LoraConfig(
    r=16,                                          # rank
    lora_alpha=32,                                 # scaling factor
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention layers
        "gate_proj", "up_proj", "down_proj",       # MLP layers
    ],
    lora_dropout=0.05,
    bias="none",                                   # don't train bias terms
    task_type=TaskType.CAUSAL_LM,
)

# --- Apply LoRA to the model ---
model = get_peft_model(model, lora_config)

# --- Inspect trainable parameters ---
model.print_trainable_parameters()
# Output: trainable params: 13,631,488 || all params: 6,751,318,016
#         || trainable%: 0.2019%

# --- Save and load LoRA adapter (small file!) ---
model.save_pretrained("./lora-adapter")
# Saved adapter is typically 10-50 MB, NOT 14 GB

# --- Load adapter later ---
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./lora-adapter")
```

### 3.8 LoRA Initialization

```
Matrix A: Initialized with Kaiming uniform (small random values)
Matrix B: Initialized with zeros

Why zeros for B?
    DeltaW = B * A
    If B = 0, then DeltaW = 0 at start
    The model starts identical to the pre-trained model
    Training gradually learns the update from this starting point

This is important: it guarantees no performance degradation at initialization.
```

---

## 4. QLoRA (Quantized LoRA)

### 4.1 What Is QLoRA?

QLoRA (Dettmers et al., 2023) combines **4-bit quantization** of the base model with LoRA
adapters trained in higher precision (fp16/bf16). This dramatically reduces memory requirements.

```
Standard LoRA:
┌─────────────────────────────────────┐
│  Base Model Weights (fp16)          │  ← 14 GB for 7B model
│  ████████████████████████████████   │
│                                     │
│  LoRA Adapters (fp16)               │  ← ~20 MB
│  ██                                 │
└─────────────────────────────────────┘
Total VRAM: ~14 GB + optimizer states

QLoRA:
┌─────────────────────────────────────┐
│  Base Model Weights (4-bit NF4)     │  ← ~3.5 GB for 7B model
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │  (frozen, quantized)
│                                     │
│  LoRA Adapters (fp16/bf16)          │  ← ~20 MB
│  ██                                 │  (trainable, full precision)
└─────────────────────────────────────┘
Total VRAM: ~4-6 GB + optimizer states for adapters only
```

### 4.2 Three Key Innovations

#### Innovation 1: NF4 (Normal Float 4-bit) Quantization

```
Standard 4-bit quantization:
    Equally spaced quantization levels across the range
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    Problem: Neural network weights follow a normal distribution,
             not a uniform one. Wastes precision in sparse tails.

NF4 quantization:
    Quantization levels optimized for normally-distributed data
    More levels near zero (where most weights live),
    fewer in the tails

    Normal distribution of weights:
            ▲
          ██│██
         ███│███
        ████│████
       █████│█████
      ██████│██████
    ─────────────────►
    NF4 levels: |||||||  ||  |    |       (more levels near center)
    Uniform:    |   |   |   |   |   |    (evenly spaced)
```

#### Innovation 2: Double Quantization

Quantization itself requires storing **quantization constants** (scale factors). Double
quantization quantizes these constants too, saving additional memory.

```
Single quantization:
    Weights (4-bit) + quantization constants (fp32)
    Constants: 1 per block of 64 weights = fp32 overhead

Double quantization:
    Weights (4-bit) + quantization constants (8-bit, quantized!)
    Saves ~0.37 bits per parameter
    For 7B model: saves ~0.37 * 7B / 8 = ~325 MB
```

#### Innovation 3: Paged Optimizers

Uses NVIDIA unified memory to handle GPU memory spikes by paging optimizer states to CPU RAM
when GPU memory is full (similar to virtual memory in operating systems).

```
Normal training (memory spike during gradient step):
    GPU: [model] [gradients] [optimizer] ← OOM!

Paged optimizers:
    GPU: [model] [gradients] [partial optimizer]
    CPU: [overflow optimizer states]    ← automatically paged

    When GPU has space again, pages are moved back.
```

### 4.3 Memory Comparison

```
Memory Requirements for Training a 7B Model:
─────────────────────────────────────────────

Full Fine-Tuning (fp32):
    Model:      28 GB
    Optimizer:  56 GB (Adam: 2 copies)
    Gradients:  28 GB
    Activations: ~8 GB
    ──────────────────
    Total:     ~120 GB  →  needs 2-4x A100 80GB

Full Fine-Tuning (fp16 mixed precision):
    Model:      14 GB
    Optimizer:  28 GB
    Gradients:  14 GB
    Activations: ~4 GB
    ──────────────────
    Total:     ~60 GB   →  needs 1x A100 80GB

LoRA (fp16):
    Model:      14 GB   (frozen, but still in memory)
    Optimizer:  ~40 MB  (only for LoRA params)
    Gradients:  ~20 MB  (only for LoRA params)
    Activations: ~4 GB
    ──────────────────
    Total:     ~18 GB   →  needs 1x A100 40GB or RTX 3090

QLoRA (4-bit + fp16 adapters):
    Model:      3.5 GB  (4-bit quantized)
    Optimizer:  ~40 MB  (only for LoRA params)
    Gradients:  ~20 MB  (only for LoRA params)
    Activations: ~2 GB
    ──────────────────
    Total:     ~6 GB    →  fits on RTX 3060/4060 (12GB)!
```

### 4.4 Code Example: QLoRA Fine-Tuning

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# --- 4-bit quantization configuration ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                  # use 4-bit quantization
    bnb_4bit_quant_type="nf4",          # NF4 data type (better than fp4)
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bf16
    bnb_4bit_use_double_quant=True,     # double quantization
)

# --- Load quantized model ---
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- Prepare model for k-bit training ---
# Handles gradient checkpointing and layer norm casting
model = prepare_model_for_kbit_training(model)

# --- LoRA configuration ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- Apply LoRA ---
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 13,631,488 || all params: 3,540,389,888 || trainable%: 0.3850

# --- Load dataset ---
dataset = load_dataset("json", data_files="train.jsonl", split="train")

# --- Formatting function (chat template) ---
def format_instruction(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example.get('input', '')}

### Response:
{example['output']}"""

# --- Train with SFTTrainer ---
training_args = SFTConfig(
    output_dir="./qlora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.001,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,                         # use bf16 for training computations
    max_seq_length=2048,
    gradient_checkpointing=True,       # save memory at cost of speed
    optim="paged_adamw_8bit",          # paged optimizer
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    formatting_func=format_instruction,
    args=training_args,
    tokenizer=tokenizer,
)

trainer.train()

# --- Save the LoRA adapter ---
trainer.save_model("./qlora-adapter")
```

### 4.5 QLoRA vs LoRA: When to Choose Which

| Scenario | Choose |
|---|---|
| Have A100 or better GPUs | LoRA (fp16) -- slightly better quality |
| Consumer GPU (RTX 3060/4060, 12-24 GB) | QLoRA -- fits in memory |
| Large model (70B+) | QLoRA -- often the only practical option |
| Production quality is critical | LoRA (fp16) then quantize for serving |
| Experimentation / prototyping | QLoRA -- fast iteration on cheap hardware |

---

## 5. Hugging Face Ecosystem

### 5.1 Ecosystem Overview

```
Hugging Face Ecosystem for Fine-Tuning
───────────────────────────────────────

┌─────────────────────────────────────────────────────────┐
│                   Hugging Face Hub                       │
│  (Models, Datasets, Spaces -- sharing & collaboration)   │
└──────┬──────────────────────────────────┬───────────────┘
       │                                  │
       ▼                                  ▼
┌─────────────┐    ┌──────────┐    ┌─────────────┐
│ transformers │    │ datasets  │    │    PEFT      │
│ (models &    │    │ (loading, │    │ (LoRA,       │
│  tokenizers) │    │  process) │    │  adapters)   │
└──────┬──────┘    └─────┬────┘    └──────┬──────┘
       │                 │                │
       └────────┬────────┘                │
                │                         │
                ▼                         │
         ┌────────────┐                   │
         │    TRL      │◄─────────────────┘
         │ (SFTTrainer,│
         │  DPOTrainer,│
         │  PPOTrainer)│
         └──────┬─────┘
                │
       ┌────────┴────────┐
       │                 │
       ▼                 ▼
┌────────────┐    ┌──────────────┐
│ accelerate  │    │ bitsandbytes │
│ (multi-GPU, │    │ (4-bit, 8-bit│
│  distributed│    │  quantization│
│  training)  │    │  )           │
└─────────────┘    └──────────────┘
```

### 5.2 transformers Library

The core library for loading and using pre-trained models.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Load any model from the Hub ---
model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",           # auto-shard across GPUs
    attn_implementation="flash_attention_2",  # use Flash Attention
)

# --- Generate text ---
inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# --- Key Auto classes ---
# AutoModelForCausalLM        - decoder-only (GPT, LLaMA, Mistral)
# AutoModelForSeq2SeqLM       - encoder-decoder (T5, BART)
# AutoModelForSequenceClassification  - classification head
# AutoModelForTokenClassification     - NER, POS tagging
# AutoTokenizer                - loads the right tokenizer automatically
```

### 5.3 datasets Library

```python
from datasets import load_dataset, Dataset, DatasetDict

# --- Load from Hugging Face Hub ---
dataset = load_dataset("tatsu-lab/alpaca")
# DatasetDict({
#     train: Dataset({features: ['instruction','input','output'], num_rows: 52002})
# })

# --- Load from local files ---
dataset = load_dataset("json", data_files="data.jsonl", split="train")
dataset = load_dataset("csv", data_files="data.csv", split="train")
dataset = load_dataset("parquet", data_files="data.parquet", split="train")

# --- Create from Python dict ---
dataset = Dataset.from_dict({
    "instruction": ["Summarize this", "Translate to French"],
    "input": ["Long article...", "Hello world"],
    "output": ["Summary...", "Bonjour le monde"],
})

# --- Common operations ---
dataset = dataset.shuffle(seed=42)
dataset = dataset.select(range(1000))           # first 1000 rows
dataset = dataset.filter(lambda x: len(x["output"]) > 10)
dataset = dataset.map(tokenize_function, batched=True)
dataset = dataset.train_test_split(test_size=0.1)

# --- Streaming for large datasets ---
dataset = load_dataset("cerebras/SlimPajama-627B", streaming=True)
for example in dataset["train"]:
    process(example)
    break  # just demonstrating
```

### 5.4 PEFT Library

```python
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM,       # convenience class
)

# --- Create LoRA config ---
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# --- Apply to model ---
model = get_peft_model(base_model, config)

# --- Save adapter (lightweight) ---
model.save_pretrained("./my-adapter")  # saves only adapter weights + config

# --- Load adapter on top of base model ---
model = AutoPeftModelForCausalLM.from_pretrained(
    "./my-adapter",
    torch_dtype=torch.float16,
    device_map="auto",
)

# --- Merge adapter into base model (for deployment) ---
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
```

### 5.5 TRL (Transformer Reinforcement Learning)

```python
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig

# SFTTrainer: Supervised Fine-Tuning (most common)
# DPOTrainer: Direct Preference Optimization
# PPOTrainer: Proximal Policy Optimization (RLHF)
# RewardTrainer: Train reward models

# --- SFTTrainer handles tokenization, formatting, packing ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,       # optional: applies PEFT automatically
    formatting_func=format_fn,      # or use dataset_text_field="text"
    args=SFTConfig(
        output_dir="./output",
        max_seq_length=2048,
        packing=True,               # pack multiple samples into one sequence
    ),
)
```

### 5.6 Accelerate (Distributed Training)

```python
# accelerate handles multi-GPU, mixed precision, DeepSpeed, FSDP

# Configuration (run in terminal):
# $ accelerate config    (interactive setup)
# $ accelerate launch train.py

# In code:
from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="bf16")
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

### 5.7 BitsAndBytes (Quantization)

```python
from transformers import BitsAndBytesConfig

# --- 4-bit quantization ---
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # or "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- 8-bit quantization ---
bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config_4bit,
    device_map="auto",
)
```

---

## 6. Dataset Preparation

### 6.1 Data Formats

#### Instruction Format (Alpaca-style)

```json
{
    "instruction": "Classify the sentiment of the following review.",
    "input": "The food was absolutely delicious and the service was outstanding!",
    "output": "Positive"
}
```

#### Chat / Conversational Format

```json
{
    "messages": [
        {"role": "system", "content": "You are a helpful medical assistant."},
        {"role": "user", "content": "What are the symptoms of the flu?"},
        {"role": "assistant", "content": "Common flu symptoms include fever, cough, sore throat, body aches, headache, chills, and fatigue."}
    ]
}
```

#### ShareGPT Format (Multi-turn)

```json
{
    "conversations": [
        {"from": "human", "value": "What is Python?"},
        {"from": "gpt", "value": "Python is a high-level programming language..."},
        {"from": "human", "value": "Show me a hello world example."},
        {"from": "gpt", "value": "Here is a simple example:\n```python\nprint('Hello, World!')\n```"}
    ]
}
```

#### Completion Format (Plain Text)

```text
<s>[INST] What is the capital of France? [/INST] The capital of France is Paris.</s>
```

### 6.2 Data Quality Principles

```
Data Quality Hierarchy:
────────────────────────

         ▲
        /│\      Quality >> Quantity
       / │ \
      /  │  \    1,000 high-quality examples
     /   │   \   often beat 100,000 low-quality ones
    /    │    \
   / Accuracy  \
  / Diversity   \
 / Consistency   \
/ Relevance       \
──────────────────────

Key principles:
  1. Accuracy:    Outputs must be factually correct
  2. Diversity:   Cover the full range of expected inputs
  3. Consistency: Follow a uniform format and style
  4. Relevance:   Data should match the target task
  5. Completeness: Outputs should be thorough
```

### 6.3 Data Cleaning Pipeline

```python
import re
from datasets import load_dataset, Dataset

def clean_dataset(dataset):
    """Comprehensive data cleaning pipeline."""

    def clean_example(example):
        # 1. Strip whitespace
        for key in ["instruction", "input", "output"]:
            if key in example and example[key]:
                example[key] = example[key].strip()

        # 2. Remove excessive whitespace
        for key in ["instruction", "input", "output"]:
            if key in example and example[key]:
                example[key] = re.sub(r'\s+', ' ', example[key])

        return example

    def filter_quality(example):
        # 3. Remove empty outputs
        if not example.get("output") or len(example["output"].strip()) < 5:
            return False

        # 4. Remove too-short instructions
        if not example.get("instruction") or len(example["instruction"].strip()) < 10:
            return False

        # 5. Remove too-long examples (may cause OOM)
        total_len = len(example.get("instruction", "")) + \
                    len(example.get("input", "")) + \
                    len(example.get("output", ""))
        if total_len > 10000:
            return False

        # 6. Filter out low-quality patterns
        low_quality_patterns = [
            r"I don't know",
            r"As an AI",
            r"I cannot",
            r"I'm sorry, but",
        ]
        for pattern in low_quality_patterns:
            if re.search(pattern, example.get("output", ""), re.IGNORECASE):
                return False

        return True

    # Apply cleaning and filtering
    dataset = dataset.map(clean_example)
    dataset = dataset.filter(filter_quality)

    return dataset


# --- Deduplication ---
def deduplicate(dataset, column="instruction"):
    """Remove duplicate entries based on a column."""
    seen = set()
    indices_to_keep = []

    for i, example in enumerate(dataset):
        text = example[column].lower().strip()
        if text not in seen:
            seen.add(text)
            indices_to_keep.append(i)

    return dataset.select(indices_to_keep)


# --- Usage ---
dataset = load_dataset("json", data_files="raw_data.jsonl", split="train")
print(f"Before cleaning: {len(dataset)} examples")

dataset = clean_dataset(dataset)
dataset = deduplicate(dataset)
print(f"After cleaning: {len(dataset)} examples")
```

### 6.4 Train/Validation/Test Split

```python
from datasets import DatasetDict

# --- Standard split ---
dataset = dataset.shuffle(seed=42)
split = dataset.train_test_split(test_size=0.1, seed=42)

# Further split test into validation and test
val_test = split["test"].train_test_split(test_size=0.5, seed=42)

final_dataset = DatasetDict({
    "train": split["train"],           # 90%
    "validation": val_test["train"],   # 5%
    "test": val_test["test"],          # 5%
})

print(final_dataset)
# DatasetDict({
#     train: Dataset({features: [...], num_rows: 9000})
#     validation: Dataset({features: [...], num_rows: 500})
#     test: Dataset({features: [...], num_rows: 500})
# })
```

### 6.5 Tokenization and Formatting

```python
def create_training_prompt(example, tokenizer):
    """Format example into a training prompt with proper chat template."""

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]

    # Use the tokenizer's built-in chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# Apply formatting
formatted_dataset = dataset.map(
    lambda x: create_training_prompt(x, tokenizer),
    remove_columns=dataset.column_names,
)

# Tokenize
def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=2048,
        padding=False,       # dynamic padding is more efficient
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = formatted_dataset.map(tokenize, batched=True)
```

### 6.6 Data Augmentation Techniques

```
Technique                  Description                           When to Use
────────────────────────   ────────────────────────────────────   ──────────────────
Paraphrasing               Rephrase instructions differently     Small datasets
Back-translation           Translate to another language & back   Multilingual tasks
Synonym replacement        Swap words with synonyms              Simple augmentation
Self-instruct              Use LLM to generate more examples     Bootstrap from few examples
Evol-Instruct              Progressively increase complexity     Complex reasoning tasks
Template variation         Vary the instruction template         Format robustness
Few-shot to zero-shot      Remove examples from few-shot data    Generalization
```

```python
# Example: Using an LLM to augment data (self-instruct style)
from openai import OpenAI

client = OpenAI()

def augment_with_llm(seed_examples, num_new=100):
    """Generate new training examples using an LLM."""
    augmented = []

    seed_text = "\n".join([
        f"Instruction: {ex['instruction']}\nOutput: {ex['output']}"
        for ex in seed_examples[:5]
    ])

    for i in range(num_new):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Generate a new instruction-output pair similar to these examples. "
                           "Make it diverse and high quality."
            }, {
                "role": "user",
                "content": f"Examples:\n{seed_text}\n\nGenerate a new unique example:"
            }],
            temperature=0.9,
        )

        # Parse the response into instruction/output
        text = response.choices[0].message.content
        # ... parse and validate ...
        augmented.append(parsed_example)

    return augmented
```

---

## 7. Training Process

### 7.1 Training Configuration Deep Dive

```python
from trl import SFTConfig

training_args = SFTConfig(
    # --- Output ---
    output_dir="./output",
    overwrite_output_dir=True,

    # --- Training duration ---
    num_train_epochs=3,                  # 1-3 for fine-tuning (less is often more)
    max_steps=-1,                        # -1 = use num_train_epochs

    # --- Batch size ---
    per_device_train_batch_size=4,       # per GPU
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,       # effective batch = 4 * 4 = 16
    # Rule: effective_batch_size = per_device * num_gpus * gradient_accumulation

    # --- Learning rate ---
    learning_rate=2e-4,                  # typical for LoRA/QLoRA
    # Full fine-tuning: 1e-5 to 5e-5
    # LoRA/QLoRA:       1e-4 to 3e-4

    # --- Scheduler ---
    lr_scheduler_type="cosine",          # cosine or linear
    warmup_ratio=0.03,                   # 3% of total steps for warmup
    # warmup_steps=100,                  # alternative: fixed number

    # --- Regularization ---
    weight_decay=0.001,                  # L2 regularization
    max_grad_norm=0.3,                   # gradient clipping

    # --- Precision ---
    bf16=True,                           # use bf16 (preferred on Ampere+)
    # fp16=True,                         # use fp16 (older GPUs)

    # --- Memory optimization ---
    gradient_checkpointing=True,         # recompute activations to save memory
    optim="paged_adamw_8bit",            # 8-bit paged Adam (QLoRA)
    # optim="adamw_torch",               # standard Adam

    # --- Sequence length ---
    max_seq_length=2048,

    # --- Packing ---
    packing=True,                        # pack multiple samples per sequence
    # Increases efficiency by reducing padding waste

    # --- Logging ---
    logging_steps=10,
    logging_dir="./logs",
    report_to="wandb",                   # or "tensorboard", "none"

    # --- Saving ---
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,                  # keep only last 3 checkpoints

    # --- Evaluation ---
    eval_strategy="steps",
    eval_steps=500,

    # --- Reproducibility ---
    seed=42,

    # --- Data loading ---
    dataloader_num_workers=4,
    dataloader_pin_memory=True,

    # --- Distributed training ---
    # ddp_find_unused_parameters=False,  # for multi-GPU
    # fsdp="full_shard auto_wrap",       # for FSDP
)
```

### 7.2 Visual: Learning Rate Schedule

```
Learning Rate Over Training Steps (Cosine with Warmup):

LR
  ▲
  │     peak LR (2e-4)
  │    ╱‾‾‾╲
  │   ╱      ╲
  │  ╱        ╲
  │ ╱          ╲
  │╱             ╲
  │                ╲
  │ warmup          ╲  cosine decay
  │ (3%)             ╲
  │                    ╲___
  └────────────────────────────► steps
  0    100   500   1000  1500  2000

Warmup: Linearly increase LR from 0 to peak
Cosine: Smoothly decrease LR following cosine curve
```

### 7.3 Gradient Accumulation Explained

```
Without gradient accumulation (batch_size=16, but GPU can only fit 4):
    OOM Error!

With gradient accumulation (batch_size=4, accumulation_steps=4):
    Step 1: Forward + backward on batch of 4   → accumulate gradients
    Step 2: Forward + backward on batch of 4   → accumulate gradients
    Step 3: Forward + backward on batch of 4   → accumulate gradients
    Step 4: Forward + backward on batch of 4   → accumulate gradients
    → Now update weights (effective batch size = 4 * 4 = 16)

    Same mathematical result as batch_size=16, but fits in memory!
```

### 7.4 Complete Training Script

```python
"""
Complete QLoRA fine-tuning script.
Usage: python train.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import wandb

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DATASET_PATH = "data/train.jsonl"
OUTPUT_DIR = "./output/mistral-7b-qlora"
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4

# ============================================================
# 1. Load quantized model
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================================
# 2. Prepare for training
# ============================================================
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

# ============================================================
# 3. Load and format dataset
# ============================================================
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.shuffle(seed=42)

# Split into train/eval
split = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

def format_example(example):
    """Format into chat template."""
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    return {"text": text}

train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(format_example, remove_columns=eval_dataset.column_names)

# ============================================================
# 4. Training
# ============================================================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    weight_decay=0.001,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    max_grad_norm=0.3,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=500,
    report_to="wandb",
    seed=42,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    args=training_args,
    tokenizer=tokenizer,
)

# ============================================================
# 5. Train and save
# ============================================================
print("Starting training...")
trainer.train()

print("Saving model...")
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print("Done!")
```

### 7.5 Packing Explained

```
Without packing (padding waste):
┌──────────────────────────────────────────┐
│ Sample 1 tokens ████████░░░░░░░░░░░░░░░ │  ← 60% padding waste
│ Sample 2 tokens ███████████████░░░░░░░░░ │  ← 40% padding waste
│ Sample 3 tokens ██████░░░░░░░░░░░░░░░░░ │  ← 70% padding waste
└──────────────────────────────────────────┘

With packing (samples concatenated with EOS separator):
┌──────────────────────────────────────────┐
│ Sample1████████<eos>Sample2██████████████ │  ← ~0% waste
│ █<eos>Sample3██████<eos>Sample4█████████ │  ← ~0% waste
│ ██████████<eos>Sample5████████████████░░ │  ← minimal waste
└──────────────────────────────────────────┘

Attention mask ensures samples don't attend to each other.
```

---

## 8. Evaluation

### 8.1 Loss Curves

```
Ideal training:                       Overfitting:
Loss                                  Loss
  ▲                                     ▲
  │ ╲                                   │ ╲        ╱ validation loss
  │  ╲  train loss                      │  ╲     ╱   (goes up!)
  │   ╲─────────                        │   ╲  ╱
  │    ╲──── validation loss            │    ╲╱
  │     ╲────────────                   │     ╲──── train loss
  │      ╲───────────                   │      ╲──── (keeps going down)
  └──────────────────► epoch            └──────────────────► epoch
  1    2    3    4                       1    2    3    4

Good: both curves decrease and converge     Bad: gap between train and val grows
```

### 8.2 Key Metrics

```python
import math
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_metric

# --- 1. Perplexity (lower is better) ---
# Measures how "surprised" the model is by the test data
def compute_perplexity(eval_loss):
    """Compute perplexity from evaluation loss."""
    return math.exp(eval_loss)

# Example: eval_loss=2.5 → perplexity=12.18
# Good perplexity for a fine-tuned model: < 10 on domain data

# --- 2. Task-specific metrics ---
# For classification:
predictions = ["positive", "negative", "positive", "neutral"]
references = ["positive", "negative", "negative", "neutral"]

accuracy = accuracy_score(references, predictions)  # 0.75
f1 = f1_score(references, predictions, average="macro")

# --- 3. ROUGE (for summarization) ---
from evaluate import load
rouge = load("rouge")
results = rouge.compute(
    predictions=["The cat sat on the mat"],
    references=["The cat is on the mat"],
)
# {'rouge1': 0.857, 'rouge2': 0.6, 'rougeL': 0.857}

# --- 4. Custom evaluation with generation ---
def evaluate_model(model, tokenizer, test_examples, max_new_tokens=256):
    """Generate outputs and compute metrics."""
    model.eval()
    predictions = []

    for example in test_examples:
        prompt = format_prompt(example["instruction"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,        # low temp for evaluation
                do_sample=False,         # greedy decoding
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        predictions.append(response.strip())

    return predictions
```

### 8.3 Benchmarks

```
Common LLM Benchmarks:
──────────────────────
Benchmark       What it measures                 Task Type
─────────────   ─────────────────────────────    ──────────────
MMLU            Multitask language understanding  Multiple choice
HellaSwag       Common-sense reasoning            Sentence completion
ARC             Science reasoning                 Multiple choice
TruthfulQA      Truthfulness                      Open-ended
GSM8K           Math reasoning                    Word problems
HumanEval       Code generation                   Function completion
MT-Bench        Multi-turn conversation quality   Open-ended (LLM judge)
AlpacaEval      Instruction following             Pairwise comparison

Running benchmarks with lm-evaluation-harness:
$ pip install lm-eval
$ lm_eval --model hf \
    --model_args pretrained=./my-model,peft=./my-adapter \
    --tasks mmlu,hellaswag,arc_easy \
    --batch_size 8
```

### 8.4 Preventing Overfitting

| Technique | How It Helps |
|---|---|
| **Early stopping** | Stop when validation loss stops improving |
| **Fewer epochs** | 1-3 epochs is typical; more risks overfitting |
| **LoRA dropout** | Regularization within adapter layers |
| **Weight decay** | L2 regularization on weights |
| **Lower rank (r)** | Fewer trainable parameters = less overfitting |
| **More data** | Larger dataset reduces overfitting risk |
| **Data augmentation** | Increases effective dataset size |
| **Gradient clipping** | Prevents training instability |

---

## 9. Model Merging & Deployment

### 9.1 Merging LoRA Adapters

```python
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- Method 1: Using AutoPeftModelForCausalLM ---
model = AutoPeftModelForCausalLM.from_pretrained(
    "./qlora-adapter",
    torch_dtype=torch.float16,
    device_map="auto",
)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")

# --- Method 2: Manual loading ---
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./qlora-adapter")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")

# Save tokenizer alongside
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.save_pretrained("./merged-model")
```

### 9.2 Converting to GGUF (for llama.cpp / Ollama)

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Convert HF model to GGUF
python convert_hf_to_gguf.py ../merged-model --outtype f16 --outfile model-f16.gguf

# Quantize for efficient inference
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M

# Common quantization levels:
# Q4_K_M  - good balance of quality and size (recommended)
# Q5_K_M  - slightly better quality, larger
# Q8_0    - near-original quality, larger
# Q2_K    - smallest, noticeable quality loss

# Test locally
./llama-cli -m model-q4_k_m.gguf -p "Hello, how are you?" -n 100

# Or use with Ollama:
# Create a Modelfile:
# FROM ./model-q4_k_m.gguf
# PARAMETER temperature 0.7
# SYSTEM "You are a helpful assistant."

# ollama create my-model -f Modelfile
# ollama run my-model
```

### 9.3 Serving with vLLM

```python
# --- Install ---
# pip install vllm

# --- Serve merged model ---
# vllm serve ./merged-model --port 8000

# --- Or serve with LoRA adapter ---
# vllm serve mistralai/Mistral-7B-v0.1 \
#     --enable-lora \
#     --lora-modules my-adapter=./qlora-adapter \
#     --port 8000

# --- Python client ---
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="./merged-model",    # or "my-adapter" for LoRA
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=0.7,
    max_tokens=256,
)
print(response.choices[0].message.content)
```

### 9.4 Serving with Text Generation Inference (TGI)

```bash
# Docker deployment
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v ./merged-model:/model \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id /model \
    --quantize bitsandbytes-nf4 \
    --max-input-length 2048 \
    --max-total-tokens 4096
```

### 9.5 Deployment Quantization (GPTQ, AWQ)

```python
# --- GPTQ quantization (post-training, calibration-based) ---
from transformers import AutoModelForCausalLM, GPTQConfig

gptq_config = GPTQConfig(
    bits=4,
    dataset="c4",            # calibration dataset
    tokenizer=tokenizer,
    group_size=128,
)

quantized_model = AutoModelForCausalLM.from_pretrained(
    "./merged-model",
    quantization_config=gptq_config,
    device_map="auto",
)
quantized_model.save_pretrained("./model-gptq-4bit")

# --- AWQ quantization (Activation-aware Weight Quantization) ---
# pip install autoawq
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("./merged-model")
tokenizer = AutoTokenizer.from_pretrained("./merged-model")

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized("./model-awq-4bit")
```

### 9.6 Deployment Architecture

```
Production Deployment Pipeline:
───────────────────────────────

Training                    Optimization              Serving
┌────────────┐             ┌─────────────┐           ┌─────────────┐
│ QLoRA      │  merge      │ Merge       │  quantize │ vLLM /      │
│ Training   │────────────►│ Adapter +   │──────────►│ TGI /       │
│            │             │ Base Model  │           │ llama.cpp   │
└────────────┘             └─────────────┘           └──────┬──────┘
                                                            │
                                                            ▼
                                                    ┌──────────────┐
                                                    │ API Gateway   │
                                                    │ (FastAPI /    │
                                                    │  Load         │
                                                    │  Balancer)    │
                                                    └──────┬───────┘
                                                           │
                                                    ┌──────┴───────┐
                                                    │  Clients     │
                                                    └──────────────┘
```

---

## 10. Advanced Topics

### 10.1 DPO (Direct Preference Optimization)

DPO aligns a model with human preferences **without** a separate reward model. Instead of the
complex RLHF pipeline, DPO directly optimizes the policy using preference pairs.

```
RLHF Pipeline (complex):
    SFT Model → Reward Model Training → PPO Training → Aligned Model
    (3 models in memory, unstable, expensive)

DPO Pipeline (simpler):
    SFT Model → DPO Training (with preference pairs) → Aligned Model
    (1 model, stable, cheaper)
```

**DPO data format:**
```json
{
    "prompt": "Explain quantum computing simply.",
    "chosen": "Quantum computing uses quantum bits (qubits) that can be 0, 1, or both simultaneously...",
    "rejected": "Quantum computing is a type of computer that is very fast and uses quantum mechanics..."
}
```

```python
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig

# --- DPO Training ---
dpo_config = DPOConfig(
    output_dir="./dpo-output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,             # lower LR for DPO
    beta=0.1,                       # DPO temperature parameter
    # Higher beta = more conservative updates
    # Lower beta = larger updates (may be unstable)
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_length=2048,
    max_prompt_length=512,
    logging_steps=10,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,                  # None = use implicit reference (LoRA)
    train_dataset=preference_dataset,
    peft_config=lora_config,
    args=dpo_config,
    tokenizer=tokenizer,
)

trainer.train()
```

### 10.2 RLHF (Reinforcement Learning from Human Feedback)

```
RLHF Pipeline:
──────────────

Step 1: Supervised Fine-Tuning (SFT)
    Pre-trained model + instruction data → SFT model

Step 2: Reward Model Training
    Collect human preference data (chosen vs rejected)
    Train a reward model to predict human preferences

Step 3: PPO (Proximal Policy Optimization)
    Use reward model to score generations
    Optimize SFT model with PPO to maximize reward

    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    │  Prompt ──► SFT Model ──► Response ──► Reward Model  │
    │                ▲                          │           │
    │                │        reward signal     │           │
    │                └──────────────────────────┘           │
    │                     (PPO update)                     │
    └──────────────────────────────────────────────────────┘

Challenges:
    - Needs 3 models in memory (policy, reference, reward)
    - Reward hacking (model exploits reward model)
    - Training instability
    - Expensive (human annotations + compute)
```

### 10.3 Multi-Task Fine-Tuning

Train a single model on multiple tasks simultaneously.

```python
# Combine datasets with task identifiers
combined_dataset = []

for example in summarization_data:
    combined_dataset.append({
        "instruction": f"[SUMMARIZE] {example['instruction']}",
        "output": example["output"],
    })

for example in classification_data:
    combined_dataset.append({
        "instruction": f"[CLASSIFY] {example['instruction']}",
        "output": example["output"],
    })

for example in translation_data:
    combined_dataset.append({
        "instruction": f"[TRANSLATE] {example['instruction']}",
        "output": example["output"],
    })

# Shuffle and train as usual
import random
random.shuffle(combined_dataset)
```

### 10.4 Continual Learning

Fine-tuning sequentially on new tasks without forgetting previous ones.

```
Strategies:
    1. Replay buffer:  Mix old task data with new task data
    2. EWC (Elastic Weight Consolidation): Penalize changes to important weights
    3. Progressive LoRA: Train separate LoRA adapters for each task
    4. Adapter merging: Merge adapters from different tasks

Progressive LoRA:
    Task A → LoRA adapter A
    Task B → LoRA adapter B
    Task C → LoRA adapter C
    Inference: Load base model + relevant adapter(s)
```

### 10.5 Mixture of Experts (MoE) Fine-Tuning

```
MoE Architecture:
    Input ──► Router ──► Expert 1 ──►┐
                    ├──► Expert 2 ──►├──► Weighted Sum ──► Output
                    ├──► Expert 3    │
                    └──► Expert N ──►┘

    Only top-K experts are active per token (typically K=2)
    Allows larger models with same compute cost

Fine-tuning MoE models (e.g., Mixtral 8x7B):
    - Apply LoRA to each expert's layers
    - Can selectively fine-tune certain experts
    - Memory: Only active experts need gradients
    - QLoRA makes Mixtral 8x7B trainable on a single GPU

Example:
    Mixtral 8x7B: 47B total params, ~13B active per token
    QLoRA on Mixtral: ~24 GB VRAM (fits on RTX 4090)
```

### 10.6 Recent Techniques

```
Technique               Year    Key Idea
─────────────────────   ────    ───────────────────────────────────────
DoRA                    2024    Weight-Decomposed Low-Rank Adaptation
                                (decomposes into magnitude + direction)
LongLoRA                2023    Efficient fine-tuning for long contexts
                                (shift short attention + LoRA)
NEFTune                 2023    Add noise to embeddings during training
                                (improves instruction following)
ORPO                    2024    Odds Ratio Preference Optimization
                                (combines SFT and alignment in one step)
Unsloth                 2024    Optimized LoRA training (2x faster)
                                (custom CUDA kernels, less memory)
```

---

## 11. Q&A Section

### Q1: What is LoRA and how does it work?

**Answer:** LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that freezes
the original pre-trained weights and adds small, trainable low-rank decomposition matrices
alongside them. Instead of updating a full weight matrix `W` (d x d), LoRA adds `DeltaW = B * A`
where `A` is (d x r) and `B` is (r x d), with `r` being a small rank (e.g., 8 or 16). This
reduces trainable parameters by ~99.8% while achieving comparable performance to full fine-tuning.
Matrix B is initialized to zeros so the model starts identical to the pre-trained version.

---

### Q2: What is the difference between LoRA and QLoRA?

**Answer:** LoRA keeps the base model in fp16/bf16 (half precision) and adds trainable low-rank
adapters. QLoRA goes further by quantizing the base model to 4-bit precision (using NF4 data
type) while keeping the LoRA adapters in fp16/bf16. QLoRA also introduces double quantization
(quantizing the quantization constants) and paged optimizers (offloading optimizer states to
CPU). The result: QLoRA uses roughly 4x less memory than LoRA, enabling fine-tuning of a 7B
model on a single consumer GPU with 6-8 GB VRAM.

---

### Q3: When should you fine-tune vs use RAG vs prompt engineering?

**Answer:**
- **Prompt engineering**: When the base model can do the task with the right instructions. Cheapest and fastest to iterate.
- **RAG**: When you need access to specific, potentially changing knowledge (documents, databases). The model's behavior is fine but it lacks certain information.
- **Fine-tuning**: When you need to change the model's behavior, style, format, or reasoning patterns. When you have consistent, high-quality examples and prompting alone cannot achieve the desired output quality.

Often the best approach combines them: fine-tune for behavior and format, then use RAG for knowledge grounding.

---

### Q4: What is the rank parameter (r) in LoRA and how do you choose it?

**Answer:** The rank `r` controls the dimensionality of the low-rank decomposition. Lower rank
means fewer trainable parameters and less capacity; higher rank means more parameters and more
capacity but also more risk of overfitting and higher memory usage.

Guidelines:
- **r=4-8**: Simple tasks, small datasets, limited compute
- **r=16**: Good default for most tasks
- **r=32-64**: Complex tasks, large datasets, ample compute
- **r=128+**: Rarely needed; approaches full fine-tuning capacity

The effective scaling is `alpha / r`, so when increasing `r`, proportionally increase `alpha` to maintain the same update magnitude.

---

### Q5: How do you prepare a dataset for fine-tuning?

**Answer:** Key steps:
1. **Format**: Convert to instruction/chat format with clear input-output pairs
2. **Clean**: Remove empty, duplicate, and low-quality examples
3. **Filter**: Remove too-short or too-long examples, filter refusal patterns
4. **Deduplicate**: Remove exact and near-duplicates
5. **Split**: Create train/validation/test splits (e.g., 90/5/5)
6. **Validate**: Manually review a sample for quality
7. **Tokenize**: Apply the model's tokenizer and ensure sequence lengths fit within context window

Quality matters far more than quantity: 1,000 excellent examples often outperform 100,000 mediocre ones.

---

### Q6: What is catastrophic forgetting and how do you prevent it?

**Answer:** Catastrophic forgetting occurs when a model trained on new data loses its previously
learned capabilities. For example, a model fine-tuned heavily on medical data may forget how to
write code or do math.

Prevention strategies:
- **Use PEFT methods** (LoRA/QLoRA): Since most weights are frozen, general capabilities are preserved
- **Mix general data** with domain data during training (e.g., 80% domain + 20% general)
- **Use lower learning rates** (1e-5 to 5e-5 for full FT)
- **Train for fewer epochs** (1-3)
- **Weight decay** and regularization
- **Evaluate on general benchmarks** alongside domain metrics during training

---

### Q7: Explain quantization (4-bit, 8-bit) in the context of LLMs.

**Answer:** Quantization reduces the precision of model weights from higher bit representations
to lower ones:

- **FP32** (32 bits): Full precision. 1 param = 4 bytes. 7B model = 28 GB.
- **FP16/BF16** (16 bits): Half precision. 1 param = 2 bytes. 7B model = 14 GB.
- **INT8** (8 bits): 1 param = 1 byte. 7B model = 7 GB. Minimal quality loss.
- **INT4/NF4** (4 bits): 1 param = 0.5 bytes. 7B model = 3.5 GB. Small quality loss.

NF4 (Normal Float 4) is optimized for normally-distributed weights and provides better quality
than naive 4-bit quantization. Double quantization further compresses the quantization constants.

For training, BitsAndBytes handles 4-bit/8-bit quantization. For inference/deployment, GPTQ and
AWQ are common choices.

---

### Q8: How do you evaluate a fine-tuned model?

**Answer:**
1. **Loss metrics**: Monitor training and validation loss curves. Divergence indicates overfitting.
2. **Perplexity**: exp(loss). Lower is better. Domain-specific perplexity should decrease.
3. **Task-specific metrics**: Accuracy, F1, ROUGE, BLEU depending on the task.
4. **Benchmark suites**: MMLU, HellaSwag, etc., to check general capability retention.
5. **Human evaluation**: Gold standard. Have domain experts rate outputs for accuracy, helpfulness, and safety.
6. **LLM-as-judge**: Use a stronger model (e.g., GPT-4) to evaluate outputs (cost-effective alternative to human eval).
7. **A/B testing**: Compare fine-tuned model against baseline in production.

---

### Q9: What is the PEFT library and what methods does it support?

**Answer:** PEFT (Parameter-Efficient Fine-Tuning) is a Hugging Face library that provides
implementations of methods that fine-tune only a small subset of model parameters. It supports:

- **LoRA**: Low-rank adaptation of weight matrices
- **QLoRA**: LoRA on quantized models (via BitsAndBytes integration)
- **Prefix Tuning**: Prepend trainable vectors to attention keys/values
- **Prompt Tuning**: Prepend trainable soft tokens to input
- **(IA)^3**: Learned vectors that rescale activations
- **AdaLoRA**: Adaptive rank allocation for LoRA

PEFT integrates seamlessly with transformers and TRL, allowing you to wrap any HF model with a
few lines of code.

---

### Q10: How do you choose target modules for LoRA?

**Answer:** The target modules determine which layers get LoRA adapters:

- **Minimal (q_proj, v_proj)**: The original LoRA paper's default. Fast, lightweight, good for simple tasks.
- **Standard (q_proj, k_proj, v_proj, o_proj)**: All attention projections. Good balance for most tasks.
- **Full (all linear layers)**: Includes MLP layers (gate_proj, up_proj, down_proj). Maximum capacity, best for complex tasks.

Guidelines: Start with the standard attention-only configuration. If performance is insufficient,
add MLP layers. More target modules means more trainable parameters, more memory, and a bigger
adapter file. You can inspect which modules are available with:
```python
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name)
```

---

### Q11: What learning rate should you use for fine-tuning?

**Answer:**
- **Full fine-tuning**: 1e-5 to 5e-5 (conservative to avoid catastrophic forgetting)
- **LoRA/QLoRA**: 1e-4 to 3e-4 (can be higher since base weights are frozen)
- **DPO**: 5e-6 to 5e-5 (lower to avoid instability)

Use a cosine or linear scheduler with 3-5% warmup. If training is unstable (loss spikes), reduce
the learning rate. If training is too slow (loss barely decreases), increase it. Always monitor
validation loss to detect the right stopping point.

---

### Q12: What is DPO and how does it differ from RLHF?

**Answer:** DPO (Direct Preference Optimization) and RLHF both align models with human
preferences, but they differ in approach:

**RLHF**: Train a reward model on preference data, then use PPO (reinforcement learning) to
optimize the language model against that reward model. Requires 3 models in memory (policy,
reference, reward), is complex to implement, and can be unstable.

**DPO**: Directly optimizes the language model using preference pairs (chosen/rejected) without a
separate reward model. It reformulates the RLHF objective into a simple classification-like loss.
Requires only 1-2 models, is stable, and is much simpler to implement.

DPO has largely replaced RLHF in practice due to its simplicity and comparable results.

---

### Q13: How do you deploy a fine-tuned model in production?

**Answer:** Steps:
1. **Merge**: Merge LoRA adapter into base model (`model.merge_and_unload()`)
2. **Quantize**: Apply GPTQ or AWQ for efficient inference, or convert to GGUF
3. **Serve**: Use an inference server:
   - **vLLM**: High-throughput, PagedAttention, continuous batching
   - **TGI**: Hugging Face's inference server, good Docker support
   - **llama.cpp / Ollama**: For CPU or edge deployment
4. **API layer**: Wrap with FastAPI or use the server's built-in API (OpenAI-compatible)
5. **Scale**: Add load balancing, auto-scaling, and monitoring
6. **Monitor**: Track latency, throughput, error rates, and output quality

---

### Q14: What is the difference between SFT and RLHF?

**Answer:**
- **SFT (Supervised Fine-Tuning)**: Train the model to imitate examples. Learns from input-output pairs directly. Simple and stable. This is the standard "fine-tuning" most people mean.
- **RLHF (Reinforcement Learning from Human Feedback)**: Train the model to maximize a learned reward function based on human preferences. More complex but can teach nuanced behaviors that are hard to demonstrate with examples (like being helpful without being harmful).

Typical alignment pipeline: Pre-training -> SFT -> RLHF/DPO. SFT teaches the model *what* to do; RLHF/DPO refines *how* to do it based on human preferences.

---

### Q15: How much data do you need for fine-tuning?

**Answer:** It depends on the task complexity:
- **Simple format changes** (e.g., JSON output): 50-200 examples
- **Instruction following**: 500-2,000 examples
- **Domain adaptation**: 1,000-10,000 examples
- **Complex reasoning tasks**: 5,000-50,000+ examples
- **Full pre-training behavior shift**: 100,000+ examples

Rules of thumb:
- Quality > quantity. 500 excellent examples beat 50,000 noisy ones.
- Start small, evaluate, then add more data if needed.
- If loss plateaus quickly, you may need more diverse (not just more) data.
- For LoRA with small rank, less data is needed since there are fewer parameters to fit.

---

### Q16: What is gradient checkpointing and when should you use it?

**Answer:** Gradient checkpointing (also called activation checkpointing) is a memory
optimization technique. Normally, all intermediate activations are stored during the forward pass
for use in backpropagation. Gradient checkpointing discards most activations and recomputes them
during the backward pass.

- **Memory savings**: Reduces activation memory from O(n) to O(sqrt(n)) where n is model depth
- **Speed trade-off**: ~20-30% slower training due to recomputation
- **When to use**: Almost always during fine-tuning, especially with QLoRA. The memory savings are significant and the speed cost is acceptable.

Enable with: `gradient_checkpointing=True` in TrainingArguments.

---

### Q17: What is the difference between fp16 and bf16? Which should you use?

**Answer:**
- **fp16** (float16): 1 sign bit, 5 exponent bits, 10 mantissa bits. Narrower range but higher precision within that range. Can overflow/underflow more easily.
- **bf16** (bfloat16): 1 sign bit, 8 exponent bits, 7 mantissa bits. Same range as fp32 but lower precision. More numerically stable for training.

Use **bf16** if your GPU supports it (NVIDIA Ampere/A100+, RTX 30xx+). Use **fp16** on older
GPUs (V100, RTX 20xx). bf16 is generally preferred for training because its larger exponent range
prevents overflow issues that can cause NaN losses.

---

### Q18: How do you handle long sequences during fine-tuning?

**Answer:**
- **Truncation**: Simply cut sequences at max_length. Simplest but loses information.
- **Chunking**: Split long texts into overlapping chunks, each within max_length.
- **Packing**: Concatenate multiple short sequences to fill max_length efficiently.
- **RoPE scaling**: Extend the model's positional encoding to handle longer contexts (e.g., from 4K to 16K). Techniques include linear scaling, NTK-aware scaling, and YaRN.
- **Flash Attention**: Reduces memory from O(n^2) to O(n) for attention computation, enabling longer sequences.
- **Gradient checkpointing**: Reduces activation memory, allowing longer sequences.

For training, the typical approach is to use packing (for short inputs) or truncation with gradient checkpointing (for long inputs).

---

### Q19: What is packing in SFTTrainer and why use it?

**Answer:** Packing concatenates multiple training examples into a single sequence (separated by
EOS tokens) to minimize padding waste. Without packing, short examples are padded to max_length,
wasting compute on padding tokens.

Benefits:
- **Efficiency**: Reduces wasted compute by 30-70% for datasets with variable-length examples
- **Speed**: Fewer forward passes needed per epoch
- **Memory**: Better GPU utilization

Caveats:
- Attention masks must prevent cross-contamination between packed samples
- Some implementations may not handle this correctly (check your TRL version)
- Disable packing if you see quality degradation

Enable with: `packing=True` in SFTConfig.

---

### Q20: How do you fine-tune a model for function calling / tool use?

**Answer:** Function calling fine-tuning teaches a model to generate structured tool calls:

1. **Data format**: Each example includes available tools and the expected tool call:
```json
{
    "messages": [
        {"role": "system", "content": "You have access to: get_weather(city: str)"},
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": null, "tool_calls": [
            {"function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}
        ]},
        {"role": "tool", "content": "Sunny, 22C"},
        {"role": "assistant", "content": "The weather in Paris is sunny at 22C."}
    ]
}
```
2. **Include negative examples** where the model should NOT call tools
3. **Train with chat template** that supports tool definitions
4. **Evaluate**: Check tool call accuracy (correct function + correct arguments)

Models like Mistral and LLaMA 3 have native tool-calling support in their chat templates.

---

### Q21: What are common failure modes in fine-tuning?

**Answer:**
1. **Overfitting**: Model memorizes training data, poor generalization. Fix: more data, fewer epochs, lower rank, dropout.
2. **Catastrophic forgetting**: Loses general capabilities. Fix: mix general data, use PEFT, lower LR.
3. **Mode collapse**: Model produces repetitive, generic outputs. Fix: increase temperature, check data diversity.
4. **NaN loss**: Training diverges. Fix: lower LR, use bf16 instead of fp16, check data for issues.
5. **Reward hacking** (RLHF/DPO): Model exploits reward signal. Fix: better reward model, KL penalty.
6. **Poor data quality**: Garbage in, garbage out. Fix: invest in data cleaning and validation.
7. **Wrong format/template**: Mismatch between training and inference formatting. Fix: use the exact same template.

---

### Q22: How do you fine-tune for a specific language (e.g., non-English)?

**Answer:**
1. **Choose a multilingual base model** (e.g., LLaMA 3, Qwen, multilingual variants)
2. **Collect target-language data**: Instructions and responses in the target language
3. **Consider vocabulary**: If the base tokenizer poorly handles your language, token efficiency will be low. Some practitioners extend the vocabulary, but this is complex.
4. **Mix languages**: Include some English data to prevent forgetting English capabilities (useful for code-switching)
5. **Evaluate in the target language**: Use language-specific benchmarks
6. **Typical data**: 5,000-50,000 examples in the target language for good results with LoRA

---

### Q23: What is the difference between adapter merging strategies?

**Answer:** When you have multiple LoRA adapters, you can combine them:

- **Linear merging**: Simple weighted average of adapter weights. `W = W_base + w1 * adapter1 + w2 * adapter2`. Fast but may lose capabilities.
- **TIES merging**: Trims small values, resolves sign conflicts, then merges. Better at preserving individual adapter strengths.
- **DARE**: Randomly drops parameters before merging. Reduces interference between adapters.
- **Task arithmetic**: Treat adapters as task vectors and perform arithmetic operations (add, negate, combine).

Use cases: Combine a coding adapter with a math adapter to get a model good at both, without retraining.

---

### Q24: What is the role of the `alpha` parameter in LoRA?

**Answer:** The `alpha` parameter is a scaling factor that controls the magnitude of the LoRA
update. The actual scaling applied is `alpha / r`, where `r` is the rank. This means:

- If `r=16, alpha=16`: scaling = 1.0 (standard)
- If `r=16, alpha=32`: scaling = 2.0 (amplified updates)
- If `r=16, alpha=8`: scaling = 0.5 (dampened updates)

The reason for this scaling: as you change the rank, you want to be able to keep the update
magnitude roughly constant without retuning the learning rate. A common practice is to set
`alpha = 2 * r`, giving a scaling factor of 2.0. Some practitioners fix alpha (e.g., alpha=16)
and only vary the rank.

---

### Q25: How do you debug a fine-tuning run that is not converging?

**Answer:** Systematic debugging checklist:

1. **Check data**: Are examples correctly formatted? Is the tokenization correct? Print a few tokenized examples and decode them to verify.
2. **Check loss**: Is training loss decreasing at all? If not, LR may be too low or data is malformed.
3. **Check learning rate**: Try 10x higher and 10x lower. Plot loss vs LR with a learning rate finder.
4. **Check gradients**: Are they NaN or zero? Use `max_grad_norm` for clipping.
5. **Check precision**: Try bf16 instead of fp16, or use fp32 for debugging.
6. **Check template**: Ensure training template matches what the model expects.
7. **Reduce complexity**: Try training on 100 examples first. If it can memorize them, the training pipeline works.
8. **Check labels**: Ensure loss is computed only on output tokens, not on input/padding tokens.
9. **Inspect outputs**: Generate text at various checkpoints to see qualitative progress.
10. **Baseline**: Compare against a known-good configuration from a tutorial or paper.

```python
# Quick sanity check: can the model overfit on a tiny dataset?
tiny_dataset = dataset.select(range(50))
trainer = SFTTrainer(
    model=model,
    train_dataset=tiny_dataset,
    args=SFTConfig(
        output_dir="./debug",
        num_train_epochs=20,        # overfit deliberately
        per_device_train_batch_size=4,
        learning_rate=3e-4,
        logging_steps=1,
        max_seq_length=512,
    ),
    peft_config=lora_config,
)
trainer.train()
# Training loss should approach 0. If not, something is wrong.
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FINE-TUNING QUICK REFERENCE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Method          VRAM (7B)    Trainable %    Quality    Speed       │
│  ─────────────   ─────────   ───────────    ────────   ─────       │
│  Full FT (fp16)  ~60 GB      100%           Best       Slow        │
│  LoRA (fp16)     ~18 GB      ~0.2%          Very Good  Fast        │
│  QLoRA (4-bit)   ~6 GB       ~0.2%          Good       Fast        │
│                                                                     │
│  Common Hyperparameters:                                            │
│  ──────────────────────                                             │
│  LoRA rank (r):        16 (default)                                 │
│  LoRA alpha:           32 (2 * r)                                   │
│  Learning rate:        2e-4 (LoRA) / 2e-5 (full FT)                │
│  Epochs:               1-3                                          │
│  Batch size:           4-16 (effective, with grad accumulation)     │
│  Warmup:               3-5% of total steps                          │
│  Scheduler:            cosine                                       │
│  Weight decay:         0.001-0.01                                   │
│  Max seq length:       2048-4096                                    │
│                                                                     │
│  Key Libraries:                                                     │
│  ──────────────                                                     │
│  transformers    - Models and tokenizers                             │
│  peft            - LoRA, QLoRA, adapters                            │
│  trl             - SFTTrainer, DPOTrainer                           │
│  datasets        - Data loading and processing                      │
│  bitsandbytes    - 4-bit / 8-bit quantization                       │
│  accelerate      - Multi-GPU / distributed training                 │
│                                                                     │
│  Deployment Pipeline:                                               │
│  ───────────────────                                                │
│  Train (QLoRA) → Merge adapter → Quantize (GPTQ/AWQ/GGUF) → Serve │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

*This document covers the essential theory and practice of LLM fine-tuning for backend AI
engineer interviews. Focus on understanding the trade-offs between methods, the mathematical
intuition behind LoRA, and practical deployment considerations.*
