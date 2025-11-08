# nanochat training report

Generated: 2025-10-27 01:24:38

## Environment

### Git Information
- Branch: master
- Commit: c75fe54 (dirty)
- Message: readme tweak, link to new discussion and add file structure

### Hardware
- Platform: Linux
- CPUs: 16 cores (32 logical)
- Memory: 23.0 GB
- GPUs: 1x NVIDIA GeForce RTX 5090
- GPU Memory: 31.8 GB total
- CUDA Version: 12.8
- Hourly Rate: $2.00/hour

### Software
- Python: 3.10.18
- PyTorch: 2.8.0+cu128


### Bloat
- Characters: 401,897
- Lines: 9,818
- Files: 48
- Tokens (approx): 100,474
- Dependencies (uv.lock lines): 2,220

Run started: 2025-10-27 01:24:38

---

## Tokenizer training
timestamp: 2025-10-27 01:25:05

- max_chars: 1,000,000,000
- doc_cap: 10,000
- vocab_size: 65,536
- train_time: 20.7890
- num_special_tokens: 9
- token_bytes_min: 1
- token_bytes_max: 32
- token_bytes_mean: 6.9240
- token_bytes_std: 2.8765


## Tokenizer evaluation
timestamp: 2025-10-27 01:25:07

### Comparison with GPT-2

| Text Type | Bytes | GPT-2 Tokens | GPT-2 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 404 | 4.50 | 374 | 4.86 | +7.4% |
| korean | 893 | 745 | 1.20 | 712 | 1.25 | +4.4% |
| code | 1259 | 576 | 2.19 | 494 | 2.55 | +14.2% |
| math | 1834 | 936 | 1.96 | 966 | 1.90 | -3.2% |
| science | 1112 | 260 | 4.28 | 228 | 4.88 | +12.3% |
| fwe-train | 4208518 | 900364 | 4.67 | 856862 | 4.91 | +4.8% |
| fwe-val | 4908443 | 1059062 | 4.63 | 1010560 | 4.86 | +4.6% |

### Comparison with GPT-4

| Text Type | Bytes | GPT-4 Tokens | GPT-4 Ratio | Ours Tokens | Ours Ratio | Relative Diff % |
|-----------|-------|--------------|--------------|-------------|------------|-----------------|
| news | 1819 | 387 | 4.70 | 374 | 4.86 | +3.4% |
| korean | 893 | 364 | 2.45 | 712 | 1.25 | -95.6% |
| code | 1259 | 309 | 4.07 | 494 | 2.55 | -59.9% |
| math | 1834 | 832 | 2.20 | 966 | 1.90 | -16.1% |
| science | 1112 | 249 | 4.47 | 228 | 4.88 | +8.4% |
| fwe-train | 4208518 | 874799 | 4.81 | 856862 | 4.91 | +2.1% |
| fwe-val | 4908443 | 1029691 | 4.77 | 1010560 | 4.86 | +1.9% |


## Base model training
timestamp: 2025-10-27 04:51:10

- run: run10
- device_type: 
- depth: 10
- max_seq_len: 2048
- num_iterations: -1
- target_flops: -1.0000
- target_param_data_ratio: 20
- device_batch_size: 8
- total_batch_size: 524,288
- embedding_lr: 0.2000
- unembedding_lr: 0.0040
- weight_decay: 0.0000
- matrix_lr: 0.0200
- grad_clip: 1.0000
- warmup_ratio: 0.0000
- warmdown_ratio: 0.2000
- final_lr_frac: 0.0000
- eval_every: 250
- eval_tokens: 10,485,760
- core_metric_every: 2000
- core_metric_max_per_task: 500
- sample_every: 2000
- model_tag: 
- Number of parameters: 133,038,080
- Number of FLOPs per token: 7.038566e+08
- Calculated number of iterations: 5075
- Number of training tokens: 2,660,761,600
- Tokens : Params ratio: 20.0000
- DDP world size: 1
- warmup_ratio: 0.0000
- warmdown_ratio: 0.2000
- final_lr_frac: 0.0000
- Minimum validation bpb: 0.9539
- Final validation bpb: 0.9539
- CORE metric estimate: 0.1334
- MFU %: 16.31%
- Total training flops: 1.872795e+18
- Total training time: 194.68m
- Peak memory usage: 7496.89MiB


## Base model loss
timestamp: 2025-10-27 04:56:34

- train bpb: 0.9586
- val bpb: 0.9540
- sample 0: <|bos|>The capital of France is Paris. The capital is Paris. The capital is Paris. The capital is Paris
- sample 1: <|bos|>The chemical symbol of gold is gold. The symbol of gold is gold. The symbol of gold is gold.
- sample 2: <|bos|>If yesterday was Friday, then tomorrow will be. The day is a time to celebrate the 50th anniversary of the birth
- sample 3: <|bos|>The opposite of hot is hot. The opposite of hot is hot. The opposite of hot is hot.
- sample 4: <|bos|>The planets of the solar system are: Jupiter, Saturn, Uranus, Neptune, and Neptune. The planets are the only
- sample 5: <|bos|>My favorite color is the color of the sky. I love the color of the sky. I love
- sample 6: <|bos|>If 5*x + 3 = 13, then x is 13. If 5*x + 3 = 13, then


## Base model evaluation
timestamp: 2025-10-27 05:11:05

- Model: base_model (step 5075)
- CORE metric: 0.1266
- hellaswag_zeroshot: 0.0769
- jeopardy: 0.0038
- bigbench_qa_wikidata: 0.3168
- arc_easy: 0.3401
- arc_challenge: 0.0057
- copa: 0.2600
- commonsense_qa: 0.1165
- piqa: 0.2383
- openbook_qa: 0.0560
- lambada_openai: 0.2562
- hellaswag: 0.0746
- winograd: 0.1355
- winogrande: 0.0008
- bigbench_dyck_languages: 0.1140
- agi_eval_lsat_ar: 0.1467
- bigbench_cs_algorithms: 0.3826
- bigbench_operators: 0.1238
- bigbench_repeat_copy_logic: 0.0000
- squad: 0.0169
- coqa: 0.0724
- boolq: -0.1315
- bigbench_language_identification: 0.1802


## Midtraining
timestamp: 2025-10-27 05:46:45

- run: run10
- device_type: 
- dtype: bfloat16
- num_iterations: -1
- max_seq_len: 2048
- device_batch_size: 8
- unembedding_lr: 0.0040
- embedding_lr: 0.2000
- matrix_lr: 0.0200
- init_lr_frac: 1.0000
- weight_decay: 0.0000
- eval_every: 150
- eval_tokens: 10,485,760
- total_batch_size: 524,288
- dry_run: 0
- Number of iterations: 813
- DDP world size: 1
- Minimum validation bpb: 0.8500


## Chat evaluation mid
timestamp: 2025-10-27 06:40:36

- source: mid
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.2685
- ARC-Challenge: 0.2628
- MMLU: 0.2551
- GSM8K: 0.0000
- HumanEval: 0.0000
- SpellingBee: 0.0000
- ChatCORE metric: 0.0081


## Chat SFT
timestamp: 2025-10-27 06:45:09

- run: run10
- source: mid
- device_type: 
- dtype: bfloat16
- device_batch_size: 4
- num_epochs: 1
- num_iterations: -1
- target_examples_per_step: 32
- unembedding_lr: 0.0040
- embedding_lr: 0.2000
- matrix_lr: 0.0200
- weight_decay: 0.0000
- init_lr_frac: 0.0200
- eval_every: 100
- eval_steps: 100
- eval_metrics_every: 200
- eval_metrics_max_problems: 1024
- Training rows: 22,439
- Number of iterations: 701
- Training loss: 5.0734
- Validation loss: 4.5011


## Chat evaluation sft
timestamp: 2025-10-27 07:38:18

- source: sft
- task_name: None
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- device_type: 
- ARC-Easy: 0.2660
- ARC-Challenge: 0.2662
- MMLU: 0.2533
- GSM8K: 0.0000
- HumanEval: 0.0000
- SpellingBee: 0.0000
- ChatCORE metric: 0.0079


## Chat RL
timestamp: 2025-10-27 12:35:02

- run: run10
- source: sft
- dtype: bfloat16
- device_batch_size: 8
- examples_per_step: 16
- num_samples: 16
- max_new_tokens: 256
- temperature: 1.0000
- top_k: 50
- unembedding_lr: 0.0040
- embedding_lr: 0.2000
- matrix_lr: 0.0200
- weight_decay: 0.0000
- init_lr_frac: 0.0500
- num_epochs: 1
- save_every: 60
- eval_every: 60
- eval_examples: 400


## Chat evaluation rl
timestamp: 2025-10-27 13:14:07

- source: rl
- task_name: GSM8K
- dtype: bfloat16
- temperature: 0.0000
- max_new_tokens: 512
- num_samples: 1
- top_k: 50
- batch_size: 8
- model_tag: None
- step: None
- max_problems: None
- device_type: 
- GSM8K: 0.0129


## Summary

- Characters: 401,897
- Lines: 9,818
- Files: 48
- Tokens (approx): 100,474
- Dependencies (uv.lock lines): 2,220

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.1266   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2628   | 0.2662   | -        |
| ARC-Easy        | -        | 0.2685   | 0.2660   | -        |
| GSM8K           | -        | 0.0000   | 0.0000   | 0.0129   |
| HumanEval       | -        | 0.0000   | 0.0000   | -        |
| MMLU            | -        | 0.2551   | 0.2533   | -        |
| ChatCORE        | -        | 0.0081   | 0.0079   | -        |

Total wall clock time: 6h13m
