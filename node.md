

```shell
pip install -U accelerate evaluate datasets evaluate jsonlines numexpr peft pybind11 pytablewriter rouge-score sacrebleu scikit-learn sqlitedict zstandard dill word2number more_itertools 

python lm_eval/__main__.py --model hf --device cpu --batch_size auto \
    --model_args pretrained="LLMs/EleutherAI/pythia-70m" \
    --tasks gsm8k,ai2_arc,hellaswag,mmlu,truthfulqa,winogrande 


# 5-shot
python lm_eval/__main__.py --model vllm \
    --model_args pretrained="LLMs/meta-llama/Meta-Llama-3.1-8B-Instruct",tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.95,data_parallel_size=1 \
    --tasks mmlu,winogrande,gsm8k --batch_size auto --num_fewshot 5 --fewshot_as_multiturn --apply_chat_template true \
    --output_path ./output --write_out --log_samples --show_config

# 25-shot
python lm_eval/__main__.py --model vllm \
    --model_args pretrained="LLMs/meta-llama/Meta-Llama-3.1-8B-Instruct",tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.95,data_parallel_size=1 \
    --tasks ai2_arc --batch_size auto --num_fewshot 25 --fewshot_as_multiturn --apply_chat_template true \
    --output_path ./output --write_out --log_samples --show_config

# 10-shot
python lm_eval/__main__.py --model vllm \
    --model_args pretrained="LLMs/meta-llama/Meta-Llama-3.1-8B-Instruct",tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.95,data_parallel_size=1 \
    --tasks hellaswag --batch_size auto --num_fewshot 10 --fewshot_as_multiturn --apply_chat_template true \
    --output_path ./output --write_out --log_samples --show_config

# 0-shot
python lm_eval/__main__.py --model vllm \
    --model_args pretrained="LLMs/meta-llama/Meta-Llama-3.1-8B-Instruct",tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.95,data_parallel_size=1 \
    --tasks truthfulqa --batch_size auto --num_fewshot 0 --fewshot_as_multiturn --apply_chat_template true \
    --output_path ./output --write_out --log_samples --show_config
```

AI2 Reasoning Challenge, HellaSwag, MMLU, TruthfulQA, Winogrande, and GSM8k

- ai2_arc: 25 shots
- hellaswag: 10 shots
- mmlu: 5 shots
- truthfulqa: 0 shots
- winogrande: 5 shots
- gsm8k: 5 shots


EleutherAI__pythia-70m:
| Metric                | Value                     |
|-----------------------|---------------------------|
| Avg.                  | 25.28   |
| ARC (25-shot)         | 21.59          |
| HellaSwag (10-shot)   | 27.29    |
| MMLU (5-shot)         | 25.9         |
| TruthfulQA (0-shot)   | 47.06   |
| Winogrande (5-shot)   | 51.46   |
| GSM8K (5-shot)        | 0.3        |

res:

vllm (pretrained=LLMs/meta-llama/Meta-Llama-3.1-8B-Instruct,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.95,data_parallel_size=1), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto
**No multi-turn**
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7862|±  |0.0113|
|     |       |strict-match    |     5|exact_match|↑  |0.7619|±  |0.0117|
**With multi-turn**
vllm (pretrained=LLMs/meta-llama/Meta-Llama-3.1-8B-Instruct,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.95,data_parallel_size=1), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8340|±  |0.0102|
|     |       |strict-match    |     5|exact_match|↑  |0.8089|±  |0.0108|







