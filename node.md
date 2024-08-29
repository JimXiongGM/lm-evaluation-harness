

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
    --tasks truthfulqa --batch_size auto --num_fewshot 0 \
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

vllm (pretrained=LLMs/meta-llama/Meta-Llama-3.1-8B-Instruct,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.95,data_parallel_size=1), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto
|                 Tasks                 |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|---------------------------------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k                                  |      3|flexible-extract|     5|exact_match|↑  |0.8340|±  |0.0102|
|                                       |       |strict-match    |     5|exact_match|↑  |0.8089|±  |0.0108|
|mmlu                                   |      2|none            |      |acc        |↑  |0.6831|±  |0.0037|
| - humanities                          |      2|none            |      |acc        |↑  |0.6438|±  |0.0067|
|  - formal_logic                       |      1|none            |     5|acc        |↑  |0.5794|±  |0.0442|
|  - high_school_european_history       |      1|none            |     5|acc        |↑  |0.7455|±  |0.0340|
|  - high_school_us_history             |      1|none            |     5|acc        |↑  |0.8235|±  |0.0268|
|  - high_school_world_history          |      1|none            |     5|acc        |↑  |0.8481|±  |0.0234|
|  - international_law                  |      1|none            |     5|acc        |↑  |0.8182|±  |0.0352|
|  - jurisprudence                      |      1|none            |     5|acc        |↑  |0.7870|±  |0.0396|
|  - logical_fallacies                  |      1|none            |     5|acc        |↑  |0.8160|±  |0.0304|
|  - moral_disputes                     |      1|none            |     5|acc        |↑  |0.7514|±  |0.0233|
|  - moral_scenarios                    |      1|none            |     5|acc        |↑  |0.5587|±  |0.0166|
|  - philosophy                         |      1|none            |     5|acc        |↑  |0.7395|±  |0.0249|
|  - prehistory                         |      1|none            |     5|acc        |↑  |0.7654|±  |0.0236|
|  - professional_law                   |      1|none            |     5|acc        |↑  |0.4987|±  |0.0128|
|  - world_religions                    |      1|none            |     5|acc        |↑  |0.8421|±  |0.0280|
| - other                               |      2|none            |      |acc        |↑  |0.7441|±  |0.0075|
|  - business_ethics                    |      1|none            |     5|acc        |↑  |0.7700|±  |0.0423|
|  - clinical_knowledge                 |      1|none            |     5|acc        |↑  |0.7623|±  |0.0262|
|  - college_medicine                   |      1|none            |     5|acc        |↑  |0.6647|±  |0.0360|
|  - global_facts                       |      1|none            |     5|acc        |↑  |0.3800|±  |0.0488|
|  - human_aging                        |      1|none            |     5|acc        |↑  |0.6951|±  |0.0309|
|  - management                         |      1|none            |     5|acc        |↑  |0.8155|±  |0.0384|
|  - marketing                          |      1|none            |     5|acc        |↑  |0.8889|±  |0.0206|
|  - medical_genetics                   |      1|none            |     5|acc        |↑  |0.8100|±  |0.0394|
|  - miscellaneous                      |      1|none            |     5|acc        |↑  |0.8429|±  |0.0130|
|  - nutrition                          |      1|none            |     5|acc        |↑  |0.7941|±  |0.0232|
|  - professional_accounting            |      1|none            |     5|acc        |↑  |0.5426|±  |0.0297|
|  - professional_medicine              |      1|none            |     5|acc        |↑  |0.7684|±  |0.0256|
|  - virology                           |      1|none            |     5|acc        |↑  |0.5241|±  |0.0389|
| - social sciences                     |      2|none            |      |acc        |↑  |0.7748|±  |0.0074|
|  - econometrics                       |      1|none            |     5|acc        |↑  |0.5526|±  |0.0468|
|  - high_school_geography              |      1|none            |     5|acc        |↑  |0.8384|±  |0.0262|
|  - high_school_government_and_politics|      1|none            |     5|acc        |↑  |0.9016|±  |0.0215|
|  - high_school_macroeconomics         |      1|none            |     5|acc        |↑  |0.6897|±  |0.0235|
|  - high_school_microeconomics         |      1|none            |     5|acc        |↑  |0.7899|±  |0.0265|
|  - high_school_psychology             |      1|none            |     5|acc        |↑  |0.8661|±  |0.0146|
|  - human_sexuality                    |      1|none            |     5|acc        |↑  |0.8168|±  |0.0339|
|  - professional_psychology            |      1|none            |     5|acc        |↑  |0.7206|±  |0.0182|
|  - public_relations                   |      1|none            |     5|acc        |↑  |0.6818|±  |0.0446|
|  - security_studies                   |      1|none            |     5|acc        |↑  |0.7224|±  |0.0287|
|  - sociology                          |      1|none            |     5|acc        |↑  |0.8259|±  |0.0268|
|  - us_foreign_policy                  |      1|none            |     5|acc        |↑  |0.8600|±  |0.0349|
| - stem                                |      2|none            |      |acc        |↑  |0.5921|±  |0.0084|
|  - abstract_algebra                   |      1|none            |     5|acc        |↑  |0.3900|±  |0.0490|
|  - anatomy                            |      1|none            |     5|acc        |↑  |0.6593|±  |0.0409|
|  - astronomy                          |      1|none            |     5|acc        |↑  |0.7697|±  |0.0343|
|  - college_biology                    |      1|none            |     5|acc        |↑  |0.8194|±  |0.0322|
|  - college_chemistry                  |      1|none            |     5|acc        |↑  |0.5100|±  |0.0502|
|  - college_computer_science           |      1|none            |     5|acc        |↑  |0.5300|±  |0.0502|
|  - college_mathematics                |      1|none            |     5|acc        |↑  |0.3800|±  |0.0488|
|  - college_physics                    |      1|none            |     5|acc        |↑  |0.4510|±  |0.0495|
|  - computer_security                  |      1|none            |     5|acc        |↑  |0.7900|±  |0.0409|
|  - conceptual_physics                 |      1|none            |     5|acc        |↑  |0.6170|±  |0.0318|
|  - electrical_engineering             |      1|none            |     5|acc        |↑  |0.6345|±  |0.0401|
|  - elementary_mathematics             |      1|none            |     5|acc        |↑  |0.4921|±  |0.0257|
|  - high_school_biology                |      1|none            |     5|acc        |↑  |0.8065|±  |0.0225|
|  - high_school_chemistry              |      1|none            |     5|acc        |↑  |0.6207|±  |0.0341|
|  - high_school_computer_science       |      1|none            |     5|acc        |↑  |0.7600|±  |0.0429|
|  - high_school_mathematics            |      1|none            |     5|acc        |↑  |0.4074|±  |0.0300|
|  - high_school_physics                |      1|none            |     5|acc        |↑  |0.4768|±  |0.0408|
|  - high_school_statistics             |      1|none            |     5|acc        |↑  |0.5648|±  |0.0338|
|  - machine_learning                   |      1|none            |     5|acc        |↑  |0.5179|±  |0.0474|
|winogrande                             |      1|none            |     5|acc        |↑  |0.7743|±  |0.0117|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.6831|±  |0.0037|
| - humanities     |      2|none  |      |acc   |↑  |0.6438|±  |0.0067|
| - other          |      2|none  |      |acc   |↑  |0.7441|±  |0.0075|
| - social sciences|      2|none  |      |acc   |↑  |0.7748|±  |0.0074|
| - stem           |      2|none  |      |acc   |↑  |0.5921|±  |0.0084|


vllm (pretrained=LLMs/meta-llama/Meta-Llama-3.1-8B-Instruct,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.95,data_parallel_size=1), gen_kwargs: (None), limit: None, num_fewshot: 25, batch_size: auto
|    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge|      1|none  |    25|acc     |↑  |0.6058|±  |0.0143|
|             |       |none  |    25|acc_norm|↑  |0.5819|±  |0.0144|
|arc_easy     |      1|none  |    25|acc     |↑  |0.8194|±  |0.0079|
|             |       |none  |    25|acc_norm|↑  |0.7420|±  |0.0090|



vllm (pretrained=LLMs/meta-llama/Meta-Llama-3.1-8B-Instruct,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.95,data_parallel_size=1), gen_kwargs: (None), limit: None, num_fewshot: 10, batch_size: auto
|  Tasks  |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|---------|------:|------|-----:|--------|---|-----:|---|-----:|
|hellaswag|      1|none  |    10|acc     |↑  |0.6071|±  |0.0049|
|         |       |none  |    10|acc_norm|↑  |0.7730|±  |0.0042|


vllm (pretrained=LLMs/meta-llama/Meta-Llama-3.1-8B-Instruct,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.95,data_parallel_size=1), gen_kwargs: (None), limit: None, num_fewshot: 0, batch_size: auto
|    Tasks     |Version|Filter|n-shot|  Metric   |   | Value |   |Stderr|
|--------------|------:|------|-----:|-----------|---|------:|---|-----:|
|truthfulqa_gen|      3|none  |     0|bleu_acc   |↑  | 0.6206|±  |0.0170|
|              |       |none  |     0|bleu_diff  |↑  |15.1732|±  |1.1029|
|              |       |none  |     0|bleu_max   |↑  |36.0844|±  |0.8798|
|              |       |none  |     0|rouge1_acc |↑  | 0.6034|±  |0.0171|
|              |       |none  |     0|rouge1_diff|↑  |21.4229|±  |1.5974|
|              |       |none  |     0|rouge1_max |↑  |60.6926|±  |0.9970|
|              |       |none  |     0|rouge2_acc |↑  | 0.5520|±  |0.0174|
|              |       |none  |     0|rouge2_diff|↑  |22.7438|±  |1.6518|
|              |       |none  |     0|rouge2_max |↑  |48.2655|±  |1.2086|
|              |       |none  |     0|rougeL_acc |↑  | 0.6034|±  |0.0171|
|              |       |none  |     0|rougeL_diff|↑  |21.4863|±  |1.6128|
|              |       |none  |     0|rougeL_max |↑  |58.9137|±  |1.0282|
|truthfulqa_mc1|      2|none  |     0|acc        |↑  | 0.3647|±  |0.0169|
|truthfulqa_mc2|      2|none  |     0|acc        |↑  | 0.5397|±  |0.0150|


