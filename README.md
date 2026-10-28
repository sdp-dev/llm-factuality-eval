# LLM Factuality Evaluation

How factually consistent are open-weight LLMs when summarizing medical literature? This project compares **LLaMA-3.1-8B-Instruct**, **Mistral-7B-Instruct**, and **DeepSeek-R1-Distill-Qwen-7B** on summaries of COVID-19 side-effect papers from [CORD-19](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge), scored with a QAGS-style consistency pipeline.

Full writeup: [`writeup.pdf`](writeup.pdf)

## Method

1. **Data:** Parsed 100k CORD-19 PMC papers, filtered to 51k mentioning side effects / adverse effects / complications / symptoms, and sampled 250 for evaluation.
2. **Summaries:** Each model summarized each paper (inputs truncated to 8K tokens, temperature 0.2, max 1,024 new tokens, bfloat16).
3. **QAGS scoring:** FLAN-T5 generates questions from each summary; an extractive QA model (RoBERTa) answers them against both the summary and the source paper; answers are compared with BERTScore F1.

## Results

| Model | Summaries | Scored (non-NaN) | QAGS (NaN = 0) | QAGS (excl. NaN) |
|---|---|---|---|---|
| Mistral-7B-Instruct | 250 | 118 | 0.2138 | **0.4529** |
| LLaMA-3.1-8B-Instruct | 250 | 74 | 0.1126 | 0.3805 |
| DeepSeek-R1-Distill-Qwen-7B | 250 | 94 | 0.1246 | 0.3313 |

Mistral performed best, but every model stayed under 0.50 F1. All three also tended to turn hedged findings ("these findings suggest") into confident claims.

## Repo layout

```
notebooks/
  dataset_analysis.ipynb     CORD-19 loading, filtering, and stats
  summary_generation.ipynb   Summary generation with the three models
  qags_eval.ipynb            QAGS question generation, QA, and scoring
results/
  final_subset.csv           The 250 evaluated papers
  mistral-deepseek-llama-250.csv  Summaries from all three models
  qags_results.csv           Per-paper QAGS scores
  *_test_*.csv               Intermediate LLaMA runs
writeup.pdf
```

The notebooks were written for Google Colab (GPU, Google Drive paths, Hugging Face login). The full filtered dataset (~1.2 GB) isn't included; rebuild it from CORD-19 with `dataset_analysis.ipynb`.

## Authors

Elena Lukac, Diya Nair, Sachi Patel, Adneen Talib
