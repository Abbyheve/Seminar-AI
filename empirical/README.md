# Empirical part: FinQA with recent local LLMs

Evaluates three local models (via Ollama) on a fixed sample of 200 FinQA test questions in three settings:

- **FinBen protocol**: FinBen's exact prompts and scoring rule (from their dataset `TheFinAI/flare-finqa` and
  their code), so the results are directly comparable with FinBen's Table 3 (Xie et al., 2024)
- **Chain-of-thought** and **tool use** (the model writes Python that is executed, with one repair round)

The same responses are also scored with a strict and a tolerant number match, to show how much the score
depends on the scoring rule.

| File | Purpose |
|---|---|
| `finqa_eval.py` | Shared code: data loading, prompts, Ollama calls, code execution, answer parsing, scoring |
| `01_run_experiments.ipynb` | Pilot (speed check) and main run; writes raw answers to `results/*.jsonl` (resumable) |
| `02_analysis.ipynb` | Scoring, tables, figures, significance tests, error analysis; no model calls |

Outputs: `results/` (raw answers), `tables/` (CSV and LaTeX `.tex` for `\input{}`, needs `\usepackage{booktabs}`), `figures/` (vector PDF for `\includegraphics`).

## Running

```bash
pip install -r requirements.txt
ollama pull qwen3.5:9b && ollama pull gemma4:26b && ollama pull qwen3-coder:30b
jupyter lab
```

Then run `01_run_experiments.ipynb` (pilot first, then the main run), then `02_analysis.ipynb`.
