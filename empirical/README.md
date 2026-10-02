# Empirical part: FinQA with recent local LLMs

Evaluates three local models (via Ollama) on a fixed sample of 200 FinQA test questions, in two settings
(chain-of-thought vs. tool use with Python execution), and compares the results with FinBen (Xie et al., 2024).

| File | Purpose |
|---|---|
| `finqa_eval.py` | Shared code: data loading, prompts, Ollama calls, code execution, answer parsing, scoring |
| `01_run_experiments.ipynb` | Pilot (speed check) and main run; writes raw answers to `results/*.jsonl` (resumable) |
| `02_analysis.ipynb` | Scoring, tables, figures, significance tests, error analysis; no model calls |

Outputs: `results/` (raw answers), `tables/` (CSV), `figures/` (PDF for the paper).

## Running

```bash
pip install -r requirements.txt
ollama pull qwen3.5:9b && ollama pull gemma4:26b && ollama pull qwen3-coder:30b
jupyter lab
```

Then run `01_run_experiments.ipynb` (pilot first, then the main run), then `02_analysis.ipynb`.
