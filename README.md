# Multi-Document Analysis System

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Run

```bash
streamlit run main.py
```

## Evaluation

Run RAGAS evaluation:

```bash
python run_evaluation.py
```

Results saved to `results/evaluation_results_YYYYMMDD_HHMMSS.json`

### Metrics

- **Correctness**: Answer accuracy vs ground truth
- **Faithfulness**: Answer grounded in context
- **Precision**: Context relevance
- **Relevancy**: Answer relevance to query

### Result Format

```json
{
  "model": "gpt-4.1",
  "date": "2025-10-28T15:38:11Z",
  "metrics": {
    "correctness": 0.873,
    "faithfulness": 0.783,
    "precision": 0.932,
    "relevancy": 0.870
  },
  "test_cases": [
    {
      "query": "Compare revenue...",
      "expected": "Accenture: +8%...",
      "actual": "Accenture: +8%...",
      "scores": {
        "correctness": 1.0,
        "faithfulness": 1.0,
        "precision": 1.0,
        "relevancy": 0.904
      }
    }
  ]
}
```
