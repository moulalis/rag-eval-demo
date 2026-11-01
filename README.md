# RAG Evaluation Demo (How It Runs, video 2)

A tiny, from-scratch RAG evaluation system for a made-up airline, Cloudhopper Airlines.
Pure Python standard library: no packages to install.

## Files
| File | What it does |
| --- | --- |
| `docs/` | Four short policy pages: baggage, cancellations, pets, bereavement |
| `test_set.json` | Step 1: the exam paper (questions, correct answers, source pages) |
| `rag_app.py` | The RAG app: BM25 keyword search + an LLM answer, with two prompt versions |
| `evaluate.py` | Steps 2 and 3: tests search and answers separately, saves `results.json` |
| `quality_gate.py` | Step 4: blocks the release (exit code 1) if scores drop below the line |
| `.github/workflows/rag-eval.yml` | Runs the exam and the gate on every pull request |

## Run it
1. Choose a model:
   - Free and local: install [Ollama](https://ollama.com), then `ollama pull llama3.1:8b`
   - Or a hosted API: `export LLM_PROVIDER=openai` (with `OPENAI_API_KEY`)
     or `export LLM_PROVIDER=anthropic` (with `ANTHROPIC_API_KEY`)
   - Optional: `export LLM_MODEL=<model name>` (check your provider's docs for current names)
2. Run the exam with the original prompt, then the gate:
   ```
   python evaluate.py --prompt v1
   python quality_gate.py
   ```
3. Run it again with the fixed prompt:
   ```
   python evaluate.py --prompt v2
   python quality_gate.py
   ```

## Notes
- Real models vary. With prompt v1, most models invent an answer to the Wi-Fi question,
  but if yours doesn't, try a smaller model or run it a couple of times before recording.
- In a real project, grow `test_set.json` to 50+ questions from real support tickets and search logs,
  and have a person spot-check the judge's grades every week.
