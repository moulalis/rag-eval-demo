"""Run the exam: test search and answers separately, then save a scorecard.

Usage:  python evaluate.py --prompt v1     (the original prompt)
        python evaluate.py --prompt v2     (the stricter, fixed prompt)
"""
import argparse
import json
import re
from pathlib import Path

import llm
import rag_app

JUDGE_PROMPT = """You are grading an AI assistant's answer for an airline.

Question: {question}
Policy pages the assistant was given:
{context}
Assistant's answer: {answer}
Reference answer: {reference}

Grade two things:
- faithful: true only if EVERY claim in the answer is supported by the policy pages.
  Saying "I don't know" is faithful when the pages don't contain the answer.
- relevant: true if the answer addresses the question and agrees with the reference answer.

Reply with JSON only, like: {{"faithful": true, "relevant": true, "reason": "one short sentence"}}"""


def judge(question, pages, answer, reference):
    """LLM-as-a-judge: a second model grades the answer."""
    context = "\n\n".join(f"[{p['source']}] {p['text']}" for p in pages) or "(no pages)"
    raw = llm.ask(JUDGE_PROMPT.format(question=question, context=context,
                                      answer=answer, reference=reference))
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        verdict = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        verdict = {}
    return {"faithful": bool(verdict.get("faithful")),
            "relevant": bool(verdict.get("relevant")),
            "reason": verdict.get("reason", "Judge reply could not be read")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", choices=["v1", "v2"], default="v1")
    ap.add_argument("--tests", default="test_set.json")
    args = ap.parse_args()

    tests = json.loads(Path(args.tests).read_text())
    results = []
    print(f"\nRunning {len(tests)} exam questions with prompt {args.prompt} "
          f"({llm.PROVIDER}: {llm.MODEL})\n")

    for t in tests:
        answer, pages = rag_app.answer(t["question"], args.prompt)

        # Test 1: search. Did the right page come back?
        found = [p["source"] for p in pages]
        search_ok = None if t["source"] is None else t["source"] in found

        # Test 2: the answer. Is it faithful to the pages, and relevant?
        verdict = judge(t["question"], pages, answer, t["reference_answer"])
        answer_ok = verdict["faithful"] and verdict["relevant"]

        passed = answer_ok and search_ok is not False
        results.append({**t, "answer": answer, "pages_found": found,
                        "search_ok": search_ok, **verdict, "passed": passed})

        mark = "PASS" if passed else "FAIL"
        search_txt = "n/a (no page has the answer)" if search_ok is None else ("found" if search_ok else "MISSED")
        print(f"[{mark}] Q{t['id']}: {t['question']}")
        print(f"       search: {search_txt}   faithful: {verdict['faithful']}   relevant: {verdict['relevant']}")
        print(f"       answer: {answer[:160]}")
        print(f"       judge:  {verdict['reason']}\n")

    with_source = [r for r in results if r["search_ok"] is not None]
    scores = {
        "search_hit_rate": round(sum(r["search_ok"] for r in with_source) / max(len(with_source), 1), 2),
        "answer_pass_rate": round(sum(r["faithful"] and r["relevant"] for r in results) / len(results), 2),
        "overall_pass_rate": round(sum(r["passed"] for r in results) / len(results), 2),
    }
    Path("results.json").write_text(json.dumps({"prompt": args.prompt, "scores": scores,
                                                "results": results}, indent=2))
    print("Scorecard:", scores)
    print("Saved to results.json")


if __name__ == "__main__":
    main()
