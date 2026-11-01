"""A deliberately small RAG app: keyword search (BM25) + an LLM answer."""
import math
import re
from collections import Counter
from pathlib import Path

import llm

DOCS_DIR = Path(__file__).parent / "docs"
TOP_K = 2

# Version 1: the "looks fine in a demo" prompt.
PROMPT_V1 = """You are Cloudhopper Airlines' friendly assistant.
Use the policy excerpts below to help the customer.
Always give the customer a confident, helpful answer.

Policy excerpts:
{context}

Customer question: {question}
Answer:"""

# Version 2: the fix. Answer only from the pages, or say you don't know.
PROMPT_V2 = """You are Cloudhopper Airlines' assistant.
Answer using ONLY the policy excerpts below.
If the answer is not in the excerpts, reply exactly: "I don't know based on our policies."

Policy excerpts:
{context}

Customer question: {question}
Answer:"""

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


def _tokens(text):
    return re.findall(r"[a-z0-9$]+", text.lower())


def load_chunks():
    """Split every policy page into paragraph-sized chunks."""
    chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        for para in path.read_text().split("\n\n"):
            para = para.strip()
            if para and not para.startswith("#"):
                chunks.append({"source": path.name, "text": para})
    return chunks


CHUNKS = load_chunks()


def search(question, k=TOP_K, k1=1.5, b=0.75):
    """Rank chunks with BM25, a classic keyword-search formula."""
    docs = [_tokens(c["text"]) for c in CHUNKS]
    avg_len = sum(map(len, docs)) / len(docs)
    df = Counter(t for d in docs for t in set(d))
    n = len(docs)
    scores = []
    for chunk, words in zip(CHUNKS, docs):
        tf = Counter(words)
        score = 0.0
        for t in _tokens(question):
            if t in tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(words) / avg_len))
        scores.append((score, chunk))
    scores.sort(key=lambda s: s[0], reverse=True)
    return [c for s, c in scores[:k] if s > 0]


def answer(question, prompt_version="v1"):
    """Retrieve, augment, generate. Returns the answer and the pages it used."""
    pages = search(question)
    context = "\n\n".join(f"[{p['source']}] {p['text']}" for p in pages) or "(no matching pages)"
    reply = llm.ask(PROMPTS[prompt_version].format(context=context, question=question))
    return reply, pages


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "How much does my first checked bag cost?"
    reply, pages = answer(q)
    print("Pages:", [p["source"] for p in pages])
    print("Answer:", reply)
