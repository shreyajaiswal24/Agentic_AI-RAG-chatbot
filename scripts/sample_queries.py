"""Run the sample queries against a running API and print a compact report.

    uvicorn app.main:app --port 8000      # in one terminal
    python -m scripts.sample_queries      # in another
"""

import argparse
import json

import httpx

SAMPLE_QUERIES = [
    # (question, what it exercises)
    ("What is Agentic AI and how does it differ from traditional AI tools?", "directly answered"),
    ("What are the core pillars of an Agentic AI system, from perception to execution?", "single section"),
    ("What challenges do multi-agent systems face and how can they be mitigated?", "multi-chunk synthesis"),
    ("Which industries does the eBook give Agentic AI use cases for?", "list spread across pages"),
    ("What is the capital of France?", "not in the PDF -> refusal"),
    ("Who is the CEO of Konverge AI and when was the company founded?", "hallucination bait"),
    ("Tell me about agents.", "vague"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--json", action="store_true", help="print raw JSON responses")
    args = parser.parse_args()

    with httpx.Client(base_url=args.url, timeout=60) as client:
        for question, purpose in SAMPLE_QUERIES:
            body = client.post("/ask", json={"question": question}).json()
            print("=" * 100)
            print(f"Q: {question}   [{purpose}]")
            print(f"route={body['route']}  grounded={body['grounded']}  relevance_score={body['relevance_score']}")
            print(f"pages={[c['page'] for c in body['retrieved_chunks']]}  scores={[c['score'] for c in body['retrieved_chunks']]}")
            print(f"A: {body['answer']}")
            if args.json:
                print(json.dumps(body, indent=2))


if __name__ == "__main__":
    main()
