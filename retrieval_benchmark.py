# retrieval_benchmark.py

from retriever import retrieve_chunks

benchmark_questions = [
    "What does he say about Meta?",
    "What is his opinion on open source AI?",
    "What concerns does he raise about children using AI?",
    "What are the key takeaways from the discussion?",
    "What criticism does he make?"
]

modes = [
    "similarity",
    "mmr",
    "rerank"
]

for mode in modes:

    print("\n" + "=" * 80)
    print(f"MODE: {mode.upper()}")
    print("=" * 80)

    for q in benchmark_questions:

        print(f"\nQUESTION: {q}")

        docs = retrieve_chunks(
            q,
            mode=mode
        )

        print("\nTOP RETRIEVED CHUNK:")

        if docs:
            print(docs[0].page_content[:300])