from generation import ask

benchmark_questions = [
    "What are the key takeaways from the discussion?",
]

retrieval_modes = [
    "similarity",
    "mmr",
    "rerank"
]

for mode in retrieval_modes:

    print("\n" + "="*80)
    print(f"TESTING MODE: {mode.upper()}")
    print("="*80)

    for q in benchmark_questions:

        print("\nQUESTION:")
        print(q)

        result = ask(
            q,
            retrieval_mode=mode
        )

        print("\nANSWER:")
        print(result.answer[:300])