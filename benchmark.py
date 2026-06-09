from generation import ask

benchmark_questions = [

    # Fact retrieval
    "What are AI concerns?",
    "What does he say about Meta?",
    "What is his opinion on open source AI?",
    "What concerns does he raise about children using AI?",
    "What does he say about AI safety?",

    # Summaries
    "Summarize the video.",
    "Summarize his views on AI.",
    "What are the key takeaways from the discussion?",

    # Reasoning / synthesis
    "What are the benefits and risks of AI according to him?",
    "How does he think AI will affect society?",
    "What future does he want for AI systems?",

    # Edge cases
    "What questions did the host ask?",
    "What examples does he give?",
    "What criticism does he make?",
    "What does he say about the future of AI?"
]

for i, q in enumerate(benchmark_questions, start=1):

    print("\n" + "="*80)
    print(f"QUESTION {i}: {q}")
    print("="*80)

    result = ask(q)

    print("\nANSWER:")
    print(result.answer)

    print("\nCITATIONS:")
    for c in result.citations:
        print(c.timestamp)
        print(c.youtube_link)

    print("\n")