import re

from langchain_core.documents import Document


def create_windowed_chunks(
    docs: list[Document],
    chunk_size_words: int = 180,
    overlap_words: int = 40,
) -> list[Document]:
    if chunk_size_words < 1:
        raise ValueError("chunk_size_words must be greater than zero")
    if overlap_words < 0 or overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be between zero and chunk_size_words")

    segments = []
    for doc in docs:
        text = re.sub(r"\s+", " ", doc.page_content).strip()
        if not text:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            words = sentence.split()
            for start in range(0, len(words), chunk_size_words):
                segment_text = " ".join(words[start:start + chunk_size_words])
                if segment_text:
                    segments.append((segment_text, doc.metadata, len(segment_text.split())))

    chunks = []
    start = 0
    while start < len(segments):
        end = start
        word_count = 0
        while end < len(segments) and (
            word_count + segments[end][2] <= chunk_size_words or end == start
        ):
            word_count += segments[end][2]
            end += 1

        window = segments[start:end]
        first_metadata = window[0][1]
        last_metadata = window[-1][1]
        chunk_index = len(chunks)
        video_id = first_metadata["video_id"]
        segment_start = first_metadata["segment_id"]
        segment_end = last_metadata["segment_id"]
        start_time = first_metadata["start"]
        end_time = last_metadata["start"] + last_metadata["duration"]

        chunks.append(
            Document(
                page_content=" ".join(segment[0] for segment in window),
                metadata={
                    "source": "youtube",
                    "video_id": video_id,
                    "start": start_time,
                    "end": end_time,
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "chunk_id": (
                        f"{video_id}:{segment_start}:{segment_end}:{chunk_index}"
                    ),
                },
            )
        )

        if end == len(segments):
            break

        next_start = end
        overlap_count = 0
        for index in range(end - 1, start, -1):
            segment_words = segments[index][2]
            if overlap_count + segment_words > overlap_words:
                break
            overlap_count += segment_words
            next_start = index
        start = max(start + 1, next_start)

    return chunks
