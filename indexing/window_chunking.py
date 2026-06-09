from langchain_core.documents import Document

def create_windowed_chunks(
    docs,
    window_size=20,
    stride=10
):
    """
    Creates overlapping transcript windows.

    Example:
    window_size = 5
    stride = 3

    Windows:
    0-4
    3-7
    6-10
    """

    merged_docs = []

    for i in range(0, len(docs), stride):

        # current sliding window
        window = docs[i:i + window_size]

        # if window empty
        if not window:
            continue

        # merge text from all docs in window
        merged_text = " ".join(
            doc.page_content for doc in window
        )

        # metadata
        start_time = window[0].metadata["start"]

        end_time = (
            window[-1].metadata["start"]
            + window[-1].metadata["duration"]
        )

        merged_doc = Document(
            page_content=merged_text,
            metadata={
                "source": "youtube",
                "video_id": window[0].metadata["video_id"],

                # temporal grounding
                "start": start_time,
                "end": end_time,

                # segment tracking
                "segment_start": window[0].metadata["segment_id"],
                "segment_end": window[-1].metadata["segment_id"]
            }
        )

        merged_docs.append(merged_doc)

    return merged_docs
