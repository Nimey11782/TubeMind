#This function takes a youtube URL ,extracts video id, loads the transcript, converts
#it into plain text , which is wrapped inside a langchain document and returns it safely

import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.documents import Document
from indexing.window_chunking import create_windowed_chunks

def extract_video_id(url: str) -> str: # it is used to extract yt video id from url
    patterns = [
        r"v=([0-9A-Za-z_-]{11})",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
        r"shorts/([0-9A-Za-z_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError("Invalid YouTube URL")  

def ingest_youtube_transcript(url: str) -> list[Document]:# ->list[document] is the return type
    video_id = extract_video_id(url)
    api = YouTubeTranscriptApi()

    try:
        transcript_list = api.fetch(video_id, languages=["en"])
        #transcript_list is a list in which each chunk has text ,start time ,duration 
        #we need all text,start and duration so preserving them in document
        #The system preserves transcript segment + timestamp + duration + video source instead of one giant text
        documents=[]
        for idx,chunk in enumerate(transcript_list):
            text = chunk.text.strip()

            if not text:
                continue

            if text.startswith("[") and text.endswith("]"):
                continue
            doc=Document(
                page_content=chunk.text,
                metadata={
                    "source": "youtube",
                    "video_id": video_id,
                    "start": chunk.start,
                    "duration": chunk.duration,
                    "segment_id":idx
                }
            )

            documents.append(doc)
        #Document is langchain standard data format which has page content and meta data
        #RAG pipeline understands this document 

        return documents
        #langchain expects list of documents so return [doc] and not doc

    except TranscriptsDisabled: #if video captions are off
        raise RuntimeError("No captions available for this video")
    

# if __name__ == "__main__":
#     test_url = "https://www.youtube.com/watch?v=ULvplwBTbQk"
#     docs = ingest_youtube_transcript(test_url)

#     print(type(docs))
#     print(type(docs[0]))
#     print(docs[0].metadata)
#     print(docs[0].page_content[:300])

#     windowed_docs = create_windowed_chunks(docs)

#     print(windowed_docs[0].page_content)
#     print(windowed_docs[0].metadata)
#     # for i in docs[:30]:
#     #     print(i)
