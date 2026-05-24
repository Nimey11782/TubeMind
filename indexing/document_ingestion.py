#This function takes a youtube URL ,extracts video id, loads the transcript, converts
#it into plain text , which is wrapped inside a langchain document and returns it safely

import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.documents import Document

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
        text = " ".join(chunk.text for chunk in transcript_list)
        #transcript_list is a list in which each chunk has text ,start time ,duration 
        #we only need text which is extracted and stored in text variable - chunk text is converted into plain text

        doc = Document(
            page_content=text,
            metadata={
                "source": "youtube",
                "video_id": video_id
            }
        )
        #Document is langchain standard data format which has page content and meta data
        #RAG pipeline understands this document 

        return [doc]
        #langchain expects list of documents so return [doc] and not doc

    except TranscriptsDisabled: #if video captions are off
        raise RuntimeError("No captions available for this video")
    

if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=ULvplwBTbQk"
    docs = ingest_youtube_transcript(test_url)

    print(type(docs))
    print(type(docs[0]))
    print(docs[0].metadata)
    print(docs[0].page_content[:300])
