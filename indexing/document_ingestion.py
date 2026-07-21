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

import requests

def get_free_proxies():
    try:
        res = requests.get("https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=5000&country=all&ssl=all&anonymity=all")
        if res.status_code == 200:
            proxies = res.text.strip().split("\r\n")
            return [p for p in proxies if p][:10]
    except Exception:
        pass
    return []

def ingest_youtube_transcript(url: str) -> list[Document]:
    video_id = extract_video_id(url)
    
    transcript_list = None
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id, languages=["en"])
    except Exception as e:
        if "TranscriptsDisabled" in str(type(e)):
            raise RuntimeError("No captions available for this video")
            
        print(f"Direct fetch failed: {type(e).__name__}. Trying proxies...")
        proxies = get_free_proxies()
        
        for proxy_ip in proxies:
            print(f"Trying proxy: {proxy_ip}")
            try:
                session = requests.Session()
                session.proxies = {"http": f"http://{proxy_ip}", "https": f"http://{proxy_ip}"}
                
                api = YouTubeTranscriptApi(http_client=session)
                transcript_list = api.fetch(video_id, languages=["en"])
                
                print("Proxy successful!")
                break
            except Exception as proxy_e:
                print(f"Proxy failed: {type(proxy_e).__name__}")
                
        if not transcript_list:
            raise RuntimeError("Failed to fetch transcript. YouTube may be blocking the server and all proxies failed.")

    documents=[]
    for idx, chunk in enumerate(transcript_list):
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

    return documents
    

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
