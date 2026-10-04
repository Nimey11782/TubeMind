import re
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.documents import Document


def extract_video_id(url: str) -> str:
    parsed_url = urlparse(url.strip())
    hostname = (parsed_url.hostname or "").lower()
    video_id = None

    if hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed_url.path == "/watch":
            video_id = parse_qs(parsed_url.query).get("v", [None])[0]
        else:
            match = re.match(
                r"^/(?:shorts|embed|live)/([0-9A-Za-z_-]{11})(?:/|$)",
                parsed_url.path,
            )
            if match:
                video_id = match.group(1)
    elif hostname == "youtu.be":
        video_id = parsed_url.path.strip("/").split("/", 1)[0]

    if video_id and re.fullmatch(r"[0-9A-Za-z_-]{11}", video_id):
        return video_id
    raise ValueError("Invalid YouTube URL")

def ingest_youtube_transcript(url: str) -> list[Document]:
    video_id = extract_video_id(url)

    try:
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    except TranscriptsDisabled as exc:
        raise RuntimeError("No captions available for this video") from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch the English transcript for video {video_id}: {exc}"
        ) from exc

    documents = []
    for idx, chunk in enumerate(transcript_list):
        text = chunk.text.strip()

        if not text:
            continue

        if text.startswith("[") and text.endswith("]"):
            continue

        doc = Document(
            page_content=text,
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
