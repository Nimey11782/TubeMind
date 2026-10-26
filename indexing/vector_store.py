import os

import numpy as np
import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from indexing.embedding import get_embeddings

load_dotenv()
_DATABASE_URL = os.getenv("DATABASE_URL")


def _connect():
    if not _DATABASE_URL:
        raise RuntimeError("DATABASE_URL is required for PostgreSQL vector storage")
    connection = psycopg.connect(_DATABASE_URL)
    register_vector(connection)
    return connection


def verify_vector_database():
    if not _DATABASE_URL:
        raise RuntimeError("DATABASE_URL is required for PostgreSQL vector storage")

    with _connect() as connection:
        connection.execute("SELECT 1 FROM transcript_chunks LIMIT 0")


def store_embeddings(
    docs: list[Document],
    user_id: int,
    chat_id: int,
) -> None:
    if not docs:
        raise ValueError("Cannot create a vector index without transcript chunks")

    vectors = get_embeddings().embed_documents(
        [doc.page_content for doc in docs]
    )
    rows = [
        (
            user_id,
            chat_id,
            str(doc.metadata["chunk_id"]),
            str(doc.metadata["video_id"]),
            doc.page_content,
            Jsonb(doc.metadata),
            Vector(np.asarray(vector, dtype=np.float32)),
        )
        for doc, vector in zip(docs, vectors, strict=True)
    ]

    with _connect() as connection:
        connection.execute(
            "SELECT pg_advisory_xact_lock(%s, %s)",
            (user_id, chat_id),
        )
        connection.execute(
            "DELETE FROM transcript_chunks WHERE user_id = %s AND chat_id = %s",
            (user_id, chat_id),
        )
        connection.cursor().executemany("""
            INSERT INTO transcript_chunks
                (user_id, chat_id, chunk_id, video_id, content, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, rows)


def load_chat_documents(user_id: int, chat_id: int) -> list[Document]:
    with _connect() as connection:
        rows = connection.execute("""
            SELECT content, metadata
            FROM transcript_chunks
            WHERE user_id = %s AND chat_id = %s
            ORDER BY chunk_id
        """, (user_id, chat_id)).fetchall()
    return [
        Document(page_content=row[0], metadata=row[1])
        for row in rows
    ]


def has_chat_vectors(user_id: int, chat_id: int) -> bool:
    with _connect() as connection:
        row = connection.execute("""
            SELECT EXISTS (
                SELECT 1 FROM transcript_chunks
                WHERE user_id = %s AND chat_id = %s
            )
        """, (user_id, chat_id)).fetchone()
    return row[0]


def search_similar(
    query: str,
    user_id: int,
    chat_id: int,
    limit: int,
) -> list[Document]:
    query_vector = Vector(
        np.asarray(get_embeddings().embed_query(query), dtype=np.float32)
    )
    with _connect() as connection:
        connection.execute("SET hnsw.iterative_scan = strict_order")
        rows = connection.execute("""
            SELECT content, metadata
            FROM transcript_chunks
            WHERE user_id = %s AND chat_id = %s
            ORDER BY embedding <=> %s
            LIMIT %s
        """, (user_id, chat_id, query_vector, limit)).fetchall()
    return [
        Document(page_content=row[0], metadata=row[1])
        for row in rows
    ]


def search_mmr(
    query: str,
    user_id: int,
    chat_id: int,
    limit: int,
    fetch_limit: int,
    lambda_mult: float,
) -> list[Document]:
    from langchain_core.vectorstores.utils import maximal_marginal_relevance

    query_vector = np.asarray(get_embeddings().embed_query(query), dtype=np.float32)
    pg_query_vector = Vector(query_vector)
    with _connect() as connection:
        connection.execute("SET hnsw.iterative_scan = strict_order")
        rows = connection.execute("""
            SELECT content, metadata, embedding
            FROM transcript_chunks
            WHERE user_id = %s AND chat_id = %s
            ORDER BY embedding <=> %s
            LIMIT %s
        """, (user_id, chat_id, pg_query_vector, fetch_limit)).fetchall()
    if not rows:
        return []

    indexes = maximal_marginal_relevance(
        query_vector,
        np.asarray([row[2].to_numpy() for row in rows], dtype=np.float32),
        lambda_mult=lambda_mult,
        k=limit,
    )
    return [
        Document(page_content=rows[index][0], metadata=rows[index][1])
        for index in indexes
    ]
