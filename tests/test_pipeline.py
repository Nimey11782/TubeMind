import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from youtube_transcript_api import TranscriptsDisabled

import indexing.embedding as embedding
import indexing.index_pipeline as index_pipeline
import indexing.vector_store as vector_store
import indexing.document_ingestion as document_ingestion
import retriever
from indexing.window_chunking import create_windowed_chunks


class DocumentIngestionTests(unittest.TestCase):
    def test_extract_video_id_from_supported_urls(self):
        video_id = "abcdefghijk"
        urls = (
            f"https://www.youtube.com/watch?v={video_id}",
            f"https://m.youtube.com/watch?feature=share&v={video_id}",
            f"https://youtu.be/{video_id}?si=example",
            f"https://youtube.com/shorts/{video_id}",
            f"https://youtube.com/embed/{video_id}",
            f"https://youtube.com/live/{video_id}",
        )

        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(document_ingestion.extract_video_id(url), video_id)

    def test_rejects_non_youtube_urls(self):
        with self.assertRaises(ValueError):
            document_ingestion.extract_video_id("https://example.com/watch?v=abcdefghijk")

    @patch("indexing.document_ingestion.YouTubeTranscriptApi")
    def test_fetches_and_normalizes_transcript_segments(self, api_class):
        api_class.return_value.fetch.return_value = [
            SimpleNamespace(text="  Hello there.  ", start=1.5, duration=2.0),
            SimpleNamespace(text=" [Music] ", start=3.5, duration=1.0),
            SimpleNamespace(text="   ", start=4.5, duration=1.0),
        ]

        docs = document_ingestion.ingest_youtube_transcript(
            "https://youtu.be/abcdefghijk"
        )

        api_class.return_value.fetch.assert_called_once_with(
            "abcdefghijk", languages=["en"]
        )
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "Hello there.")
        self.assertEqual(docs[0].metadata["segment_id"], 0)

    @patch("indexing.document_ingestion.YouTubeTranscriptApi")
    def test_reports_disabled_transcripts_without_proxy_retries(self, api_class):
        api_class.return_value.fetch.side_effect = TranscriptsDisabled("disabled")

        with self.assertRaisesRegex(RuntimeError, "No captions available"):
            document_ingestion.ingest_youtube_transcript(
                "https://youtu.be/abcdefghijk"
            )

        api_class.assert_called_once_with()
        api_class.return_value.fetch.assert_called_once()


class ChunkingTests(unittest.TestCase):
    def test_creates_overlapping_windows_with_timestamp_metadata(self):
        docs = [
            Document(
                page_content=f"segment {index}.",
                metadata={
                    "video_id": "abcdefghijk",
                    "start": float(index * 2),
                    "duration": 2.0,
                    "segment_id": index,
                },
            )
            for index in range(5)
        ]

        chunks = create_windowed_chunks(
            docs,
            chunk_size_words=4,
            overlap_words=2,
        )

        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0].page_content, "segment 0. segment 1.")
        self.assertEqual(chunks[0].metadata["start"], 0.0)
        self.assertEqual(chunks[0].metadata["end"], 4.0)
        self.assertEqual(chunks[1].metadata["segment_start"], 1)
        self.assertEqual(
            chunks[-1].page_content,
            "segment 3. segment 4.",
        )
        self.assertNotEqual(
            chunks[0].metadata["chunk_id"],
            chunks[1].metadata["chunk_id"],
        )

    def test_rejects_invalid_window_parameters(self):
        with self.assertRaises(ValueError):
            create_windowed_chunks([], chunk_size_words=0)
        with self.assertRaises(ValueError):
            create_windowed_chunks([], overlap_words=180)

    def test_splits_oversized_sentences_without_losing_timestamps(self):
        doc = Document(
            page_content="one two three four five six.",
            metadata={
                "video_id": "abcdefghijk",
                "start": 3.0,
                "duration": 2.0,
                "segment_id": 4,
            },
        )

        chunks = create_windowed_chunks(
            [doc],
            chunk_size_words=3,
            overlap_words=0,
        )

        self.assertEqual(
            [chunk.page_content for chunk in chunks],
            ["one two three", "four five six."],
        )
        self.assertTrue(all(chunk.metadata["start"] == 3.0 for chunk in chunks))


class EmbeddingTests(unittest.TestCase):
    @patch("indexing.embedding.LocalSentenceTransformerEmbeddings")
    def test_reuses_embedding_model(self, embeddings_class):
        with patch.object(embedding, "_embeddings", None):
            first = embedding.get_embeddings()
            second = embedding.get_embeddings()

        self.assertIs(first, second)
        embeddings_class.assert_called_once()

    @patch("indexing.vector_store._connect")
    @patch("indexing.vector_store.get_embeddings")
    def test_stores_vectors_separately_for_each_chat(
        self, get_embeddings, connect
    ):
        connection = Mock()
        connect.return_value.__enter__.return_value = connection
        get_embeddings.return_value.embed_documents.return_value = [[0.1] * 384]
        doc = Document(
            page_content="transcript chunk",
            metadata={
                "chunk_id": "abcdefghijk:0:1:0",
                "video_id": "abcdefghijk",
            },
        )

        vector_store.store_embeddings([doc], user_id=7, chat_id=12)

        self.assertEqual(
            connection.execute.call_args_list[0].args,
            ("SELECT pg_advisory_xact_lock(%s, %s)", (7, 12)),
        )
        self.assertEqual(
            connection.execute.call_args_list[1].args,
            ("DELETE FROM transcript_chunks WHERE user_id = %s AND chat_id = %s", (7, 12)),
        )
        connection.cursor.return_value.executemany.assert_called_once()
        statement, rows = connection.cursor.return_value.executemany.call_args.args
        self.assertIn("INSERT INTO transcript_chunks", statement)
        self.assertEqual(rows[0][:5], (7, 12, "abcdefghijk:0:1:0", "abcdefghijk", "transcript chunk"))

    def test_rejects_empty_index(self):
        with self.assertRaisesRegex(ValueError, "without transcript chunks"):
            vector_store.store_embeddings([], user_id=7, chat_id=12)


class VectorSearchTests(unittest.TestCase):
    @patch("indexing.vector_store._connect")
    @patch("indexing.vector_store.get_embeddings")
    def test_search_uses_cosine_distance_scoped_to_user_and_chat(
        self, get_embeddings, connect
    ):
        metadata = {
            "chunk_id": "abcdefghijk:0:1:0",
            "video_id": "abcdefghijk",
        }
        cursor = Mock()
        cursor.fetchall.return_value = [("relevant chunk", metadata)]
        connection = Mock()
        connection.execute.side_effect = [None, cursor]
        connect.return_value.__enter__.return_value = connection
        get_embeddings.return_value.embed_query.return_value = [0.1] * 384

        results = vector_store.search_similar(
            "question",
            user_id=7,
            chat_id=12,
            limit=20,
        )

        self.assertEqual([doc.page_content for doc in results], ["relevant chunk"])
        search_sql, parameters = connection.execute.call_args_list[1].args
        self.assertIn("embedding <=> %s", search_sql)
        self.assertIn("user_id = %s AND chat_id = %s", search_sql)
        self.assertEqual(parameters[0:2], (7, 12))
        self.assertEqual(parameters[3], 20)


class RetrievalTests(unittest.TestCase):
    @patch("retriever.rerank_documents")
    @patch("retriever.bm25_search")
    @patch("retriever.search_similar")
    def test_default_retrieval_fuses_dense_and_bm25_results(
        self, search_similar, bm25_search, rerank_documents
    ):
        first = Document(
            page_content="dense result",
            metadata={"chunk_id": "first"},
        )
        shared = Document(
            page_content="found by both",
            metadata={"chunk_id": "shared"},
        )
        sparse = Document(
            page_content="keyword result",
            metadata={"chunk_id": "sparse"},
        )
        search_similar.return_value = [first, shared]
        bm25_search.return_value = [shared, sparse]

        result = retriever.retrieve_chunks("question", user_id=7, chat_id=12)

        search_similar.assert_called_once_with(
            "question",
            7,
            12,
            limit=20,
        )
        bm25_search.assert_called_once_with(
            "question",
            7,
            12,
            top_k=20,
        )
        fused_docs = retriever.reciprocal_rank_fusion(
            [first, shared],
            [shared, sparse],
            top_k=20,
        )
        rerank_documents.assert_called_once_with(
            "question",
            fused_docs,
            top_n=4,
        )
        self.assertEqual(result, rerank_documents.return_value)

    @patch("retriever.rerank_documents")
    @patch("retriever.search_mmr")
    def test_rerank_mode_is_explicit(self, search_mmr, rerank_documents):
        documents = [Document(page_content="candidate chunk")]
        search_mmr.return_value = documents

        result = retriever.retrieve_chunks("question", mode="rerank")

        rerank_documents.assert_called_once_with("question", documents, top_n=4)
        self.assertEqual(result, rerank_documents.return_value)
        search_mmr.assert_called_once_with(
            "question",
            1,
            1,
            limit=10,
            fetch_limit=20,
            lambda_mult=0.6,
        )

    @patch("retriever.search_similar")
    def test_rejects_invalid_mode_before_searching(self, search_similar):
        with self.assertRaisesRegex(ValueError, "Unknown retrieval mode"):
            retriever.retrieve_chunks("question", mode="unknown")

        search_similar.assert_not_called()

    def test_ingestion_clears_only_that_chat_bm25_cache(self):
        with patch.object(retriever, "_bm25_indexes", {
            (7, 12): (Mock(), []),
            (7, 13): (Mock(), []),
        }):
            retriever.clear_chat_cache(7, 12)
            self.assertNotIn((7, 12), retriever._bm25_indexes)
            self.assertIn((7, 13), retriever._bm25_indexes)

    def test_reciprocal_rank_fusion_promotes_shared_results(self):
        first = Document(
            page_content="first",
            metadata={"chunk_id": "first"},
        )
        shared = Document(
            page_content="shared",
            metadata={"chunk_id": "shared"},
        )
        third = Document(
            page_content="third",
            metadata={"chunk_id": "third"},
        )

        results = retriever.reciprocal_rank_fusion(
            [first, shared],
            [shared, third],
            top_k=3,
        )

        self.assertEqual(
            [doc.metadata["chunk_id"] for doc in results],
            ["shared", "first", "third"],
        )

    @patch("onnxruntime.InferenceSession")
    @patch("onnxruntime.SessionOptions")
    @patch("flashrank.Ranker")
    def test_flashrank_limits_onnx_cpu_threads(
        self,
        ranker_class,
        session_options_class,
        inference_session,
    ):
        ranker_class.return_value.model_dir = Path("model-cache")
        with patch.object(retriever, "_reranker", None):
            first = retriever._get_reranker()
            second = retriever._get_reranker()

        self.assertIs(first, second)
        options = session_options_class.return_value
        self.assertGreaterEqual(options.intra_op_num_threads, 1)
        self.assertLessEqual(options.intra_op_num_threads, 2)
        self.assertEqual(options.inter_op_num_threads, 1)
        inference_session.assert_called_once()
        self.assertEqual(
            inference_session.call_args.kwargs["sess_options"],
            options,
        )

    @patch("retriever._get_reranker")
    def test_flashrank_reranks_candidates_and_returns_the_best_passage(
        self, get_reranker
    ):
        relevant = Document(
            page_content="The source explains nuclear fusion.",
            metadata={"chunk_id": "relevant"},
        )
        unrelated = Document(
            page_content="A passage about cooking.",
            metadata={"chunk_id": "unrelated"},
        )
        get_reranker.return_value.rerank.return_value = [
            {"id": "relevant", "score": 0.9},
            {"id": "unrelated", "score": 0.1},
        ]

        results = retriever.rerank_documents(
            "What is nuclear fusion?",
            [unrelated, relevant],
            top_n=1,
        )

        self.assertEqual(results, [relevant])
        rerank_request = (
            get_reranker.return_value.rerank.call_args.args[0]
        )
        self.assertEqual(rerank_request.query, "What is nuclear fusion?")
        self.assertEqual(
            [passage["id"] for passage in rerank_request.passages],
            ["unrelated", "relevant"],
        )

    @patch("retriever.load_chat_documents")
    def test_bm25_retrieves_keyword_matches_and_caches_chat_index(
        self, load_chat_documents
    ):
        relevant = Document(
            page_content="The nuclear fusion reaction releases energy.",
            metadata={"chunk_id": "relevant"},
        )
        unrelated = Document(
            page_content="A biography discusses childhood education.",
            metadata={"chunk_id": "unrelated"},
        )
        third = Document(
            page_content="A documentary explains how stars form.",
            metadata={"chunk_id": "third"},
        )
        load_chat_documents.return_value = [relevant, unrelated, third]
        with patch.object(retriever, "_bm25_indexes", {}):
            first_result = retriever.bm25_search("nuclear fusion", 7, 12)
            cached_index = retriever._bm25_indexes[(7, 12)]
            second_result = retriever.bm25_search("nuclear fusion", 7, 12)
            self.assertIs(retriever._bm25_indexes[(7, 12)], cached_index)

        load_chat_documents.assert_called_once_with(7, 12)
        self.assertEqual(first_result[0], relevant)
        self.assertEqual(second_result, first_result)


class GenerationTests(unittest.TestCase):
    @patch("generation.main_chain", new_callable=Mock)
    @patch("generation.get_context_and_docs", return_value=("No context.", []))
    def test_skips_llm_when_retrieval_finds_no_sources(
        self, get_context_and_docs, invoke
    ):
        from generation import ask

        response = ask("What does the video say?")

        self.assertEqual(
            response.answer,
            "This wasn't covered in the video.",
        )
        self.assertEqual(response.citations, [])
        invoke.invoke.assert_not_called()
        get_context_and_docs.assert_called_once()

    def test_streams_answer_tokens_then_numbered_citations(self):
        from generation import stream_answer

        doc = Document(
            page_content="The speaker discusses fusion.",
            metadata={
                "video_id": "abcdefghijk",
                "start": 5.0,
                "end": 9.0,
            },
        )

        async def fake_astream(_):
            yield "Fusion"
            yield " is discussed."

        async def collect_events():
            with patch(
                "generation.get_context_and_docs",
                return_value=("context", [doc]),
            ), patch("generation.main_chain", new_callable=Mock) as chain:
                chain.astream.side_effect = lambda inputs: fake_astream(inputs)
                return [
                    event
                    async for event in stream_answer("Question")
                ]

        events = asyncio.run(collect_events())

        self.assertEqual(
            events,
            [
                {"type": "token", "content": "Fusion"},
                {"type": "token", "content": " is discussed."},
                {
                    "type": "done",
                    "citations": [
                        {
                            "source_number": 1,
                            "timestamp": "00:05 - 00:09",
                            "video_id": "abcdefghijk",
                            "youtube_link": (
                                "https://youtube.com/watch?v=abcdefghijk&t=5"
                            ),
                            "snippet": "The speaker discusses fusion.",
                        }
                    ],
                },
            ],
        )

    def test_citations_have_the_same_source_numbers_as_context(self):
        from generation import build_citations, build_context

        docs = [
            Document(
                page_content="Transcript excerpt",
                metadata={
                    "video_id": "abcdefghijk",
                    "start": 61.0,
                    "end": 70.0,
                },
            )
        ]

        context = build_context(docs)
        citations = build_citations(docs)

        self.assertIn("[SOURCE 1]", context)
        self.assertEqual(citations[0].source_number, 1)
        self.assertEqual(citations[0].timestamp, "01:01 - 01:10")


class RagServiceTests(unittest.TestCase):
    @patch("rag_service.run_ingestion")
    def test_ingestion_stores_vectors_for_the_requested_user_and_chat(self, run_ingestion):
        import rag_service

        response = rag_service.ingest(
            rag_service.IngestRequest(
                url="https://youtu.be/abcdefghijk",
                user_id=7,
                chat_id=12,
            )
        )

        self.assertEqual(response["video_id"], "abcdefghijk")
        run_ingestion.assert_called_once_with(
            "https://youtu.be/abcdefghijk",
            7,
            12,
        )

    @patch("rag_service.has_chat_vectors", return_value=True)
    @patch("rag_service.ask")
    def test_chat_passes_only_requested_chat_history_and_scope(
        self,
        ask,
        has_chat_vectors,
    ):
        import rag_service
        from generation import RAGResponse

        ask.return_value = RAGResponse(answer="Answer", citations=[])
        response = rag_service.chat(
            rag_service.ChatRequest(
                question="Question",
                user_id=7,
                chat_id=12,
                history=[
                    {"role": "user", "content": "Earlier question"},
                    {"role": "assistant", "content": "Earlier answer"},
                ],
            )
        )

        self.assertEqual(response.answer, "Answer")
        has_chat_vectors.assert_called_once_with(7, 12)
        ask.assert_called_once()
        self.assertEqual(ask.call_args.args[0], "Question")
        self.assertEqual(
            [type(message).__name__ for message in ask.call_args.args[1]],
            ["HumanMessage", "AIMessage"],
        )
        self.assertEqual(ask.call_args.kwargs["user_id"], 7)
        self.assertEqual(ask.call_args.kwargs["chat_id"], 12)


class IngestionPipelineTests(unittest.TestCase):
    @patch("indexing.index_pipeline.clear_chat_cache")
    @patch("indexing.index_pipeline.store_embeddings")
    @patch("indexing.index_pipeline.create_windowed_chunks")
    @patch("indexing.index_pipeline.ingest_youtube_transcript")
    def test_stores_and_registers_new_index(
        self,
        ingest_transcript,
        chunk_documents,
        store_embeddings,
        clear_chat_cache,
    ):
        transcript = [Document(page_content="transcript segment")]
        chunks = [Document(page_content="transcript chunk")]
        ingest_transcript.return_value = transcript
        chunk_documents.return_value = chunks
        index_pipeline.run_ingestion("https://youtu.be/abcdefghijk", 7, 12)
        chunk_documents.assert_called_once_with(transcript)
        store_embeddings.assert_called_once_with(chunks, 7, 12)
        clear_chat_cache.assert_called_once_with(7, 12)

    @patch("indexing.index_pipeline.store_embeddings")
    @patch("indexing.index_pipeline.create_windowed_chunks")
    @patch("indexing.index_pipeline.ingest_youtube_transcript")
    def test_rejects_transcripts_without_usable_chunks(
        self, ingest_transcript, chunk_documents, store_embeddings
    ):
        ingest_transcript.return_value = []
        chunk_documents.return_value = []

        with self.assertRaisesRegex(ValueError, "usable captions"):
            index_pipeline.run_ingestion("https://youtu.be/abcdefghijk", 7, 12)

        store_embeddings.assert_not_called()


if __name__ == "__main__":
    unittest.main()
