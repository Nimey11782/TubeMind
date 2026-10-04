import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings
from youtube_transcript_api import TranscriptsDisabled

import indexing.embedding as embedding
import indexing.index_pipeline as index_pipeline
import indexing.document_ingestion as document_ingestion
import retriever
import database.db as database
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

    @patch("indexing.embedding.get_embeddings")
    @patch("indexing.embedding.FAISS")
    def test_persists_non_empty_index(self, faiss_class, get_embeddings):
        vectorstore = faiss_class.from_documents.return_value
        docs = [Document(page_content="transcript chunk")]

        with tempfile.TemporaryDirectory() as directory:
            result = embedding.store_embeddings(docs, directory)

        self.assertIs(result, vectorstore)
        kwargs = faiss_class.from_documents.call_args.kwargs
        self.assertEqual(kwargs["documents"], docs)
        self.assertIs(kwargs["embedding"], get_embeddings.return_value)
        self.assertEqual(len(kwargs["ids"]), 1)
        vectorstore.save_local.assert_called_once_with(
            str(Path(directory).resolve())
        )

    @patch("indexing.embedding.get_embeddings")
    @patch("indexing.embedding.FAISS")
    def test_updates_same_video_without_removing_other_videos(
        self, faiss_class, get_embeddings
    ):
        existing_store = faiss_class.load_local.return_value
        existing_store.index_to_docstore_id = {
            0: "another-video",
            1: "previous-video",
        }
        existing_store.docstore.search.side_effect = [
            Document(
                page_content="other",
                metadata={"video_id": "lmnopqrstuv"},
            ),
            Document(
                page_content="old",
                metadata={"video_id": "abcdefghijk"},
            ),
        ]
        replacement = Document(
            page_content="new",
            metadata={
                "video_id": "abcdefghijk",
                "chunk_id": "abcdefghijk:0:1:0",
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "index.faiss").touch()
            result = embedding.store_embeddings([replacement], directory)

        self.assertIs(result, existing_store)
        existing_store.delete.assert_called_once_with(ids=["previous-video"])
        existing_store.add_documents.assert_called_once_with(
            [replacement],
            ids=["abcdefghijk:0:1:0"],
        )
        existing_store.save_local.assert_called_once()

    def test_user_indexes_are_isolated(self):
        first_user_path = embedding.get_user_index_path(1)
        second_user_path = embedding.get_user_index_path(2)

        self.assertNotEqual(first_user_path, second_user_path)
        self.assertEqual(
            first_user_path.parent,
            embedding.DEFAULT_INDEX_PATH,
        )
        with self.assertRaises(ValueError):
            embedding.get_user_index_path(0)

    def test_real_faiss_reingestion_replaces_only_the_same_video(self):
        first_video = Document(
            page_content="Old transcript for the first video.",
            metadata={
                "video_id": "abcdefghijk",
                "chunk_id": "abcdefghijk:0:1:0",
            },
        )
        another_video = Document(
            page_content="Transcript for an unrelated video.",
            metadata={
                "video_id": "lmnopqrstuv",
                "chunk_id": "lmnopqrstuv:0:1:0",
            },
        )
        replacement_video = Document(
            page_content="Updated transcript for the first video.",
            metadata={
                "video_id": "abcdefghijk",
                "chunk_id": "abcdefghijk:0:1:0",
            },
        )
        embeddings = FakeEmbeddings(size=8)

        with tempfile.TemporaryDirectory() as directory, patch.object(
            embedding,
            "get_embeddings",
            return_value=embeddings,
        ):
            embedding.store_embeddings(
                [first_video, another_video],
                directory,
            )
            vectorstore = embedding.store_embeddings(
                [replacement_video],
                directory,
            )

        stored_documents = [
            vectorstore.docstore.search(docstore_id)
            for docstore_id in vectorstore.index_to_docstore_id.values()
        ]
        self.assertEqual(vectorstore.index.ntotal, 2)
        self.assertEqual(
            {doc.metadata["video_id"] for doc in stored_documents},
            {"abcdefghijk", "lmnopqrstuv"},
        )
        self.assertIn(
            "Updated transcript",
            next(
                doc.page_content
                for doc in stored_documents
                if doc.metadata["video_id"] == "abcdefghijk"
            ),
        )

    def test_rejects_empty_index(self):
        with self.assertRaisesRegex(ValueError, "without transcript chunks"):
            embedding.store_embeddings([])


class RetrievalTests(unittest.TestCase):
    @patch("retriever.rerank_documents")
    @patch("retriever.bm25_search")
    @patch("retriever.get_vectorstore")
    def test_default_retrieval_fuses_dense_and_bm25_results(
        self, get_vectorstore, bm25_search, rerank_documents
    ):
        vectorstore = get_vectorstore.return_value
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
        vectorstore.similarity_search.return_value = [first, shared]
        bm25_search.return_value = [shared, sparse]

        result = retriever.retrieve_chunks("question")

        get_vectorstore.assert_called_once_with(embedding.DEFAULT_INDEX_PATH)
        vectorstore.similarity_search.assert_called_once_with(
            "question",
            k=20,
        )
        bm25_search.assert_called_once_with(
            "question",
            vectorstore,
            index_path=embedding.DEFAULT_INDEX_PATH,
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
    @patch("retriever.get_vectorstore")
    def test_rerank_mode_is_explicit(self, get_vectorstore, rerank_documents):
        vectorstore = get_vectorstore.return_value
        documents = [Document(page_content="candidate chunk")]
        vectorstore.max_marginal_relevance_search.return_value = documents

        result = retriever.retrieve_chunks("question", mode="rerank")

        rerank_documents.assert_called_once_with("question", documents, top_n=4)
        self.assertEqual(result, rerank_documents.return_value)
        vectorstore.max_marginal_relevance_search.assert_called_once_with(
            "question",
            k=10,
            fetch_k=20,
            lambda_mult=0.6,
        )

    @patch("retriever.get_vectorstore")
    def test_rejects_invalid_mode_before_loading_index(self, get_vectorstore):
        with self.assertRaisesRegex(ValueError, "Unknown retrieval mode"):
            retriever.retrieve_chunks("question", mode="unknown")

        get_vectorstore.assert_not_called()

    def test_ingestion_refreshes_the_cached_vectorstore(self):
        old_vectorstore = Mock()
        new_vectorstore = Mock()
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory)

            with patch.object(retriever, "_vectorstores", {}), patch.object(
                retriever, "_bm25_indexes", {}
            ):
                retriever.set_vectorstore(old_vectorstore, index_path)
                retriever._bm25_indexes[index_path.resolve()] = (
                    id(old_vectorstore),
                    Mock(),
                    [],
                )
                retriever.set_vectorstore(new_vectorstore, index_path)
                self.assertIs(retriever.get_vectorstore(index_path), new_vectorstore)
                self.assertNotIn(index_path.resolve(), retriever._bm25_indexes)

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

    def test_bm25_retrieves_exact_keyword_matches_and_caches_index(self):
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
        documents_by_id = {
            "relevant": relevant,
            "unrelated": unrelated,
            "third": third,
        }
        vectorstore = SimpleNamespace(
            index_to_docstore_id={
                0: "relevant",
                1: "unrelated",
                2: "third",
            },
            docstore=SimpleNamespace(search=documents_by_id.__getitem__),
        )

        with tempfile.TemporaryDirectory() as directory, patch.object(
            retriever, "_bm25_indexes", {}
        ):
            first_result = retriever.bm25_search(
                "nuclear fusion",
                vectorstore,
                index_path=directory,
            )
            cached_index = retriever._bm25_indexes[Path(directory).resolve()]
            second_result = retriever.bm25_search(
                "nuclear fusion",
                vectorstore,
                index_path=directory,
            )
            self.assertIs(
                retriever._bm25_indexes[Path(directory).resolve()],
                cached_index,
            )

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


class ApiPipelineTests(unittest.TestCase):
    @patch("main.save_message")
    @patch("main.ask")
    @patch("main.load_chat_history", return_value=[])
    @patch("main.get_user_index_path", return_value=Path("user-index"))
    def test_chat_uses_authenticated_users_index(
        self,
        get_user_index_path,
        load_chat_history,
        ask,
        save_message,
    ):
        import main
        from generation import RAGResponse

        ask.return_value = RAGResponse(answer="Answer", citations=[])
        response = main.chat(
            main.QueryRequest(question="Question"),
            current_user={"user_id": 7},
        )

        self.assertEqual(response.answer, "Answer")
        get_user_index_path.assert_called_once_with(7)
        ask.assert_called_once_with(
            "Question",
            [],
            retrieval_mode="hybrid",
            index_path=Path("user-index"),
        )
        self.assertEqual(save_message.call_count, 2)

    @patch("main.run_in_threadpool", new_callable=Mock)
    @patch("main.save_message")
    @patch("main.load_chat_history", return_value=[])
    @patch("main.get_user_index_path")
    @patch("main.stream_answer")
    def test_streaming_chat_sends_events_and_saves_completed_turn(
        self,
        stream_answer,
        get_user_index_path,
        load_chat_history,
        save_message,
        run_in_threadpool,
    ):
        import main

        async def run_in_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        async def fake_stream_answer(*args, **kwargs):
            yield {"type": "token", "content": "A streamed"}
            yield {"type": "token", "content": " answer."}
            yield {"type": "done", "citations": []}

        run_in_threadpool.side_effect = run_in_thread
        stream_answer.side_effect = fake_stream_answer

        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory)
            (index_path / "index.faiss").touch()
            get_user_index_path.return_value = index_path

            response = main.chat_stream(
                main.QueryRequest(question="Question"),
                current_user={"user_id": 7},
            )
            body = asyncio.run(self._read_stream(response))

        self.assertEqual(response.media_type, "text/event-stream")
        self.assertIn(b'"content": "A streamed"', body)
        self.assertIn(b'"type": "done"', body)
        self.assertEqual(save_message.call_count, 2)

    @staticmethod
    async def _read_stream(response):
        chunks = [chunk async for chunk in response.body_iterator]
        return "".join(
            chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            for chunk in chunks
        ).encode("utf-8")

    @patch("main.run_ingestion")
    @patch("main.get_user_index_path", return_value=Path("user-index"))
    def test_ingestion_uses_authenticated_users_index(
        self,
        get_user_index_path,
        run_ingestion,
    ):
        import main

        response = main.ingest(
            main.IngestRequest(url="https://youtu.be/abcdefghijk"),
            current_user={"user_id": 7},
        )

        self.assertEqual(response, {"status": "ok"})
        get_user_index_path.assert_called_once_with(7)
        run_ingestion.assert_called_once_with(
            "https://youtu.be/abcdefghijk",
            index_path=Path("user-index"),
        )


class IngestionPipelineTests(unittest.TestCase):
    @patch("indexing.index_pipeline.set_vectorstore")
    @patch("indexing.index_pipeline.store_embeddings")
    @patch("indexing.index_pipeline.create_windowed_chunks")
    @patch("indexing.index_pipeline.ingest_youtube_transcript")
    def test_stores_and_registers_new_index(
        self,
        ingest_transcript,
        chunk_documents,
        store_embeddings,
        set_vectorstore,
    ):
        transcript = [Document(page_content="transcript segment")]
        chunks = [Document(page_content="transcript chunk")]
        vectorstore = Mock()
        ingest_transcript.return_value = transcript
        chunk_documents.return_value = chunks
        store_embeddings.return_value = vectorstore

        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory)
            result = index_pipeline.run_ingestion(
                "https://youtu.be/abcdefghijk",
                index_path=index_path,
            )

        self.assertIs(result, vectorstore)
        chunk_documents.assert_called_once_with(transcript)
        store_embeddings.assert_called_once_with(
            chunks, index_path
        )
        set_vectorstore.assert_called_once_with(
            vectorstore, index_path
        )

    @patch("indexing.index_pipeline.store_embeddings")
    @patch("indexing.index_pipeline.create_windowed_chunks")
    @patch("indexing.index_pipeline.ingest_youtube_transcript")
    def test_rejects_transcripts_without_usable_chunks(
        self, ingest_transcript, chunk_documents, store_embeddings
    ):
        ingest_transcript.return_value = []
        chunk_documents.return_value = []

        with self.assertRaisesRegex(ValueError, "usable captions"):
            index_pipeline.run_ingestion("https://youtu.be/abcdefghijk")

        store_embeddings.assert_not_called()


class DatabaseTests(unittest.TestCase):
    def test_creates_missing_database_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "nested" / "chatbot.db"
            with patch.object(database, "DB_NAME", str(database_path)):
                connection = database.get_connection()
                try:
                    connection.execute("SELECT 1")
                finally:
                    connection.close()

            self.assertTrue(database_path.is_file())


if __name__ == "__main__":
    unittest.main()
