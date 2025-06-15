import unittest
from unittest.mock import AsyncMock, MagicMock, patch, call
import asyncio

import mm_rag.pipelines.datastructures as ds
from mm_rag.pipelines.pipes import pipe
from mm_rag.exceptions import ObjectUpsertionError

class DummyFile:
    def __init__(self, file_id="id"):
        self.metadata = MagicMock()
        self.metadata.file_id = file_id

class DummyUploader:
    def __init__(self, *a, **kw):
        self.aupload_in_vector_store = AsyncMock()
        self.aupload_in_bucket = AsyncMock()
        self.remove_object = MagicMock()

class DummyVectorStoreFactory:
    def get_vector_store(self, auth):
        return MagicMock()

class DummyBucket:
    def remove_object(self, file_id):
        pass

class DummyVectorStore:
    def remove_object(self, file_id):
        pass

class DummyDynamoDB:
    pass

class DummyExtractor:
    def extract(self, path, auth):
        return DummyFile(file_id="fileid")

class TestPipe(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dynamo = DummyDynamoDB()
        self.vectorstore_factory = DummyVectorStoreFactory()
        self.bucket = DummyBucket()
        self.auth = "user1"

    @patch("mm_rag.pipelines.pipes.TxtExtractor", DummyExtractor)
    @patch("mm_rag.pipelines.pipes.TxtUploader", DummyUploader)
    async def test_pipe_txt_success(self):
        path = "foo.txt"
        await pipe(path, self.auth, self.dynamo, self.vectorstore_factory, self.bucket)

    @patch("mm_rag.pipelines.pipes.ImgExtractor", DummyExtractor)
    @patch("mm_rag.pipelines.pipes.ImgUploader", DummyUploader)
    async def test_pipe_img_success(self):
        for ext in [".jpeg", ".jpg", ".png"]:
            path = "foo" + ext
            await pipe(path, self.auth, self.dynamo, self.vectorstore_factory, self.bucket)

    @patch("mm_rag.pipelines.pipes.PdfExtractor", DummyExtractor)
    @patch("mm_rag.pipelines.pipes.PdfUploader", DummyUploader)
    async def test_pipe_pdf_success(self):
        path = "foo.pdf"
        await pipe(path, self.auth, self.dynamo, self.vectorstore_factory, self.bucket)

    @patch("mm_rag.pipelines.pipes.DocExtractor", DummyExtractor)
    @patch("mm_rag.pipelines.pipes.PdfUploader", DummyUploader)
    async def test_pipe_docx_success(self):
        path = "foo.docx"
        await pipe(path, self.auth, self.dynamo, self.vectorstore_factory, self.bucket)

    @patch("mm_rag.pipelines.pipes.TxtExtractor", DummyExtractor)
    @patch("mm_rag.pipelines.pipes.TxtUploader", DummyUploader)
    async def test_pipe_vectorstore_upload_error(self):
        path = "foo.txt"
        # Simulate vectorstore upload error
        uploader = DummyUploader()
        uploader.aupload_in_vector_store = AsyncMock(side_effect=ObjectUpsertionError(storage="PineconeVectorStore"))
        uploader.aupload_in_bucket = AsyncMock()
        with patch("mm_rag.pipelines.pipes.TxtUploader", return_value=uploader), \
             patch("mm_rag.pipelines.pipes.TxtExtractor", DummyExtractor), \
             patch("mm_rag.pipelines.pipes.vectorstore.VectorStoreFactory.get_vector_store", return_value=DummyVectorStore()), \
             patch("mm_rag.pipelines.pipes.s3bucket", DummyBucket()):
            with self.assertRaises(ObjectUpsertionError):
                await pipe(path, self.auth, self.dynamo, self.vectorstore_factory, self.bucket)

    @patch("mm_rag.pipelines.pipes.TxtExtractor", DummyExtractor)
    @patch("mm_rag.pipelines.pipes.TxtUploader", DummyUploader)
    async def test_pipe_bucket_upload_error(self):
        path = "foo.txt"
        # Simulate bucket upload error
        uploader = DummyUploader()
        uploader.aupload_in_vector_store = AsyncMock()
        uploader.aupload_in_bucket = AsyncMock(side_effect=ObjectUpsertionError(storage="BucketService"))
        with patch("mm_rag.pipelines.pipes.TxtUploader", return_value=uploader), \
             patch("mm_rag.pipelines.pipes.TxtExtractor", DummyExtractor), \
             patch("mm_rag.pipelines.pipes.vectorstore.VectorStoreFactory.get_vector_store", return_value=DummyVectorStore()), \
             patch("mm_rag.pipelines.pipes.s3bucket", DummyBucket()):
            with self.assertRaises(ObjectUpsertionError):
                await pipe(path, self.auth, self.dynamo, self.vectorstore_factory, self.bucket)

    async def test_pipe_unsupported_file_type(self):
        path = "foo.unsupported"
        with patch("mm_rag.pipelines.pipes.ds.FileType", side_effect=ValueError("unsupported")):
            with self.assertRaises(ValueError):
                await pipe(path, self.auth, self.dynamo, self.vectorstore_factory, self.bucket)

    @patch("mm_rag.pipelines.pipes.TxtExtractor", DummyExtractor)
    @patch("mm_rag.pipelines.pipes.TxtUploader", DummyUploader)
    async def test_pipe_unexpected_exception(self):
        path = "foo.txt"
        uploader = DummyUploader()
        uploader.aupload_in_vector_store = AsyncMock(side_effect=ExceptionGroup("unexpected", [RuntimeError()]))
        uploader.aupload_in_bucket = AsyncMock()
        with patch("mm_rag.pipelines.pipes.TxtUploader", return_value=uploader), \
             patch("mm_rag.pipelines.pipes.TxtExtractor", DummyExtractor), \
             patch("mm_rag.pipelines.pipes.vectorstore.VectorStoreFactory.get_vector_store", return_value=DummyVectorStore()), \
             patch("mm_rag.pipelines.pipes.s3bucket", DummyBucket()):
            with self.assertRaises(ExceptionGroup):
                await pipe(path, self.auth, self.dynamo, self.vectorstore_factory, self.bucket)

if __name__ == "__main__":
    unittest.main()