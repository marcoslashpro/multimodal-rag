import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from mm_rag.pipelines.pipes import Piper
from mm_rag.exceptions import ObjectUpsertionError
import mm_rag.datastructures as ds

class DummyFile:
    def __init__(self, file_id="id"):
        self.metadata = MagicMock()
        self.metadata.file_id = file_id

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

class DummyUploader:
    def __init__(self, *a, **kw):
        self.aupload_in_vector_store = AsyncMock()
        self.aupload_in_bucket = AsyncMock()
        self.aupload = AsyncMock()
        self.remove_object = AsyncMock()

class TestPipe(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.dynamo = DummyDynamoDB()
        self.bucket = DummyBucket()
        self.auth = "user1"
        self.piper = Piper(
            factory = MagicMock()
        )

    async def test_pipe_txt_success(self):
        path = "foo.txt"
        with patch.object(self.piper, '_get', return_value=(DummyUploader(), MagicMock())):
            await self.piper.pipe(path, self.auth)

    async def test_pipe_img_success(self):
        for ext in [".jpeg", ".jpg", ".png"]:
            path = "foo" + ext
            with patch.object(self.piper, '_get', return_value=(DummyUploader(), MagicMock())):
                await self.piper.pipe(path, self.auth)

    async def test_pipe_pdf_success(self):
        path = "foo.pdf"
        with patch.object(self.piper, '_get', return_value=(DummyUploader(), MagicMock())):
            await self.piper.pipe(path, self.auth)

    async def test_pipe_docx_success(self):
        path = "foo.docx"
        with patch.object(self.piper, '_get', return_value=(DummyUploader(), MagicMock())):
            await self.piper.pipe(path, self.auth)

    async def test_pipe_vectorstore_upload_error(self):
        path = "foo.txt"
        uploader = DummyUploader()
        uploader.aupload = AsyncMock(side_effect=ObjectUpsertionError(storage=ds.Storages.VECTORSTORE))
        with patch.object(self.piper, '_get', return_value=(uploader, MagicMock())):
            with self.assertRaises(ObjectUpsertionError):
                await self.piper.pipe(path, self.auth)

    async def test_pipe_bucket_upload_error(self):
        path = "foo.txt"
        uploader = DummyUploader()
        uploader.aupload = AsyncMock(side_effect=ObjectUpsertionError(storage=ds.Storages.BUCKET))
        with patch.object(self.piper, '_get', return_value=(uploader, MagicMock())):
            with self.assertRaises(ObjectUpsertionError):
                await self.piper.pipe(path, self.auth)


if __name__ == "__main__":
    unittest.main()