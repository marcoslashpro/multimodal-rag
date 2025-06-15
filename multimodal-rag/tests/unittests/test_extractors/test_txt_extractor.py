import os
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from mm_rag.pipelines.extractors import (
    TxtExtractor
)
import mm_rag.pipelines.datastructures as ds

class DummyMetadata(ds.Metadata):
    def __init__(self):
        super().__init__(
            file_name="file",
            file_type=ds.FileType(".txt").value,
            author="user1",
            created="now",
        )

class TestTxtExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = TxtExtractor()
        self.metadata = DummyMetadata()

    def test_extract_metadata(self):
        meta = self.extractor._extract_metadata("foo.txt", "user1")
        self.assertEqual(meta.author, "user1")
        self.assertEqual(meta.file_name, "foo")

    def test_extract_content_success(self):
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            f.write("hello world")
            f.flush()
            content = self.extractor._extract_content(f.name)
            self.assertEqual(content, "hello world")
        os.unlink(f.name)

    def test_extract_docs(self):
        with patch("mm_rag.pipelines.utils._generate_ids", return_value=["id"]), \
             patch("mm_rag.pipelines.utils.generate_docs", return_value=[Document(page_content="abc")]):
            docs = self.extractor._extract_docs("abc", self.metadata)
            self.assertTrue(all(isinstance(doc, Document) for doc in docs))


if __name__ == "__main__":
    unittest.main()