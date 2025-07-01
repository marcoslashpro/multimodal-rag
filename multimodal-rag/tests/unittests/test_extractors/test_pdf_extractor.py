import unittest
from unittest.mock import patch, MagicMock

from PIL import Image
from langchain_core.documents import Document

from mm_rag.pipelines.extractors import (
    PdfExtractor,
    ImageTooBigError
)
import mm_rag.datastructures as ds

class DummyMetadata(ds.Metadata):
    def __init__(self):
        super().__init__(
            file_name="file",
            file_type=ds.FileType(".txt").value,
            author="user1",
            created="now",
        )

class TestPdfExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = PdfExtractor(MagicMock())
        self.metadata = DummyMetadata()

    @patch("mm_rag.pipelines.extractors.convert_from_path")
    def test_extract_content_success(self, mock_convert):
        img = Image.new("RGB", (10, 10))
        mock_convert.return_value = [img]
        pages = self.extractor._extract_content("file.pdf")
        self.assertEqual(len(pages), 1)
        self.assertIsInstance(pages[0], Image.Image)

    @patch("mm_rag.pipelines.extractors.convert_from_path")
    def test_extract_content_decompression_bomb(self, mock_convert):
        bomb_img = MagicMock()
        bomb_img.convert.side_effect = Image.DecompressionBombError("bomb")
        mock_convert.return_value = [bomb_img]
        with self.assertRaises(ImageTooBigError):
            self.extractor._extract_content("file.pdf")

    def test_extract_docs(self):
        img = Image.new("RGB", (10, 10))
        with patch("mm_rag.pipelines.utils.process_img", return_value="imgdata"), \
             patch("mm_rag.pipelines.utils.generate_ids", return_value=["id"]), \
             patch("mm_rag.pipelines.utils.generate_docs", return_value=[Document(page_content="abc")]):
            docs = self.extractor._extract_docs([img], self.metadata)
            self.assertTrue(all(isinstance(doc, Document) for doc in docs))


if __name__ == "__main__":
    unittest.main()