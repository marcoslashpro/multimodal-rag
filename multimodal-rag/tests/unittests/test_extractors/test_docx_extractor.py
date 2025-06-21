import unittest
from unittest.mock import patch, MagicMock

from PIL import Image
from langchain_core.documents import Document

from mm_rag.pipelines.extractors import (
    DocExtractor,
    FileNotValidError, ImageTooBigError
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

class TestDocExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = DocExtractor()
        self.metadata = DummyMetadata()

    @patch("mm_rag.pipelines.extractors.subprocess.run")
    @patch("mm_rag.pipelines.extractors.convert_from_path")
    def test_extract_content_success(self, mock_convert, mock_run):
        mock_run.return_value.returncode = 0
        img = Image.new("RGB", (10, 10))
        mock_convert.return_value = [img]
        with patch("os.path.splitext", return_value=("file", ".docx")):
            pages = self.extractor._extract_content("file.docx")
            self.assertEqual(len(pages), 1)
            self.assertIsInstance(pages[0], Image.Image)

    @patch("mm_rag.pipelines.extractors.subprocess.run")
    def test_extract_content_pandoc_fail(self, mock_run):
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = b"error"
        mock_run.return_value.stdout = b"stdout"
        with patch("os.path.splitext", return_value=("file", ".docx")):
            with self.assertRaises(FileNotValidError):
                self.extractor._extract_content("file.docx")

    @patch("mm_rag.pipelines.extractors.subprocess.run")
    @patch("mm_rag.pipelines.extractors.convert_from_path")
    def test_extract_content_decompression_bomb(self, mock_convert, mock_run):
        mock_run.return_value.returncode = 0
        bomb_img = MagicMock()
        bomb_img.convert.side_effect = Image.DecompressionBombError("bomb")
        mock_convert.return_value = [bomb_img]
        with patch("os.path.splitext", return_value=("file", ".docx")):
            with self.assertRaises(ImageTooBigError):
                self.extractor._extract_content("file.docx")

    def test_extract_docs(self):
        img = Image.new("RGB", (10, 10))
        with patch("mm_rag.pipelines.utils.process_img", return_value="imgdata"), \
             patch("mm_rag.pipelines.utils.generate_ids", return_value=["id"]), \
             patch("mm_rag.pipelines.utils.generate_docs", return_value=[Document(page_content="abc")]):
            docs = self.extractor._extract_docs([img], self.metadata)
            self.assertTrue(all(isinstance(doc, Document) for doc in docs))


if __name__ == "__main__":
    unittest.main()