import os
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from langchain_core.documents import Document

from mm_rag.pipelines.extractors import (
    ImgExtractor,
    ImageTooBigError
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

class TestImgExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = ImgExtractor()
        self.metadata = DummyMetadata()

    def test_extract_content_success(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = Image.new("RGB", (10, 10))
            img.save(f.name)
            img2 = self.extractor._extract_content(f.name)
            self.assertIsInstance(img2, Image.Image)
        os.unlink(f.name)

    @patch("PIL.Image.open", side_effect=Image.DecompressionBombError("bomb"))
    def test_extract_content_decompression_bomb(self, mock_open):
        with self.assertRaises(ImageTooBigError):
            self.extractor._extract_content("fake.png")

    def test_extract_docs(self):
        img = Image.new("RGB", (10, 10))
        with patch("mm_rag.pipelines.utils.process_img", return_value="imgdata"):
            docs = self.extractor._extract_docs(img, self.metadata)
            self.assertTrue(all(isinstance(doc, Document) for doc in docs))


if __name__ == "__main__":
    unittest.main()