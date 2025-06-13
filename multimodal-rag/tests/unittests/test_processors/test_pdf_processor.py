import os
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
from mm_rag.processing.processors import PdfProcessor
from mm_rag.processing.files import PdfFile
from mm_rag.processing.handlers import ImgHandler
from langchain_core.documents import Document

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_test_pdf(filename: str):
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, "This is a test PDF file.")
    c.drawString(100, 735, "It was generated using Python.")
    c.save()


class TestPdfProcessor(unittest.TestCase):
    def setUp(self):
        self.mock_pdf_file = 'test_file.pdf'
        create_test_pdf(self.mock_pdf_file)
        self.handler = MagicMock(spec=ImgHandler)
        self.processor = PdfProcessor(embedder=MagicMock(), handler=self.handler)

        self.mock_file = PdfFile(
            file_path=self.mock_pdf_file,
            owner="user1",
            processor=self.processor
        )

    def tearDown(self):
        if os.path.exists(self.mock_pdf_file):
            os.remove(self.mock_pdf_file)

    def test_process_valid_pdf_file(self):
        mock_file = PdfFile(
            file_path=self.mock_pdf_file,
            owner="user1",
            processor=self.processor
        )
        mock_file._file_content = [Image.new("RGB", (100, 100))]  # Simulate file content

        self.handler.adjust_orientation.return_value = mock_file.file_content[0]
        self.handler.adjust_shape.return_value = mock_file.file_content[0]
        self.handler.base64_encode.return_value = "encoded_page"

        docs = self.processor.process(mock_file)

        self.assertEqual(len(docs), 1)
        self.assertIsInstance(docs[0], Document)
        self.assertEqual(docs[0].page_content, "encoded_page")

    def test_process_invalid_file_type(self):
        with self.assertRaises(ValueError):
            self.processor.process("InvalidFileType")  # type: ignore[Invalid type]

    def test_load_from_path_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.processor.load_from_path("non_existent_file.pdf")

    def test_generate_ids(self):
        ids = self.processor.generate_ids("123", 3)
        self.assertEqual(len(ids), 3)
        self.assertTrue(all(id.startswith("123/chunk") for id in ids))

    def test_file_content_validation(self):
        self.mock_file._file_content = "invalid_content"  # Invalid type
        with self.assertRaises(ValueError):
            _ = self.mock_file.file_content

    def test_all_doc_id_equal_file_id(self):
        self.handler.base64_encode.return_value = "encoded_image"

        docs = self.processor.process(self.mock_file)
        for doc in docs:
            self.assertTrue(doc.id.startswith(self.mock_file.file_id))

    def test_file_id_equals_metadata_id(self):
        self.assertEqual(self.mock_file.file_id, self.mock_file.metadata.fileId)


if __name__ == "__main__":
    unittest.main()