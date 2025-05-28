from mm_rag.processing.files import PdfFile
from mm_rag.processing.processors import PdfProcessor
from mm_rag.processing.handlers import ImgHandler

from PIL import Image
from pdf2image import convert_from_path

import os

import unittest
from unittest.mock import MagicMock, patch, call

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_test_pdf(filename: str):
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, "This is a test PDF file.")
    c.drawString(100, 735, "It was generated using Python.")
    c.save()


class TestPdfFile(unittest.TestCase):
  def setUp(self):
    self.mock_pdf_file = 'test_file.pdf'
    create_test_pdf(self.mock_pdf_file)
    self.mock_processor = MagicMock(PdfProcessor)
    self.mock_processor.handler = MagicMock(ImgHandler)
    self.mock_owner = 'user123'

    self.file = PdfFile(
      self.mock_pdf_file,
      self.mock_owner,
      self.mock_processor
    )

    self.mock_processor.load_from_path.return_value = convert_from_path(self.mock_pdf_file)

  def tearDown(self) -> None:
    if os.path.exists(self.mock_pdf_file):
      os.remove(self.mock_pdf_file)

  def test_right_file_content(self):
    self.assertIsInstance(
      self.file.file_content, list
    )

    for page in self.file.file_content:
      self.assertIsInstance(page, Image.Image)

  def test_right_encodings(self):
    expected = 'encoded'
    self.mock_processor.handler.base64_encode.return_value = str(expected)

    self.assertIsInstance(self.file.encodings, list)
    for encoded in self.file.encodings:
      self.assertIsInstance(encoded, str)
      self.assertEqual(
        encoded, expected
      )


if __name__ == "__main__":
  unittest.main()