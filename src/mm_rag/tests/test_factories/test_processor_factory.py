import os

from mm_rag.processing.processors import ProcessorFactory
from mm_rag.processing.processors import PdfProcessor, ImgProcessor, TxtProcessor
from mm_rag.agents.mm_embedder import Embedder
from mm_rag.processing.handlers import ImgHandler

import unittest
from unittest.mock import MagicMock, patch, call

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from PIL import Image


def create_test_pdf(filename: str):
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, "This is a test PDF file.")
    c.drawString(100, 735, "It was generated using Python.")
    c.save()


class TestUploaderFactory(unittest.TestCase):
  def setUp(self) -> None:
    self.factory = ProcessorFactory()
    self.mock_embedder = MagicMock(Embedder)
    self.mock_handler = MagicMock(ImgHandler)

    self.mock_txt_file = 'test_file.txt'
    with open(self.mock_txt_file, 'w') as f:
      f.write('This is a test')

    self.mock_pdf_file = 'test_file.pdf'
    create_test_pdf(self.mock_pdf_file)

    self.mock_img_file = 'test.jpeg'
    Image.new('RGB', (10, 10), 'red').save(self.mock_img_file)

  def tearDown(self) -> None:
    if self.mock_txt_file:
      os.remove(self.mock_txt_file)

    if self.mock_img_file:
      os.remove(self.mock_img_file)

    if self.mock_pdf_file:
      os.remove(self.mock_pdf_file)

  def test_create_right_uploader(self):
    expected_img_uploader = self.factory.get_processor(
        self.mock_img_file,
        self.mock_embedder,
        self.mock_handler
      )

    self.assertIsInstance(
      expected_img_uploader,
      ImgProcessor
    )

    expected_pdf_uploader = self.factory.get_processor(
        self.mock_pdf_file,
        self.mock_embedder,
        self.mock_handler
      )

    self.assertIsInstance(
      expected_pdf_uploader,
      PdfProcessor
    )

    expected_txt_uploader = self.factory.get_processor(
        self.mock_txt_file,
        self.mock_embedder,
        self.mock_handler
      )

    self.assertIsInstance(
      expected_txt_uploader,
      TxtProcessor
    )


if __name__ == "__main__":
  unittest.main()