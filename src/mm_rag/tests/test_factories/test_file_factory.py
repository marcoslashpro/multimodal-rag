import os

from mm_rag.processing.files import FileFactory
from mm_rag.processing.processors import PdfProcessor, ImgProcessor, TxtProcessor
from mm_rag.processing.files import TxtFile, PdfFile, ImgFile

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
    self.factory = FileFactory()
    self.mock_owner = 'user123'
    self.mock_txt_processor = MagicMock(TxtProcessor)
    self.mock_img_processor = MagicMock(ImgProcessor)
    self.mock_pdf_processor = MagicMock(PdfProcessor)

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
    expected_img_uploader = self.factory.get_file(
        self.mock_img_file,
        self.mock_owner,
        self.mock_img_processor
      )

    self.assertIsInstance(
      expected_img_uploader,
      ImgFile
    )

    expected_pdf_uploader = self.factory.get_file(
        self.mock_pdf_file,
        self.mock_owner,
        self.mock_pdf_processor
      )

    self.assertIsInstance(
      expected_pdf_uploader,
      PdfFile
    )

    expected_txt_uploader = self.factory.get_file(
        self.mock_txt_file,
        self.mock_owner,
        self.mock_txt_processor
      )

    self.assertIsInstance(
      expected_txt_uploader,
      TxtFile
    )


if __name__ == "__main__":
  unittest.main()