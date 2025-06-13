from mm_rag.processing.files import ImgFile
from mm_rag.processing.processors import ImgProcessor
from mm_rag.processing.handlers import ImgHandler

from PIL import Image
import os

import unittest
from unittest.mock import MagicMock, patch, call


class TestImgFile(unittest.TestCase):
  def setUp(self):
    self.mock_img_file = 'test_file.jpeg'
    Image.new('RGB', (10, 10), 'red').save(self.mock_img_file)
    self.mock_processor = ImgProcessor(MagicMock(), MagicMock(ImgHandler))
    self.mock_owner = 'user123'

    self.file = ImgFile(
      self.mock_img_file,
      self.mock_owner,
      self.mock_processor
    )

  def tearDown(self) -> None:
    if os.path.exists(self.mock_img_file):
      os.remove(self.mock_img_file)

  def test_right_file_content(self):
    self.assertIsInstance(
      self.file.file_content, Image.Image
    )

  def test_right_encodings(self):
    expected = 'encoded'
    self.mock_processor.handler.base64_encode.return_value = str(expected)

    self.assertEqual(
      self.file.encodings, expected
    )

    self.assertIsInstance(self.file.encodings, str)

  def test_right_id(self):
    self.assertEqual(
      self.file.file_id, self.file.metadata.fileId
    )


if __name__ == "__main__":
  unittest.main()