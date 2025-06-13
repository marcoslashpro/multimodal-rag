from mm_rag.processing.files import TxtFile
from mm_rag.processing.processors import TxtProcessor

import os

import unittest
from unittest.mock import MagicMock, patch, call


class TestTxtFile(unittest.TestCase):
  def setUp(self):
    self.mock_txt_file = 'test_file.txt'
    self.mock_processor = TxtProcessor(MagicMock())
    self.mock_owner = 'user123'

    self.file = TxtFile(
      self.mock_txt_file,
      self.mock_owner,
      self.mock_processor
    )
    self.expected_content = 'Test'
    with open(self.mock_txt_file, 'w') as f:
      f.write(self.expected_content)


  def tearDown(self) -> None:
    if os.path.exists(self.mock_txt_file):
      os.remove(self.mock_txt_file)

  def test_right_file_content(self):
    self.assertIsInstance(
      self.file.file_content, str
    )

    self.assertEqual(self.file.file_content, self.expected_content)

  def test_right_file_id(self):
    self.assertEqual(
      self.file.file_id, self.file.metadata.fileId
    )


if __name__ == "__main__":
  unittest.main()