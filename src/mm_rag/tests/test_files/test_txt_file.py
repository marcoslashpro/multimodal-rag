from mm_rag.processing.files import TxtFile
from mm_rag.processing.processors import TxtProcessor

import os

import unittest
from unittest.mock import MagicMock, patch, call


class TestTxtFile(unittest.TestCase):
  def setUp(self):
    self.mock_txt_file = 'test_file.txt'
    self.mock_processor = MagicMock(TxtProcessor)
    self.mock_owner = 'user123'

    self.file = TxtFile(
      self.mock_txt_file,
      self.mock_owner,
      self.mock_processor
    )
    self.expected_content = 'Test'
    with open(self.mock_txt_file, 'w') as f:
      f.write(self.expected_content)

    with open(self.mock_txt_file, 'r') as f:
      self.mock_processor.load_from_path.return_value = f.read()

  def tearDown(self) -> None:
    if os.path.exists(self.mock_txt_file):
      os.remove(self.mock_txt_file)

  def test_right_file_content(self):
    self.assertIsInstance(
      self.file.file_content, str
    )

    self.assertEqual(self.file.file_content, self.expected_content)


if __name__ == "__main__":
  unittest.main()