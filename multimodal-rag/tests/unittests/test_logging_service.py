from mm_rag.logging_service.log_config import create_logger

import unittest
from unittest.mock import MagicMock, patch

import os
from pathlib import Path


class TestLoggingService(unittest.TestCase):
  def setUp(self):
    self.expected_path = Path(__file__).resolve().parent.parent.parent / f'src/mm_rag/logging_service/logs/{__name__}.log'
    self.logger = create_logger(__name__)

  def tearDown(self) -> None:
    if os.path.exists(self.expected_path):
      os.remove(self.expected_path)

  def test_log_file_creation(self):
    self.logger.info('This is a test')

    # Flush all handlers to ensure log file is written
    for handler in self.logger.handlers:
      handler.flush()

    self.assertTrue(os.path.exists(self.expected_path))



if __name__ == "__main__":
  unittest.main()