from mm_rag.config.config import config
from mm_rag.cli_entrypoints.configure import write_env

import os
from pathlib import Path

from unittest.mock import MagicMock, patch
import unittest


class TestConfig(unittest.TestCase):
    @patch('mm_rag.cli_entrypoints.configure.open', create=True)
    def test_config_creation(self, mock_open):
        # Mock the file handle returned by open
        mock_file_handle = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_handle

        # Call the function to test
        write_env()

        # Assert that open was called with the correct path and mode
        expected_path  = Path(__file__).parent.parent / '.env'
        mock_open.assert_called_once_with(expected_path, 'w')

        # Optionally, check if the file handle was used to write content
        mock_file_handle.write.assert_called()  # Ensure something was written


if __name__ == "__main__":
    unittest.main()