import unittest
from unittest.mock import MagicMock, patch
from mm_rag.processing.processors import TxtProcessor
from mm_rag.processing.files import TxtFile
from langchain_core.documents import Document


class TestTxtProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = TxtProcessor(MagicMock())

    @patch("mm_rag.processing.processors.RecursiveCharacterTextSplitter")
    def test_process_valid_txt_file(self, mock_splitter):
        mock_splitter.return_value.split_text.return_value = ["Sample", "text"]
        self.processor.generate_ids = MagicMock(return_value=["id1", "id2"])

        mock_file = TxtFile(
            file_path="test.txt",
            owner="user1",
            processor=self.processor
        )
        mock_file._file_content = "Sample text"  # Simulate file content

        docs = self.processor.process(mock_file)

        self.assertEqual(len(docs), 2)
        self.assertIsInstance(docs[0], Document)
        self.assertEqual(docs[0].page_content, "Sample")
        self.assertEqual(docs[1].page_content, "text")

    def test_process_invalid_file_type(self):
        with self.assertRaises(ValueError):
            self.processor.process("InvalidFileType")  # type: ignore[Invalid type]

    def test_load_from_path_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.processor.load_from_path("non_existent_file.txt")

    def test_generate_ids(self):
        ids = self.processor.generate_ids("123", 3)
        self.assertEqual(len(ids), 3)
        self.assertTrue(all(id.startswith("123/chunk") for id in ids))

    def test_file_content_validation(self):
        mock_file = TxtFile(
            file_path="test.txt",
            owner="user1",
            processor=self.processor
        )
        mock_file._file_content = 123  # type: ignore[Invalid type]
        with self.assertRaises(ValueError):
            _ = mock_file.file_content

if __name__ == "__main__":
    unittest.main()