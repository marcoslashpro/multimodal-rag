import unittest
from unittest.mock import MagicMock
from PIL import Image
from mm_rag.processing.processors import ImgProcessor
from mm_rag.processing.files import ImgFile
from mm_rag.processing.handlers import ImgHandler
from langchain_core.documents import Document

from uuid import UUID

class TestImgProcessor(unittest.TestCase):
    def setUp(self):
        self.handler = MagicMock(spec=ImgHandler)
        self.processor = ImgProcessor(embedder=MagicMock(), handler=self.handler)

    def test_process_valid_img_file(self):
        mock_file = ImgFile(
            file_path="test.jpg",
            owner="user1",
            processor=self.processor
        )
        mock_file._file_content = Image.new("RGB", (100, 100))  # Simulate file content

        self.handler.adjust_orientation.return_value = mock_file.file_content
        self.handler.adjust_shape.return_value = mock_file.file_content
        self.handler.base64_encode.return_value = "encoded_image"

        docs = self.processor.process(mock_file)

        self.assertEqual(len(docs), 1)
        self.assertIsInstance(docs[0], Document)
        self.assertEqual(docs[0].page_content, "encoded_image")

    def test_process_invalid_file_type(self):
        with self.assertRaises(ValueError):
            self.processor.process("InvalidFileType")  # type: ignore[Invalid type]

    def test_load_from_path_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            self.processor.load_from_path("non_existent_file.jpg")

    def test_generate_ids(self):
        ids = self.processor.generate_ids("123", 2)
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(id.startswith("123/chunk") for id in ids))

    def test_file_content_validation(self):
        mock_file = ImgFile(
            file_path="test.jpg",
            owner="user1",
            processor=self.processor
        )
        mock_file._file_content = "invalid_content"  # Invalid type
        with self.assertRaises(ValueError):
            _ = mock_file.file_content

    def test_generate_id(self):
        mock_file = ImgFile(
            file_path="test.jpg",
            owner="user1",
            processor=self.processor
        )
        self.assertTrue(
            mock_file.file_id.startswith('user1/jpg/test', 0, len(mock_file.file_id)),
            f'Expected \'user1/jpg/test\', got {mock_file.file_id}'
        )

if __name__ == "__main__":
    unittest.main()