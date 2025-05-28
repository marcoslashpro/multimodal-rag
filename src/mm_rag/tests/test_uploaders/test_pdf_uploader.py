import unittest
from unittest.mock import MagicMock
from PIL import Image
from mm_rag.pipelines.uploaders import PdfUploader
from mm_rag.processing.files import PdfFile
from langchain_core.documents import Document
from mm_rag.processing.handlers import ImgHandler


class TestPdfUploader(unittest.TestCase):
    def setUp(self):
        self.uploader = PdfUploader(
            dynamodb=MagicMock(),
            vector_store=MagicMock(),
            bucket=MagicMock(),
            handler=ImgHandler()
        )
        self.mock_file = PdfFile(file_path="test.pdf", owner="user1", processor=MagicMock())
        self.mock_file._file_content = [Image.new("RGB", (100, 100)), Image.new("RGB", (200, 200))]
        self.mock_file._encodings = ["encoded_page1", "encoded_page2"]

    def test_upload_in_vector_store(self):
        docs = [
            Document(page_content="Page 1", id="123/chunk1"),
            Document(page_content="Page 2", id="123/chunk2")
        ]
        result = self.uploader.upload_in_vector_store(self.mock_file, docs)
        self.assertTrue(result)
        self.uploader.vector_store.add_image.assert_any_call(
            self.mock_file.encodings[0], self.mock_file.metadata, docs[0].id
        )
        self.uploader.vector_store.add_image.assert_any_call(
            self.mock_file.encodings[1], self.mock_file.metadata, docs[1].id
        )

    def test_upload_in_bucket(self):
        self.uploader.handler = MagicMock()
        buffer1 = MagicMock()
        buffer2 = MagicMock()
        self.uploader.handler.save_img_to_buffer.side_effect = [buffer1, buffer2]
        docs = [
            Document(page_content="Page 1", id="123/chunk1"),
            Document(page_content="Page 2", id="123/chunk2")
        ]
        self.uploader.upload_in_bucket(self.mock_file, docs)
        self.uploader.handler.save_img_to_buffer.assert_any_call(self.mock_file.file_content[0])
        self.uploader.handler.save_img_to_buffer.assert_any_call(self.mock_file.file_content[1])
        self.uploader.bucket.upload_object_from_file.assert_any_call(buffer1, docs[0].id)
        self.uploader.bucket.upload_object_from_file.assert_any_call(buffer2, docs[1].id)

    def test_upload_in_dynamo(self):
        self.uploader.upload_in_dynamo(self.mock_file)
        self.uploader.ddb.store_file.assert_called_once_with(
            "files", self.mock_file.file_id, self.mock_file.owner, self.mock_file.metadata.__dict__
        )


if __name__ == "__main__":
    unittest.main()