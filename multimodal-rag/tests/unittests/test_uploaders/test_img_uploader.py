import unittest
from unittest.mock import MagicMock
from PIL import Image
from mm_rag.pipelines.uploaders import ImgUploader
from mm_rag.processing.files import ImgFile
from langchain_core.documents import Document
from mm_rag.processing.handlers import ImgHandler


class TestImgUploader(unittest.TestCase):
    def setUp(self):
        self.uploader = ImgUploader(
            dynamodb=MagicMock(),
            vector_store=MagicMock(),
            bucket=MagicMock(),
            handler=ImgHandler()
        )
        self.mock_file = ImgFile(file_path="test.jpg", owner="user1", processor=MagicMock())
        self.mock_file._file_content = Image.new("RGB", (100, 100))
        self.mock_file._encodings = "encoded_image"

    def test_upload_in_vector_store(self):
        docs = [Document(page_content="Image Chunk", id="123chunk/1")]
        result = self.uploader.upload_in_vector_store(self.mock_file, docs)
        self.assertTrue(result)
        self.uploader.vector_store.add_image.assert_called_once_with(
            self.mock_file.encodings, self.mock_file.metadata, self.mock_file.file_id
        )

    def test_upload_in_bucket(self):
        self.uploader.handler = MagicMock()
        self.uploader.handler.save_img_to_buffer.return_value = MagicMock()
        self.uploader.upload_in_bucket(self.mock_file)
        self.uploader.handler.save_img_to_buffer.assert_called_once_with(self.mock_file.file_content)
        self.uploader.bucket.upload_object_from_file.assert_called_once_with(
            self.uploader.handler.save_img_to_buffer.return_value, self.mock_file.file_id
        )

    def test_upload_in_dynamo(self):
        self.uploader.upload_in_dynamo(self.mock_file)
        self.uploader.ddb.store_file.assert_called_once_with(
            "files", self.mock_file.file_id, self.mock_file.owner, self.mock_file.metadata.__dict__
        )


if __name__ == "__main__":
    unittest.main()