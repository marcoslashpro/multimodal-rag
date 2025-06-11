import unittest
from unittest.mock import MagicMock
from mm_rag.pipelines.uploaders import TxtUploader
from mm_rag.processing.files import TxtFile
from mm_rag.models import dynamodb, vectorstore, s3bucket
from langchain_core.documents import Document



class TestTxtUploader(unittest.TestCase):
    def setUp(self):
        self.uploader = TxtUploader(
            dynamodb=MagicMock(dynamodb.DynamoDB),
            vector_store=MagicMock(vectorstore.PineconeVectorStore),
            bucket=MagicMock(s3bucket.BucketService)
        )
        self.mock_file = TxtFile(file_path="test.txt", owner="user1", processor=MagicMock())

    def test_upload_in_vector_store(self):
        self.mock_file._file_content = "Sample text content"
        docs = [Document(page_content="Chunk 1", id="123chunk/1")]
        result = self.uploader.upload_in_vector_store(self.mock_file, docs)
        self.assertTrue(result)
        self.uploader.vector_store.vector_store.add_documents.assert_called_once_with(docs)

    def test_upload_in_bucket(self):
        self.uploader.upload_in_bucket(self.mock_file)
        self.uploader.bucket.upload_object_from_path.assert_called_once_with(
            self.mock_file.file_path, self.mock_file.file_id
        )

    def test_upload_in_dynamo(self):
        # Call the method
        self.uploader.upload_in_dynamo(self.mock_file)

        # Assert that the store_file method was called with the correct arguments
        self.uploader.ddb.store_file.assert_called_once_with(
            "files", self.mock_file.file_id, self.mock_file.owner, self.mock_file.metadata.__dict__
        )


if __name__ == "__main__":
    unittest.main()