from mm_rag.pipelines.retrievers import Retriever
from mm_rag.processing.handlers import ImgHandler
from mm_rag.models import dynamodb, vectorstore, s3bucket

from unittest.mock import MagicMock, patch
import unittest

from PIL import Image

prop_response = {
  'matches': [
    {
      'id': 'rec3',
      'metadata': {
          'fileType': '.txt',
          'category': 'immune system',
          'text': 'Rich in vitamin C and other '
          'antioxidants, apples contribute to '
          'immune health and may reduce the '
          'risk of chronic diseases.'
        },
      'score': 0.82026422,
      'values': []
    },
    {
      'id': 'rec1',
      'metadata': {
          'fileType': '.jpeg',
          'text': 'fake_id'
        },
      'score': 0.793068111,
      'values': []
    },
  ],
 'namespace': 'example-namespace',
 '  usage': {'read_units': 6}}

class TestImgRetrieveMethod(unittest.TestCase):
  def setUp(self) -> None:
    self.mock_vector = MagicMock(vectorstore.PineconeVectorStore)
    self.mock_ddb = MagicMock(dynamodb.DynamoDB)
    self.mock_s3 = MagicMock(s3bucket.BucketService)
    self.mock_handler = MagicMock(ImgHandler)
    self.retriever = Retriever(
      self.mock_vector,
      self.mock_ddb,
      self.mock_s3,
      self.mock_handler
    )
    self.retriever.vector_store.namespace = MagicMock(str)

  def test_retrieve_method(self):
    mock_query = [0.1, 0.2, 0.3]

    with patch.object(self.retriever, 'retrieve', return_value=prop_response) as mock_retrieve,\
        patch.object(self.mock_handler, 'display', return_value=None) as mock_display:
        self.retriever.retrieve_and_display(mock_query)

        mock_retrieve.assert_called_once_with(
          mock_query
        )

        mock_display.assert_called_once_with(
          prop_response['matches'][1].get('id'),
          self.mock_s3
        )


if __name__ == "__main__":
  unittest.main()