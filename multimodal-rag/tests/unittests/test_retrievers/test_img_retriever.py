from mm_rag.pipelines.retrievers import Retriever
from mm_rag.processing.handlers import ImgHandler
from mm_rag.models import dynamodb, vectorstore, s3bucket
from mm_rag.agents.mm_embedder import Embedder

from unittest.mock import MagicMock, patch
import unittest

from langchain_core.documents import Document
from pinecone.data import QueryResponse

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
    self.mock_embedder = MagicMock(Embedder)
    self.retriever = Retriever(
      self.mock_vector,
      self.mock_ddb,
      self.mock_s3,
      self.mock_embedder,
      self.mock_handler
    )
    self.retriever._vector_store.namespace = MagicMock(str)
    self.prop_response: list[Document] = [
      Document(
        page_content=prop_response['matches'][0]['metadata']['text'],
        id=prop_response['matches'][0]['id'],
        metadata=prop_response['matches'][0]['metadata']
      ),
      Document(
        page_content=prop_response['matches'][1]['metadata']['text'],
        id=prop_response['matches'][1]['id'],
        metadata=prop_response['matches'][1]['metadata']
      )
    ]

  def test_retrieve_method(self):
    mock_query = 'mock_query'
    mock_response = [MagicMock(QueryResponse)]
    mock_response[0].id = prop_response['matches'][0].get('id')
    mock_response[0].metadata = prop_response['matches'][0].get('metadata')

    with patch('mm_rag.pipelines.retrievers.Retriever.retrieve', return_value=mock_response) as mock_retrieve,\
        patch.object(self.retriever._handler, 'display', return_value=None) as mock_display,\
        patch.object(self.retriever._embedder, 'embed_query', return_value=[0.1, 0.2]):
        self.retriever.retrieve(mock_query)

        mock_retrieve.assert_called_once_with(
          mock_query
        )


if __name__ == "__main__":
  unittest.main()