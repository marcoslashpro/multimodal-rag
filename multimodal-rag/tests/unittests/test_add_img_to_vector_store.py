from mm_rag.models.vectorstore import PineconeVectorStore
from mm_rag.agents.mm_embedder import Embedder
from mm_rag.processing.base import Metadata
from mm_rag.config.config import config
from mm_rag.utils import get_secret

from pinecone import Vector

import unittest
from unittest.mock import MagicMock, patch

from PIL import Image


class TestImgUpsertion(unittest.TestCase):
  def setUp(self):
    self.mock_embedder = MagicMock(Embedder)
    self.vector_store = PineconeVectorStore(
      self.mock_embedder,
      api_key=get_secret()['pinecone_api_key'],
      index_name=config['pinecone']['index_name'],
      namespace='mock_namespace',
      cloud='aws',
      region=MagicMock(str)
    )
    self.mock_img = MagicMock(Image.Image)
    self.mock_metadata = MagicMock(Metadata)
    self.mock_metadata.__dict__ = {'fileId': 'fakeId'}
    self.mock_embedder.embed_img.return_value = [float(i) for i in range(1024)]

  def tearDown(self) -> None:
    self.vector_store.vector_store.delete(['fakeId'])

  def test_add_img(self):
    self.assertTrue(
      self.vector_store.add_image(self.mock_img, self.mock_metadata, 'fakeId'),
    )

  @patch('mm_rag.models.vectorstore.PineconeVectorStore.index')
  def test_upsert_to_right_namespace(self, mock_index):
    mock_upsert = MagicMock()
    mock_index.upsert = mock_upsert
    self.vector_store.add_image(self.mock_img, self.mock_metadata, 'fakeId')
    expected_value = self.mock_embedder.embed_img.return_value

    mock_upsert.assert_called_once_with(
     [
       Vector(
        id='fakeId',
        values=expected_value,
        metadata=self.mock_metadata.__dict__,
        sparse_values=None
      )
    ],
    namespace=self.vector_store.namespace
    )


if __name__ == "__main__":
  unittest.main()