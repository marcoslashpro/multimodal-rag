from mm_rag.models.vectorstore import PineconeVectorStore

from unittest.mock import MagicMock, patch
import unittest



from mm_rag.models.vectorstore import PineconeVectorStore
from mm_rag.agents.mm_embedder import Embedder
from mm_rag.processing.base import Metadata
from mm_rag.config.config import config
from mm_rag.utils import get_secret

from pinecone import Vector

import unittest
from unittest.mock import MagicMock, patch

from PIL import Image


class TestPineconeVectorStore(unittest.TestCase):
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

  def test_not_upsert_if_already_exists(self):
    pass


if __name__ == "__main__":
  unittest.main()
