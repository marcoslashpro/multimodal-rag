from mm_rag.models.vectorstore import PineconeVectorStore
from mm_rag.agents.mm_embedder import Embedder
from mm_rag.datastructures import Metadata
from mm_rag.config.config import config
from mm_rag.utils import get_secret
import mm_rag.datastructures as ds
from mm_rag.exceptions import ObjectUpsertionError

from langchain_core.documents import Document

import unittest
from unittest.mock import MagicMock, patch
import pytest

from PIL import Image


mock_embedder = MagicMock(Embedder)
vector_store = PineconeVectorStore(
  mock_embedder,
  api_key=get_secret()['pinecone_api_key'],
  index_name=config['pinecone']['index_name'],
  namespace='mock_namespace',
  cloud='aws',
  region=MagicMock(str)
)
mock_img = MagicMock(Image.Image)
mock_metadata = Metadata('test', '', 'user')
mock_embedder.embed_img.return_value = [float(i) for i in range(1024)]
mock_docs = [Document(page_content='test', metadata=mock_metadata.__dict__, id=mock_metadata.file_id)]
mock_embeddings = [[0.1]]


file = ds.File(mock_metadata, 'test', mock_docs, mock_embeddings)


@pytest.mark.parametrize('docs, emb', [
  ([Document(page_content='test')], [[0.1]*2]),
  ([Document(page_content='test')]*2, [[0.1]]),
])
def test_add_throws_mismatching_length(docs, emb):
  file.docs = docs
  file.embeddings = emb
  with pytest.raises(ObjectUpsertionError):
    vector_store.add(file)


def test_add_throws_missing_id():
  file.docs = [Document(page_content='test')]
  file.embeddings = [[0.1]]
  with pytest.raises(ObjectUpsertionError):
    vector_store.add(file)


@pytest.mark.parametrize('collection', [
  'audio', 'other'
])
def test_right_namespace_creation(collection):
  assert vector_store._generate_full_namespace(collection) == \
         f"{vector_store.namespace}/{collection}"


if __name__ == "__main__":
  unittest.main()
