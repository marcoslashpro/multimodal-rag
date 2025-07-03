from . import PROP_RESPONSE
from mm_rag.pipelines.retrievers import Retriever
import mm_rag.datastructures as ds
from mm_rag.exceptions import MalformedResponseError

from langchain_core.documents import Document

from unittest.mock import MagicMock, patch, call
import pytest


class VectorStore:
  index = MagicMock()
  namespace = MagicMock()


retriever = Retriever(
  VectorStore(), MagicMock(), MagicMock(), MagicMock()
)


def test_retrieve_method():
  query = 'is this a test?'

  docs = retriever.retrieve(query)
  assert isinstance(docs, list)
  for doc in docs: assert isinstance(doc, Document)


def test_embed_called_with_right_args():
  query = 'is this a test?'
  with patch.object(retriever._embedder, 'embed', return_value=[0.1]) as mock_embed:
    retriever.retrieve(query)

    expected_embed_calls = [call(query, ds.Collection.AUDIO), call(query, ds.Collection.OTHER)]

    mock_embed.assert_has_calls(expected_embed_calls)

@pytest.mark.parametrize('namespace', [
  'diego',
  'marco',
  'testUser',
  'test user'
])
def test_vector_store_called_with_right_args(namespace):
  retriever._vector_store.namespace = namespace
  query = 'is this a test?'

  with patch.object(retriever._vector_store.index, "query") as mock_query,\
       patch.object(retriever._embedder, 'embed', return_value=[0.1]) as mock_embed:
    retriever.retrieve(query)

  expected_call = [
    call(
      top_k=retriever._top_k,
      vector=mock_embed.return_value,
      namespace=retriever._vector_store.namespace + ds.Collection.AUDIO.value,
      include_metadata=True
    ),
    call(
      top_k=retriever._top_k,
      vector=mock_embed.return_value,
      namespace=retriever._vector_store.namespace + ds.Collection.OTHER.value,
      include_metadata=True
    )
  ]

  mock_query.assert_has_calls(expected_call, any_order=True)


def test_transform_response_to_docs_success():
  docs = retriever.transform_response_to_docs(PROP_RESPONSE)

  assert isinstance(docs, list)
  for i, doc in enumerate(docs): 
    assert isinstance(doc, Document)
    assert doc.id == PROP_RESPONSE['matches'][i].get('id')
    assert doc.page_content == PROP_RESPONSE['matches'][i]['metadata']['text']
    assert doc.metadata['file_type'] == PROP_RESPONSE['matches'][i]['metadata']['file_type']


@pytest.mark.parametrize('malformed_response', [
  {
    'matches': [
      {"id": 'testId'}
    ]
  },
  {
    'matches': [
      {'metadata': 'test'}
    ]
  },
  {
    'matches': [
      {}
    ]
  }
])
def test_transform_response_to_docs_raises(malformed_response):
  with pytest.raises(MalformedResponseError):
    retriever.transform_response_to_docs(malformed_response)