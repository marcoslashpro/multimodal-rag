from . import mock_file

from mm_rag.pipelines.uploaders import TxtUploader
from mm_rag.exceptions import ObjectUpsertionError
import mm_rag.datastructures as ds

import pytest
from unittest.mock import MagicMock, patch
import copy


class DummyVectorStore:
  index = MagicMock()
  namespace = MagicMock()


VECTOR_STORE = DummyVectorStore()

TXT_UPLOADER = TxtUploader(
  VECTOR_STORE,
  MagicMock(),
  MagicMock()
)


def test_vector_store_upload_success():
  with patch.object(TXT_UPLOADER.vector_store, 'upload') as mock_add:
    TXT_UPLOADER.upload_in_vector_store(mock_file)

  mock_add.assert_called_once_with(
    id=mock_file.docs[0].id,
    embeddings=mock_file.embeddings[0],
    metadata=mock_file.docs[0].metadata,
    collection=mock_file.metadata.author + f"/{mock_file.metadata.collection}"
  )


def test_bucket_upload_success():
  with patch.object(TXT_UPLOADER.bucket, 'upload_object') as mock_add:
    TXT_UPLOADER.upload_in_bucket(mock_file)

  mock_add.assert_called_once_with(
    mock_file.docs[0].id,
    mock_file.docs[0].page_content
  )


def test_bucket_upload_fails_missing_id():
  malformed_file = copy.deepcopy(mock_file)
  malformed_file.docs[0].id = None

  with pytest.raises(ObjectUpsertionError):
    TXT_UPLOADER.upload_in_bucket(malformed_file)

def test_bucket_upload_fails_missing_docs():
  malformed_file = copy.deepcopy(mock_file)
  malformed_file.docs = None

  with pytest.raises(ObjectUpsertionError):
    TXT_UPLOADER.upload_in_bucket(malformed_file)
