from . import mock_img_file as mock_file

from mm_rag.pipelines.uploaders import ImgUploader
from mm_rag.exceptions import ObjectUpsertionError
import mm_rag.datastructures as ds

import pytest
from unittest.mock import MagicMock, patch
import copy


class DummyVectorStore:
  index = MagicMock()
  namespace = MagicMock()


VECTOR_STORE = DummyVectorStore()

IMG_UPLOADER = ImgUploader(
  VECTOR_STORE,
  MagicMock(),
  MagicMock()
)


def test_vector_store_upload_success():
  with patch.object(IMG_UPLOADER.vector_store, 'upload') as mock_add:
    IMG_UPLOADER.upload_in_vector_store(mock_file)

  mock_add.assert_called_once_with(
    id=mock_file.docs[0].id,
    embeddings=mock_file.embeddings[0],
    metadata=mock_file.docs[0].metadata,
    collection=mock_file.metadata.author + f"/{mock_file.metadata.collection}"
  )


@patch("mm_rag.pipelines.utils.save_img_to_buffer", return_value='mock_buf_save')
def test_bucket_upload_success(mock_buf_save):
  with patch.object(IMG_UPLOADER.bucket, 'upload_object_from_file') as mock_add:
    IMG_UPLOADER.upload_in_bucket(mock_file)

  mock_add.assert_called_once_with(
    mock_buf_save.return_value,
    mock_file.docs[0].id,
  )


def test_bucket_upload_fails_missing_id():
  malformed_file = copy.deepcopy(mock_file)
  malformed_file.docs[0].id = None

  with pytest.raises(ObjectUpsertionError):
    IMG_UPLOADER.upload_in_bucket(malformed_file)


def test_bucket_upload_fails_missing_docs():
  malformed_file = copy.deepcopy(mock_file)
  malformed_file.docs = None

  with pytest.raises(ObjectUpsertionError):
    IMG_UPLOADER.upload_in_bucket(malformed_file)
