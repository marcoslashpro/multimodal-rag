from . import metadata, mock_file

from mm_rag.pipelines.uploaders import Uploader
import mm_rag.datastructures as ds
from mm_rag.exceptions import ObjectUpsertionError

import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import copy

from langchain_core.documents import Document


class BaseUploader(Uploader):
  def upload_in_vector_store(self, file: ds.File) -> bool:
    return super().upload_in_vector_store(file)
  
  def upload_in_bucket(self, file: ds.File) -> bool:
    return super().upload_in_bucket(file)
mock_vs = MagicMock()
UPLOADER = BaseUploader(
  mock_vs, MagicMock(), MagicMock()
)


@pytest.mark.parametrize('malformed_file', [
  ds.File(
    metadata=metadata,
    content='test_content',
    docs=[Document(page_content='test_content', id=metadata.file_id, metadata=metadata.__dict__)],
    embeddings=[[0.1], [0.2]]
  ),
  ds.File(
    metadata=metadata,
    content='test_content',
    docs=[
      Document(page_content='test_content', id=metadata.file_id, metadata=metadata.__dict__),
      Document(page_content='test_content', id=metadata.file_id, metadata=metadata.__dict__)],
    embeddings=[[0.1]]
  )
])
def test_vector_store_upload_fail_mismatch_length(malformed_file):
  with pytest.raises(ObjectUpsertionError):
    UPLOADER.upload_in_vector_store(malformed_file) 


def test_bucket_upload_fails_missing_docs():
  malformed_file = copy.deepcopy(mock_file)
  malformed_file.docs = []

  with pytest.raises(ObjectUpsertionError):
    UPLOADER.upload_in_bucket(malformed_file)


@pytest.mark.asyncio
async def test_vectorstore_rollback_on_bucket_failure():
    file = mock_file
    with patch.object(Uploader, 'aupload_in_vector_store', new_callable=AsyncMock) as mock_vs_upload, \
         patch.object(Uploader, 'aupload_in_bucket', new_callable=AsyncMock, side_effect=ObjectUpsertionError(storage=ds.Storages.BUCKET)), \
         patch.object(UPLOADER.vector_store, 'remove_object') as mock_vs_remove:

        await UPLOADER.aupload(file)

        mock_vs_upload.assert_awaited_once_with(file)
        mock_vs_remove.assert_called_once_with(file.metadata.file_id)


@pytest.mark.asyncio
async def test_bucket_rollback_on_vectorstore_failure():
    file = mock_file
    with patch.object(Uploader, 'aupload_in_vector_store', new_callable=AsyncMock, side_effect=ObjectUpsertionError(storage=ds.Storages.VECTORSTORE)), \
         patch.object(Uploader, 'aupload_in_bucket', new_callable=AsyncMock) as mock_b_upload,\
         patch.object(UPLOADER.bucket, 'remove_object') as mock_b_remove:

        await UPLOADER.aupload(file)

        mock_b_upload.assert_awaited_once_with(file)
        mock_b_remove.assert_called_once_with(file.metadata.file_id)


@pytest.mark.parametrize('collection', [
  'audio', 'other'
])
def test_right_namespace_creation(collection):
  assert UPLOADER._generate_full_namespace(mock_file.metadata.author, collection) == \
         f"{mock_file.metadata.author}/{collection}"