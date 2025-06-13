import os
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from mm_rag.pipelines.pipes import Piper
from mm_rag.exceptions.models_exceptions import ObjectUpsertionError
from mm_rag.models.vectorstore import PineconeVectorStore


@pytest_asyncio.fixture
async def piper():
    p = Piper(
        MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
    )
    yield p


@pytest.fixture
def mock_path(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("This is a test")
    return str(file_path)


@pytest.mark.asyncio
async def test_arun_upload_rollback_vector_store_on_error(piper, mock_path):
    mock_uploader = MagicMock()
    mock_uploader.aupload_in_vector_store = AsyncMock(side_effect=ObjectUpsertionError("PineconeVectorStore"))
    mock_uploader.aupload_in_bucket = AsyncMock()

    with patch.object(piper._processor_factory, 'get_processor', return_value=MagicMock()) as mock_processor,\
      patch.object(piper._file_factory, 'get_file', return_value=MagicMock()),\
      patch.object(piper._vector_store_factory, 'get_vector_store', return_value=MagicMock()),\
      patch.object(piper._uploader_factory, 'get_uploader', return_value=mock_uploader),\
      patch.object(piper._s3, 'remove_object', return_value=None) as mock_remove,\
      patch.object(piper._retriever_factory, 'get_retriever', return_value=MagicMock()):

        await piper.arun_upload(mock_path, 'mockUser')

        mock_remove.assert_called_once()


@pytest.mark.asyncio
async def test_arun_upload_rollback_bucket_on_error(piper, mock_path):
    mock_vector_store_instance = MagicMock()
    mock_uploader = MagicMock()
    mock_uploader.aupload_in_vector_store = AsyncMock()
    mock_uploader.aupload_in_bucket = AsyncMock(side_effect=ObjectUpsertionError("BucketService"))

    with patch.object(piper._processor_factory, 'get_processor', return_value=MagicMock()),\
         patch.object(piper._file_factory, 'get_file', return_value=MagicMock()),\
         patch.object(piper._vector_store_factory, 'get_vector_store', return_value=mock_vector_store_instance),\
         patch.object(piper._uploader_factory, 'get_uploader', return_value=mock_uploader),\
         patch.object(mock_vector_store_instance, 'remove_object', return_value=None) as mock_remove,\
         patch.object(piper._retriever_factory, 'get_retriever', return_value=MagicMock()):

        await piper.arun_upload(mock_path, 'mockUser')

        mock_remove.assert_called_once()
