"""
Write pytests scripts that:
Setsup the TestClient and performs automatic tests on the '/cleanUp' route
"""
from . import send_cleanup_request, send_file_request, create_img
from mm_rag.exceptions import ObjectDeletionError
import mm_rag.datastructures as ds

from unittest.mock import patch

PATH = 'test.jpg'
create_img(PATH)


def test_cleanup_success():
  # We must upload something into the storages in order to delete successfully
  with open(PATH, 'rb') as f:
    send_file_request(PATH, f)

  response = send_cleanup_request()

  assert response.status_code == 200


@patch("mm_rag.entrypoints.cleanup", side_effect=ObjectDeletionError(ds.Storages.VECTORSTORE))
def test_cleanup_failure_with_empty_storages(mock_error):
  response = send_cleanup_request()

  assert response.status_code == 200