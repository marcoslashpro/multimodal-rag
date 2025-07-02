"""
Write pytests scripts that:
Setsup the TestClient and performs automatic tests on the '/cleanUp' route
"""
from . import send_cleanup_request, send_file_request, create_img
from mm_rag.exceptions import ObjectDeletionError
import mm_rag.datastructures as ds
from mm_rag.utils import get_secret

from unittest.mock import patch

PATH = 'test.jpg'
create_img(PATH)
token = get_secret()['bearer_pat']
visitor_token = get_secret()['visitor_pat']


def test_cleanup_success():
  # We must upload something into the storages in order to delete successfully
  with open(PATH, 'rb') as f:
    send_file_request(PATH, f, token)

  response = send_cleanup_request(token)

  assert response.status_code == 200


@patch("mm_rag.entrypoints.cleanup", side_effect=ObjectDeletionError(ds.Storages.VECTORSTORE))
def test_cleanup_failure_with_empty_storages(mock_error):
  response = send_cleanup_request(token)

  assert response.status_code == 200


def test_cleanup_right_namespace():
  # We must upload something into the storages within a given namespace
  with open(PATH, 'rb') as f:
    send_file_request(PATH, f, token)

  # We upload in another namespace
  with open(PATH, 'rb') as f:
    send_file_request(PATH, f, visitor_token)

  # We clean only a specific namespace
  response = send_cleanup_request(token)

  assert response.status_code == 200

  # We assert the other namespace still contains files by trying to clean it again
  # If it throws a `ObjectDeletionError`, then we have failed
  assert send_cleanup_request(visitor_token).status_code == 200