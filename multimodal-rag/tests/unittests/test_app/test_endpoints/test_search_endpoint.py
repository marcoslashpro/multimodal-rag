"""
Write pytests scripts that:
Setsup the TestClient and performs automatic tests on the '/search' route
"""
from . import send_search_request, send_file_request
from mm_rag.exceptions import MalformedResponseError
from mm_rag.logging_service.log_config import create_logger
import mm_rag.utils as utils

import time
import pytest
from unittest.mock import patch


TEST_QUERY = 'test prompt'
mock_path = 'test.txt'
logger = create_logger(__name__)
with open(mock_path, 'w') as f:
  f.write(TEST_QUERY)
token = utils.get_secret()['bearer_pat']


def test_retrieval_200():
  #Upload whatever into the bucket
  with open(mock_path, 'rb') as f:
    send_file_request(mock_path, f, token)
  
  time.sleep(5)

  response = send_search_request(TEST_QUERY)

  assert response.status_code == 200
  assert response.json()


@patch("mm_rag.pipelines.retrievers.Retriever.retrieve", side_effect=MalformedResponseError())
def test_retrieval_missing_pinecone_response_failure(mock_bad_response):
  response = send_search_request(TEST_QUERY)

  assert response.status_code == 500


def test_empty_query_throws_422():
  query = ''
  response = send_search_request(query)

  assert response.status_code == 422