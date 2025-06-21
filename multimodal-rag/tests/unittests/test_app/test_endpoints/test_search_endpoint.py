"""
Write pytests scripts that:
Setsup the TestClient and performs automatic tests on the '/search' route
"""
from . import send_search_request
from mm_rag.exceptions import MalformedResponseError
from mm_rag.logging_service.log_config import create_logger

import pytest
from unittest.mock import patch


TEST_QUERY = 'test prompt'
logger = create_logger(__name__)


def test_retrieval_success():
  response = send_search_request(TEST_QUERY)

  assert response.status_code == 200


@patch("mm_rag.pipelines.retrievers.Retriever.retrieve", side_effect=MalformedResponseError())
def test_retrieval_missing_pinecone_response_failure(mock_bad_response):
  response = send_search_request(TEST_QUERY)

  assert response.status_code == 500


def test_empty_query_throws_422():
  query = ''
  response = send_search_request(query)

  assert response.status_code == 422