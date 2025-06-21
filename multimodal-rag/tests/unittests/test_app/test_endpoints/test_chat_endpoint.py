"""
Write pytests scripts that:
Setsup the TestClient and performs automatic tests on the '/chat' route
"""
from . import send_chat_request
from mm_rag.exceptions import MissingResponseContentError

from langchain_core.messages import AIMessage

from unittest.mock import patch


TEST_PROMPT = 'test'


@patch('mm_rag.agents.vlm.VLM.invoke', return_value=AIMessage(content='{"is_retrieval_required": true}'))
def test_chat_endpoint_with_retrieval_success(mock_classify):
  response = send_chat_request(TEST_PROMPT)

  assert response.status_code == 200


@patch('mm_rag.agents.vlm.VLM.invoke', return_value=AIMessage(content='{"is_retrieval_required": false}'))
def test_chat_endpoint_without_retrieval_success(mock_classify):
  response = send_chat_request(TEST_PROMPT)

  assert response.status_code == 200


def test_chat_endpoint_fails_with_empty_message():
  prompt = ''
  response = send_chat_request(prompt)

  assert response.status_code == 422


@patch('mm_rag.agents.vlm.VLM._generate', side_effect=MissingResponseContentError())
def test_chat_endpoint_fails_with_invalid_message_response(mock_failed_response):
  response = send_chat_request(TEST_PROMPT)

  assert response.status_code == 500

