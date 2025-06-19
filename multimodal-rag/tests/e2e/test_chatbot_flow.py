from fastapi.testclient import TestClient
from mm_rag.app.main import app
from mm_rag.app.routes.views.chat import ChatInput
from mm_rag.utils import get_secret
from os import environ

import pytest


test_client = TestClient(app)


def test_status_ok():
  response = test_client.post(
    "/chat",
    headers={"Authorization": f"Bearer {get_secret()['bearer_pat']}"},
    json={"query": "is this a test?"}
    )

  assert response.status_code == 200


@pytest.mark.parametrize("url", [
  '/upload-file',
  '/search',
  '/chat'
]
)
def test_missing_auth(url):
  response = test_client.post(
    url,
    json={"":""}
  )

  assert response.status_code == 403


def test_retrieval_response():
  response = test_client.post(
    '/chat',
    headers={
      "Authorization": f"Bearer {get_secret()['bearer_pat']}"
    },
    json={
      "query": "Retrieve extra relevant info on rl agents"
    }
  )

  assert response.status_code == 200