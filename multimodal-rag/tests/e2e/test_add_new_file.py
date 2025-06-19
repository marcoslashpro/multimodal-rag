import pytest

from fastapi.testclient import TestClient

from io import BytesIO

from mm_rag.app.main import app
from mm_rag.utils import get_secret


test_client = TestClient(app)
test_token = "4BDzGbShH6BK6cANeA3dhrE7GTMyn2wi3tfBS8FvLs8"


def test_add_new_file_success():
  test_file_content = b'Hello World!'

  response = test_client.post(
    '/upload-file',
    headers={"Authorization": f"Bearer {get_secret()['bearer_pat']}"},
    files={"file": ("test.txt", BytesIO(test_file_content), "text/plain")}
  )

  assert response.status_code == 200
