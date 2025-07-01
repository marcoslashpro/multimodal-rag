import mm_rag.datastructures as ds

import pytest
from unittest.mock import MagicMock, patch


class DummyMetadata:
  def __init__(self, file_type: str):
    self.file_type = file_type
    self.file_name = 'test'
    self.author = 'user'
    self.created = MagicMock()


@pytest.mark.parametrize('ext, exp', [
  ('.mp3', 'audio'),
  ('.jpeg', 'other'),
  ('.wav', 'audio'),
  ('.txt', 'other'),
  ('.mp4', 'other')
])
def test_correct_file_collection(ext, exp):
  assert ds.Metadata('test', ext, 'user').collection == exp