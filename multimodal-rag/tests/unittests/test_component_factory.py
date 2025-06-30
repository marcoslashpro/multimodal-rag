from mm_rag.pipelines.extractors import *
from mm_rag.pipelines.uploaders import *
from mm_rag.pipelines.pipes import ComponentFactory

import pytest
from unittest.mock import patch, MagicMock


factory = ComponentFactory(MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
auth = 'user'


@pytest.mark.parametrize('path, expected', [
  ('test.jpeg', ImgExtractor),
  ('test.txt', TxtExtractor),
  ('test.pdf', PdfExtractor),
  ('test.docx', DocExtractor),\
  ('test.png', ImgExtractor),
  ('test.jpg', ImgExtractor)
])
def test_right_extractor(path, expected):
  assert isinstance(factory.get_extractor(path, auth), expected)


@pytest.mark.parametrize('path, expected', [
  ('test.jpeg', ImgUploader),
  ('test.txt', TxtUploader),
  ('test.pdf', PdfUploader),
  ('test.docx', PdfUploader),\
  ('test.png', ImgUploader),
  ('test.jpg', ImgUploader)
])
def test_right_uploader(path, expected):
  assert isinstance(factory.get_uploader(path, auth), expected)


@pytest.mark.parametrize('unsupported', [
  'foo.unsupported',
  'foo.stupid',
  'foo.example'
])
def test_unsupported_file_type_get_extractor_raise(unsupported):
  with pytest.raises(FileNotValidError):
    factory.get_extractor(unsupported, auth)


@pytest.mark.parametrize('unsupported', [
  'foo.unsupported',
  'foo.stupid',
  'foo.example'
])
def test_unsupported_file_type_get_uploader_raise(unsupported):
  with pytest.raises(FileNotValidError):
    factory.get_uploader(unsupported, auth)