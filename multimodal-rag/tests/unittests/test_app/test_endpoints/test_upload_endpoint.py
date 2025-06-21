import os

from unittest.mock import patch

from tempfile import NamedTemporaryFile

from mm_rag.exceptions import ImageTooBigError

from . import send_file_request, write_to_file, create_docx, create_img, create_test_pdf


# -----Tests-----#

def test_add_new_txt_file_success():
  test_file_path = 'test.txt'
  write_to_file(test_file_path,"Hello World")
  with open(test_file_path, 'rb') as f:
    response = send_file_request(test_file_path, f.read())

  assert response.status_code == 200

  if os.path.exists(test_file_path):
    os.remove(test_file_path)


def test_add_new_file_not_existing_path_failure():
  random_path = 'random/path'

  response = send_file_request(random_path, b'Test')
  assert response.status_code == 404

  if os.path.exists(random_path):
    os.remove(random_path)


def test_add_new_file_missing_file_failure():
  with NamedTemporaryFile() as f:
    f.write(b'')
    response = send_file_request(f.name, b'')
    assert response.status_code == 404


def test_add_new_image_file_success():
  test_img_path = 'test.jpg'
  create_img(test_img_path)
  with open(test_img_path, 'rb') as f:
    response = send_file_request(test_img_path, f)

  assert response.status_code == 200

  if os.path.exists(test_img_path):
    os.remove(test_img_path)


@patch("mm_rag.pipelines.extractors.ImgExtractor._extract_content", side_effect=ImageTooBigError())
def test_add_new_image_file_too_big_failure(failed_upload):
  test_img_path = 'test.jpg'
  create_img(test_img_path)
  with open(test_img_path, 'rb') as f:
    response = send_file_request(test_img_path, f)

  assert response.status_code == 413

  if os.path.exists('test.jpg'):
    os.remove('test.jpg')


TEST_PDF_PATH = 'test.pdf'


def test_pdf_upload_success():
  filename = 'test.pdf'
  create_test_pdf(filename)

  with open(filename, 'rb') as f:
    response = send_file_request(filename, f)

  assert response.status_code == 200

  if os.path.exists(filename):
    os.remove(filename)


TEST_DOC_PATH = 'test.docx'


def test_doc_upload_success():
  create_docx(TEST_DOC_PATH)
  with open(TEST_DOC_PATH, 'rb') as f:
    response = send_file_request(TEST_DOC_PATH, f)

  assert response.status_code == 200

  if os.path.exists(TEST_DOC_PATH):
    os.remove(TEST_DOC_PATH)