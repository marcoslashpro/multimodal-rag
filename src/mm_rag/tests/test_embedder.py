import base64
import json
import os
from mm_rag.agents.mm_embedder import Embedder

import unittest
from unittest.mock import MagicMock, patch, call

from PIL import Image

class TestEmbedder(unittest.TestCase):
  def setUp(self):
    self.embedder = Embedder()
    self.txt_content = ['this', 'is', 'a', 'test']
    self.img = Image.new('RGB', (100, 100), 'red')
    self.img_path = 'test_img.jpeg'
    self.img.save(self.img_path)
    with open(self.img_path, 'rb') as f:
      self.encoded_img = base64.b64encode(f.read()).decode("utf-8")

  def tearDown(self) -> None:
    os.remove(self.img_path)

  def test_embed_query(self) -> None:
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({'embedding': [0.1, 0.2, 0.3]})

    mock_response = {'body': mock_body}
    with patch.object(self.embedder.client, 'invoke_model',
                      return_value=mock_response) as mock_embed:
      result = self.embedder.embed_query(self.txt_content[0])

      self.assertIsInstance(
        result,
        list
      )
      mock_embed.assert_called_once_with(
        modelId='amazon.titan-embed-image-v1',
        body='{"inputText": "this"}',
        accept='application/json',
        contentType='application/json'
      )
      self.assertEqual(
        result,
        [0.1, 0.2, 0.3]
      )

  def test_embed_img(self) -> None:
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps({'embedding': [0.1, 0.2, 0.3]})

    mock_response = {'body': mock_body}
    with patch.object(self.embedder.client, 'invoke_model',
                      return_value=mock_response) as mock_embed:
      result = self.embedder.embed_img(self.encoded_img)
      expected_payload = json.dumps({"inputImage": self.encoded_img})

      self.assertIsInstance(
        result,
        list
      )
      mock_embed.assert_called_once_with(modelId='amazon.titan-embed-image-v1', body=expected_payload)
      self.assertEqual(
        result,
        [0.1, 0.2, 0.3]
      )

  def test_embed_docs(self) -> None:
    excepcted_calls = [call(val) for val in self.txt_content]
    with patch.object(self.embedder, 'embed_query') as mock_txt_embed:
      self.embedder.embed_documents(self.txt_content)

      self.assertEqual(mock_txt_embed.call_count, len(self.txt_content))
      mock_txt_embed.assert_has_calls(excepcted_calls)

    with self.assertRaises(ValueError):
      self.embedder.embed_documents(1)  # type: ignore[Invalid type]

if __name__ == "__main__":
  unittest.main()