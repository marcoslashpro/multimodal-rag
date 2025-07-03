from mm_rag.agents.chatbot_flow import formatter, State
from mm_rag.models.s3bucket import BucketService
import mm_rag.datastructures as ds

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


class TestMessageCreation(unittest.TestCase):
  def setUp(self) -> None:
    self.mock_state = MagicMock(State)
    self.maxDiff = None
    self.query = None

  def tearDown(self) -> None:
    return super().tearDown()

  def test_correct_text_message_creation(self):
    mock_state = {
      'retrieved': [
        Document(
          page_content='this is a test',
          id='123',
          metadata={"file_type": ".txt"}
        )
      ],
      'retriever': MagicMock(),
      'vlm': MagicMock(),
      'is_retrieval_required': 'True',
      'bucket': MagicMock(),
    }

    expected_text_message = {"messages" : [
        {
          "role": "user", "content": [
          {
            "type": "text",
            "text": f"Your role here is to answer the user's original query in the most relevant way possible. The original query is: '{self.query}'. The relevant information needed to answer this query are provided in the docs after this message. If you do not have enough information in the provided docs, then tell the user that you were not able to find enough relevant information to answer to his query. The docs: "
          },
          {"type": "text", "text": "this is a test"}
        ]
        }
      ]
    }

    formatted = formatter.formatter(
      mock_state  # type: ignore
    )
    self.assertDictEqual(expected_text_message, formatted)

  def test_code_message_creation(self):

    expected_code_message = {"messages" : [
        {
          "role": "user", "content": [
          {
            "type": "text",
            "text": f"Your role here is to answer the user's original query in the most relevant way possible. The original query is: '{self.query}'. The relevant information needed to answer this query are provided in the docs after this message. If you do not have enough information in the provided docs, then tell the user that you were not able to find enough relevant information to answer to his query. The docs: "
          },
          {"type": "text", "text": "this is code"}
        ]
        }
      ]
    }

    for file_type in ds.FileType.CODE.value:
      with self.subTest(file_type=file_type):
        mock_state = {
          'retrieved': [
            Document(
              page_content='this is code',
              id='123',
              metadata={"file_type": file_type}
            )
          ],
          'retriever': MagicMock(),
          'vlm': MagicMock(),
          'is_retrieval_required': 'True',
          'bucket': MagicMock(),
        }

        formatted = formatter.formatter(state=mock_state)
        self.assertDictEqual(formatted, expected_code_message)

  def test_correct_img_message_creation(self):
    mock_bucket = MagicMock(BucketService)
    mock_bucket.generate_presigned_url.return_value = 'test_url'

    supported_types = [ds.FileType.PDF.value, ds.FileType.DOCX.value]
    supported_types.extend(ds.FileType.IMAGE.value)

    for file_type in supported_types:
      with self.subTest(file_type=file_type):

        mock_state = {
          'retrieved': [
            Document(
              page_content='this is a test',
              id='123',
              metadata={"file_type": file_type}
            )
          ],
          'retriever': MagicMock(),
          'vlm': MagicMock(),
          'is_retrieval_required': 'True',
          'bucket': mock_bucket,

        }

        expected_img_message = {
          "messages": [
            {
            "role": "user", "content": [
              {
                "type": "text",
                "text": f"Your role here is to answer the user's original query in the most relevant way possible. The original query is: '{self.query}'. The relevant information needed to answer this query are provided in the docs after this message. If you do not have enough information in the provided docs, then tell the user that you were not able to find enough relevant information to answer to his query. The docs: "
              },
              {
                "type": "image_url",
                "image_url": {"url": "test_url"}
              }
            ]
          }
        ]
      }

        formatted = formatter.formatter(mock_state)  # type: ignore
        self.assertDictEqual(formatted, expected_img_message)

  def test_mixed_message_creation(self):
    mock_bucket = MagicMock(BucketService)
    mock_bucket.generate_presigned_url.return_value = 'test_url'

    mock_state = {
      'retrieved': [
        Document(
          page_content='test text',
          id='123',
          metadata={"file_type": ".txt"}
        ),
        Document(
          page_content='test image',
          id='123',
          metadata={"file_type": '.jpeg'}
        ),
        Document(
          page_content='test image',
          id='123',
          metadata={"file_type": '.pdf'}
        )
      ],
      'retriever': MagicMock(),
      'vlm': MagicMock(),
      'is_retrieval_required': 'True',
      'bucket': mock_bucket,
    }
    expected_message_format = {
      'messages': [
        {
          "role": "user", "content": [
          {
            "type": "text",
            "text": f"Your role here is to answer the user's original query in the most relevant way possible. The original query is: '{self.query}'. The relevant information needed to answer this query are provided in the docs after this message. If you do not have enough information in the provided docs, then tell the user that you were not able to find enough relevant information to answer to his query. The docs: "
          },
          {
            "type": "text",
            "text": "test text"
          },
          {
            "type": "image_url",
            "image_url": {"url": "test_url"}
          },
          {
            "type": "image_url",
            "image_url": {"url": "test_url"}
          }
        ]
      }
    ]
  }

    formatted = formatter.formatter(mock_state)  # type: ignore[call-args]
    print(f"Formatted documents: {formatted}")

    self.assertDictEqual(formatted, expected_message_format)

  def test_missing_retrieval(self):
    mock_state = {
      'retrieved': None,
      'retriever': MagicMock(),
      'vlm': MagicMock(),
      'is_retrieval_required': 'True',
      'bucket': MagicMock(),
    }
    expected_error_message_prompt = {'messages': [
      {
      'role': 'user', 'content': [
        {
          "type": "text",
          "text": """Unfortunately, the retrieved step failed because of some outside reasons. Inform the user and ask if he would like you to try again or respond without retrieval.""".strip()
        }
      ]
    }
    ]}
    missing_retrieval_info = formatter.formatter(mock_state)  # type: ignore

    self.assertDictEqual(missing_retrieval_info, expected_error_message_prompt)


if __name__ == "__main__":
  unittest.main()