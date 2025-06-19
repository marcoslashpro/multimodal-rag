from mm_rag.agents.chatbot_flow import formatter, State
from mm_rag.models.s3bucket import BucketService

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
          metadata={"fileType": ".txt"}
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

  def test_correct_img_message_creation(self):
    mock_bucket = MagicMock(BucketService)
    mock_bucket.generate_presigned_url.return_value = 'test_url'

    supported_types = ['jpeg', 'png', 'jpg', 'pdf']

    for file_type in supported_types:
      with self.subTest(file_type=file_type):

        mock_state = {
          'retrieved': [
            Document(
              page_content='this is a test',
              id='123',
              metadata={"fileType": file_type}
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
          metadata={"fileType": "txt"}
        ),
        Document(
          page_content='test image',
          id='123',
          metadata={"fileType": 'jpeg'}
        ),
        Document(
          page_content='test image',
          id='123',
          metadata={"fileType": 'pdf'}
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

  def test_missing_doc_info(self):
    wrong_docs = [
      Document(
        page_content='missing id',
        metadata={'fileType': 'txt'}
      ),
      Document(
        page_content='missing file type',
        id='123'
      ),
      Document(
        page_content='missing both'
      )
    ]

    for doc in wrong_docs:
      with self.subTest(doc=doc):

        mock_state = {
          'retrieved': [doc],
          'retriever': MagicMock(),
          'vlm': MagicMock(),
          'is_retrieval_required': 'True',
          'bucket': MagicMock(),
        }

        expected_error_message_prompt = {
            "messages": [
              {
                'role': 'user', 'content': [
                {
                  "type": "text",
                  "text": """Unfortunately, the retrieved step failed because of some outside reasons. Inform the user and ask if he would like you to try again or respond without retrieval.""".strip()
                }
              ]
            }
          ]
        }

        formatted = formatter.formatter(mock_state)  # type: ignore

        self.assertDictEqual(expected_error_message_prompt, formatted)

if __name__ == "__main__":
  unittest.main()