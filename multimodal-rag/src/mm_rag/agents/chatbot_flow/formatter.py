from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from mm_rag.agents.chatbot_flow import State

from mm_rag.logging_service.log_config import create_logger
from mm_rag.agents.prompts import retrieval_prompt, failed_retrieval_prompt

import mm_rag.datastructures as ds


logger = create_logger(__name__)


def formatter(state: 'State'):
  retrieved = state.get('retrieved')
  bucket = state['bucket']
  messages = state.get("messages", [])
  query = state.get('query')

  logger.debug(f"Retrieved from the state: {retrieved = }\n{bucket = }\n{messages = }\n{query = }")

  rag_prompt = retrieval_prompt.invoke({'query': query}).to_string().removeprefix("Human: ")

  final_message_content: list[dict[str, str | dict[str, str]]] = [{
    "type": "text",
    "text": rag_prompt
  }]

  if not retrieved:
    logger.debug(f"Unable to generate retrieval information in node 'formatter'")
    return {'messages': [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": failed_retrieval_prompt
          }
        ]
      }
    ]
    }

  for doc in retrieved:
    doc_type = doc.metadata.get('file_type')
    logger.debug(f'Doc type: {doc_type}')
    doc_id = doc.id
    if not doc_type or not doc_id:
      continue

    logger.debug(f"Working with doc {doc_id} of type {doc_type}")

    if doc_type in ds.FileType.IMAGE.value or doc_type == ds.FileType.PDF.value or doc_type == ds.FileType.DOCX.value:
      logger.debug(f"{doc_type=}  in: {ds.Img, ds.FileType.PDF.value, ds.FileType.DOCX.value}")
      img_url = bucket.generate_presigned_url(doc_id)
      final_message_content.append(
        {
          "type": "image_url",
          "image_url": {
            "url": img_url
          }
        }
      )

    if doc_type == ds.FileType.TXT.value or doc_type in ds.FileType.CODE.value:
      final_message_content.append(
        {
          "type": "text",
          "text": doc.page_content
        }
      )

    else:
      continue

  logger.debug(f"Augmented message content generated successfully: {final_message_content}")

  return {
    "messages": [
      {
        "role": "user",
        "content": final_message_content
      }
    ]
  }
