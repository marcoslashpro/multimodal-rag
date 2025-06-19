from typing import TYPE_CHECKING
if TYPE_CHECKING:
  from mm_rag.agents.chatbot_flow import State

from mm_rag.logging_service.log_config import create_logger
from mm_rag.agents.prompts import retrieval_prompt, failed_retrieval_prompt


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
    doc_id = doc.id
    doc_type = doc.metadata.get("fileType")
    logger.debug(f"Working with doc {doc_id} of type {doc_type}")

    if not doc_type or not doc_id:
      return {
        "messages": [
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

    doc_type = doc_type.lower().lstrip(".")

    if doc_type in ['jpeg', 'png', 'jpg', 'pdf']:
      img_url = bucket.generate_presigned_url(doc_id)
      final_message_content.append(
        {
          "type": "image_url",
          "image_url": {
            "url": img_url
          }
        }
      )

    else:
      final_message_content.append(
        {
          "type": "text",
          "text": doc.page_content
        }
      )

  logger.debug(f"Augmented message content generated successfully: {final_message_content}")

  return {
    "messages": [
      {
        "role": "user",
        "content": final_message_content
      }
    ]
  }
