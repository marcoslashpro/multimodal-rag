from typing import Annotated, Literal, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dataclasses import asdict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from mm_rag.pipelines.retrievers import Retriever
from mm_rag.logging_service.log_config import create_logger
from mm_rag.processing.handlers import ImgHandler
from mm_rag.models.s3bucket import BucketService
from mm_rag.agents.vlm import VLM, Content


logger = create_logger(__name__)


class InputClassifier(BaseModel):
  is_retrieval_required: str = Field(
      ...,
      description="Decide if external information is required in order to "
      "properly respond to the user's query."
  )


class State(TypedDict):
  messages: Annotated[list, add_messages]
  is_retrieval_required: Literal['True', 'False'] | None
  retriever: Retriever
  vlm: VLM
  retrieved: list[Document] | None
  img_handler: ImgHandler
  bucket: BucketService
  query: str

logger.debug(f"Created new State Schema: {State}")

builder = StateGraph(State)
logger.debug(f"Created graph builder: {builder}")


def classify_input(state: State):
  query = state['query']
  vlm = state['vlm']
  logger.debug(f"Classifying input: {query}")

  result = vlm.invoke(
    [
      {
        "role": "system",
        "content": [
          {
            "type": "text",
            "text": """determine if extra external information is needed to answer the user's query. Use this tool anytime that the user asks something specific and the query lacks context. Return: True: if extra info is required, False: if extra info is NOT required DO NOT return anything else except 'True' or 'False' as needed."""
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": query
          }
        ]
      }
    ]
  )

  logger.debug(f"Got result of: {result}")
  result_content = result.content

  if not isinstance(result_content, str):
    logger.error(f"Found response of type: {type(result)}")
    raise ValueError(
      f"Expected a string result from the model invokation, got {type(result_content)}:\nContent: {result_content}"
    )

  parsed = InputClassifier.model_validate(
    {"is_retrieval_required": result_content.strip()}
  )
  logger.debug(f"Parsed model output: {parsed}")

  return {"is_retrieval_required": parsed.is_retrieval_required, 'query': query}


def router(state: State):
  is_retrieval_required = state.get('is_retrieval_required', 'True')
  if is_retrieval_required == 'True':
    logger.debug(f"Routing to `retrieve`")
    return {"next": "retrieve"}

  logger.debug(f"Routing to `chatbot`")
  return {"next": "chatbot"}


def chatbot(state: State):
  vlm = state['vlm']
  last_message = state['messages'][-1]

  logger.debug(f'Running inference on the model with: {last_message}')

  try:
    response = vlm.invoke([last_message])
  except Exception as e:
    logger.error(e)
    raise

  return {
    "messages": [
      {
        "role": "assistant",
        "content": response.content
      }
    ]
  }


def retrieve(state: State):
  retriever = state['retriever']
  last_message = state['messages'][-1]
  query = state['query']
  logger.debug(f"Running retrieval on input: {last_message}")

  retrieved: list[Document] = retriever.invoke(last_message.content)
  logger.debug(f"Found in the VectorStore: {retrieved}")

  return {"retrieved": retrieved, 'query': query}


def formatter(state: State) -> dict[str, dict[str, str | list[dict[str, str | list[dict[str, str | dict[str, str]]]]]]]:
  retrieved = state.get('retrieved')
  handler = state['img_handler']
  bucket = state['bucket']
  messages = state.get("messages", [])
  query = state.get('query')
  logger.debug(f"Retrieved from the state: {retrieved = }\n{handler = }\n{bucket = }\n{messages = }\n{query = }")

  formatted: list[dict[str, str | list[dict[str, str | dict[str, str]]]]] = [
    {
      'type': 'text',
      'text': f"Your role here is to answer the user's original query in the most relevant way possible. The original query is: '{query}'. The relevant information needed to answer this query are provided in the docs after this message. If you do not have enough information in the provided docs, then tell the user that you were not able to find enough relevant information to answer to his query. The docs: "
    }
  ]

  if not retrieved:
    logger.debug(f"Unable to generate retrieval information in node 'formatter'")
    return {'messages': {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": """Unfortunately, the retrieved step failed because of some outside reasons. Inform the user and ask if he would like you to try again or respond without retrieval.""".strip()
        }
      ]
    }}

  for doc in retrieved:
    doc_id = doc.id
    doc_type = doc.metadata.get("fileType")
    logger.debug(f"Working with doc {doc_id} of type {doc_type}")

    if not doc_type or not doc_id:
      return {
        "messages": {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"""Unfortunately, there was a problem during the formatting of the retrieved documents. The retrived docs did not have {'doc_id' if not doc_id else 'doc_type'}. Please inform the user and ask if he would like you to try again or respond without retrieval.""".strip()
            }
          ]
        }
      }

    doc_type = doc_type.lower().lstrip(".")

    if doc_type in ['jpeg', 'png', 'jpg', 'pdf']:
      img_url = bucket.generate_presigned_url(doc_id)
      formatted_message_content = asdict(
          Content(
            type='image_url',
            image_url={'url': img_url}
          )
        )
      if 'text' in formatted_message_content:
        del(formatted_message_content['text'])
      formatted.append(formatted_message_content)

    else:
      formatted_message_content = asdict(
          Content(
            type='text',
            text=doc.page_content
          )
        )
      if 'image_url' in formatted_message_content:
        del(formatted_message_content['image_url'])
      formatted.append(formatted_message_content)

  logger.debug(f"Augmented message content generated successfully: {formatted}")

  return {
    "messages": {"role": "user", "content": formatted}
  }


builder.add_node("classify", classify_input)
builder.add_node("route", router)
builder.add_node("chatbot", chatbot)
builder.add_node("retrieve", retrieve)
builder.add_node("format", formatter)
builder.add_edge(START, "classify")
builder.add_edge("classify", "route")
builder.add_conditional_edges(
    "route",
    lambda state: state.get("is_retrieval_required"),
    {"True": "retrieve", "False": "chatbot"}
)
builder.add_edge("retrieve", "format")
builder.add_edge("format", "chatbot")

graph = builder.compile()


def run_chatbot(query: str, retriever: Retriever, vlm: VLM, img_handler: ImgHandler, bucket: BucketService):
  state = {
    "messages": [],
    "is_retrieve_required": None,
    'retriever': retriever,
    'vlm': vlm,
    'img_handler': img_handler,
    'bucket': bucket,
    'retrieved': None,
    'query': query
    }

  state['messages'] = state.get("messages", []) + [
    {"role": "user", "content": state['query']}
  ]

  state = graph.invoke(state)

  if state.get("messages") and len(state['messages']) > 0:
    last_message = state['messages'][-1]

    logger.debug(f"Last message content: {last_message.content}")

    return last_message.content

if __name__ == "__main__":
  pass
