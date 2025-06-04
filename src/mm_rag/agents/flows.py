import json

from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

from langchain_huggingface import ChatHuggingFace
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from mm_rag.pipelines.retrievers import Retriever
from mm_rag.logging_service.log_config import create_logger
from mm_rag.processing.handlers import ImgHandler
from mm_rag.models.s3bucket import BucketService


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
  vlm: ChatHuggingFace
  retrieved: list[Document] | None
  img_handler: ImgHandler
  bucket: BucketService
  augmented_query: dict[str, list[dict[str, str]] | str] | None

logger.debug(f"Created new State Schema: {State}")

builder = StateGraph(State)
logger.debug(f"Created graph builder: {builder}")

def classify_input(state: State):
  last_message = state['messages'][-1]
  vlm = state['vlm']
  logger.debug(f"Classifying input: {last_message}")

  result = vlm.invoke(
      [
          {
              "role": "system",
              "content": """determine if extra external information is needed to
              answer the user's query. Return:
              True: if extra info is required,
              False: if extra info is NOT required
              DO NOT return anything else except 'True' or 'False' as needed.
              """
          },
          {
              "role": "user",
              "content": last_message.content
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

  return {"is_retrieval_required": parsed.is_retrieval_required}


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

  logger.debug(f'Running inference on the model')

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
  logger.debug(f"Running retrieval on input: {last_message}")

  retrieved = retriever.invoke(last_message.content)
  logger.debug(f"Found in the VectorStore: {retrieved}")

  return {"retrieved": retrieved}


def formatter(state: State):
  retrieved = state['retrieved']
  handler = state['img_handler']
  bucket = state['bucket']
  messages = state.get("messages", [])
  logger.debug(f"Retrieved from the state: {retrieved = }\n{handler = }\n{bucket = }\n{messages = }")

  final_message: HumanMessage = HumanMessage(content=[])

  if not retrieved:
    raise  #TODO: improve error handling

  retrieved = retrieved[:1]

  for doc in retrieved:
    doc_id = doc.id
    doc_type = doc.metadata.get("fileType")
    logger.debug(f"Working with doc {doc_id} of type {doc_type}")

    if not doc_type:
      raise  #TODO: improve error handling

    if not doc_id:
      raise  #TODO: improve error handling

    doc_type = doc_type.lower().lstrip(".")

    if doc_type in ['jpeg', 'png', 'jpg', 'pdf']:
      downloaded_img = handler.download_img_from_bucket(doc_id, bucket)
      encoded = 'data:image;base64,' + handler.base64_encode(downloaded_img)
      final_message.content.append({"type": "image", "image": encoded})  # type: ignore[call-args]
      logger.debug(f"Appended img file to the doc_contents: {encoded[:20]}")

    else:
      final_message.content.append({"type": "text", "text": doc.page_content})  # type: ignore[call-args]
      logger.debug(f"Appended text file to the doc_contents: {doc.page_content}")

  logger.debug(f"Augmented message generated successfully.")

  return {"messages": final_message}


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


def run_chatbot(retriever: Retriever, vlm: ChatHuggingFace, img_handler: ImgHandler, bucket: BucketService):
  state = {
    "messages": [],
    "is_retrieve_required": None,
    'retriever': retriever,
    'vlm': vlm,
    'img_handler': img_handler,
    'bucket': bucket,
    'retrieved': None,
    'augmented_query': None
    }

  while True:
    user_input = input("You: ")

    if user_input == "quit":
      print('Au revoir')
      break

    state['messages'] = state.get("messages", []) + [
        {"role": "user", "content": user_input}
    ]

    state = graph.invoke(state)

    if state.get("messages") and len(state['messages']) > 0:
      last_message = state['messages'][-1]
      print(f"Jarvis: {last_message}\n")

if __name__ == "__main__":
  pass