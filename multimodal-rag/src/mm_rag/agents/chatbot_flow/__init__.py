from typing import TYPE_CHECKING, Annotated, Literal, Any

from mm_rag.pipelines.retrievers import Retriever
from mm_rag.models.s3bucket import BucketService
from mm_rag.agents.vlm import VLM

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

from langchain_core.documents import Document

from mm_rag.logging_service.log_config import create_logger

from . import input_classifier, formatter, chatbot, retriever, router


logger= create_logger(__name__)


class State(TypedDict):
  messages: Annotated[list, add_messages]
  is_retrieval_required: bool | None
  retriever: Retriever
  vlm: VLM
  retrieved: list[Document] | None
  bucket: BucketService
  query: str

logger.debug(f"Created new State Schema: {State}")

builder = StateGraph(State)
logger.debug(f"Created graph builder: {builder}")


builder.add_node("classify", input_classifier.classify_input)
builder.add_node("route", router.router)
builder.add_node("chatbot", chatbot.chatbot)
builder.add_node("retrieve", retriever.retrieve)
builder.add_node("format", formatter.formatter)
builder.add_edge(START, "classify")
builder.add_edge("classify", "route")
builder.add_conditional_edges(
    "route",
    lambda state: state.get("is_retrieval_required"),
    {True: "retrieve", False: "chatbot"}
)
builder.add_edge("retrieve", "format")
builder.add_edge("format", "chatbot")


graph = builder.compile()


def run_chatbot(query: str, retriever: 'Retriever', vlm: 'VLM', bucket: 'BucketService') -> Any | None:
  state = {
    "messages": [],
    "is_retrieve_required": None,
    'retriever': retriever,
    'vlm': vlm,
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
  return None
