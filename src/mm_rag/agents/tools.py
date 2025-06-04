from pydantic import BaseModel, Field

from langchain.tools import Tool
from mm_rag.pipelines.retrievers import Retriever

import json


class RetrieverInput(BaseModel):
  query: str = Field(description="This tool retrieves relevant information from a document database." \
  "Use it whenever the user's question needs additional context or specific facts that are not in the current chat history.")


def make_retrieve_tool(retriever: Retriever) -> Tool:
  def _retrieve(query: str) -> str:
    docs = retriever.retrieve(query)
    serialized_docs = retriever.from_docs_to_string(docs)

    return serialized_docs

  return Tool(
    name="retrieve",
    description="Use this tool to gather relevant information from the vector store." \
                "This tool is the only way of gathering additional information useful to answer the user's query."\
                "Use it whenever you are unsure about how to answer the user's question.",
    func=_retrieve,
    args_schema=RetrieverInput
  )
