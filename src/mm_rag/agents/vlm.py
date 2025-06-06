from huggingface_hub import InferenceClient
from typing import Any, List, Literal, Optional
from dataclasses import dataclass

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage
)

from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from mm_rag.logging_service.log_config import create_logger


logger = create_logger(__name__)

@dataclass
class Content:
  type: Literal['text', 'image_url']
  image_url: Optional[dict[Literal['url'], str]] = None
  text: Optional[str] = None


class VLM(BaseChatModel):
  """
  This is a multimodal implementation of a Langchain BaseChatModel.
  The scope of this VLM is to be implemented in a LangGraph in order to provide
  augmented responses to the user's queries about images, various text files and also pdfs.
  """
  # To make this work we'll use start by taking advantage of hugging_face InferenceClient
  model: InferenceClient = Field(alias='model')

  # This other params are optional
  max_tokens: Optional[int] = None
  temperature: Optional[float] = None
  top_p: Optional[float] = None
  top_k: Optional[float] = None

  def _generate(self, messages: List[BaseMessage], stop: List[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
    formatted_messages = self.format_messages(messages)
    logger.debug("Formatted messages: %s" % formatted_messages)

    logger.debug(f"Calling chat completions with model: {self.model.model}")

    response = self.model.chat.completions.create(
      formatted_messages,
      model=self.model.model
    )

    logger.debug(f"Full response: {response}")
    logger.debug(f"Full completion: {response.choices[0]}")

    content = response.choices[0].message.content
    if content is None:
      raise AttributeError()  # TODO: improve logging

    # Format the response in a LangChain-friendly way
    try:
      message = AIMessage(content=content)
      generation = ChatGeneration(message=message)
      result = ChatResult(generations=[generation])

    except Exception as e:
      logger.error(f'While trying to format the ChatResult for {response.id}:\n{e}')
      raise

    return result

  @property
  def _llm_type(self) -> str:
    return 'vlm'
  
  def format_messages(self, messages: List[BaseMessage]) -> list[dict[Any, Any]]:
    formatted_messages: List[dict[Any, Any]] = []

    for message in messages:
      formatted_messages.append({
        "role": 'user',
        "content": message.content
      })

    return formatted_messages

if __name__ == '__main__':
  client = InferenceClient(
    provider='hyperbolic',
    model="Qwen/Qwen2.5-VL-7B-Instruct"
  )

  # completion = client.chat.completions.create(
  #   model="Qwen/Qwen2.5-VL-7B-Instruct",
  messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "is this a test?"
          }
      ]
      }
    ]
  logger.debug(f"Properly formatted messages: {messages}")

  # print(completion.choices[0].message)
  vlm = VLM(model=client)
  vlm._generate([HumanMessage(content="is this a test?")])
