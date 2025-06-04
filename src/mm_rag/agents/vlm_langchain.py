from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.tools import Tool
from langchain_core.messages import AnyMessage, BaseMessage, ChatMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain.agents import initialize_agent, AgentType


from .prompts import system_prompt


class VLM:

  convo: list[AnyMessage] = [
    SystemMessage(system_prompt)
  ]
  _chat_model: ChatHuggingFace | None = None

  def __init__(
      self,
      tools: list[Tool] | None = None,
      model_id: str = "Qwen/Qwen2.5-VL-32B-Instruct"
  ) -> None:
    self.model_id = model_id
    self.tools = tools

  @property
  def chat_model(self) -> ChatHuggingFace:
    if not self._chat_model:
      endpoint = HuggingFaceEndpoint(  # type: ignore[call-args]
      repo_id=self.model_id,
      verbose=True
    )
      return ChatHuggingFace(llm=endpoint, verbose=True)

    return self._chat_model

  @property
  def agent(self):
    if not self.tools:
      raise AttributeError(
        f"In order to create the agent, please provide tools by using `.add_tools`"
      )

    _agent = initialize_agent(
      self.tools,
      self.chat_model,
      agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
      verbose=True
    )
    return _agent

  def chat(self, messages: list) -> None:
    print(self.agent.invoke({"input": messages}))

  def add_tools(self, tools: list[Tool]) -> None:
    self.tools = tools