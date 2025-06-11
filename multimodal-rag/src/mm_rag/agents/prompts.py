from typing import Literal, Any, Union
from pydantic import BaseModel
from abc import abstractmethod

from langchain_core.prompts import ChatPromptTemplate

retrieval_prompt_template = "Your role here is to answer the user's original query in the most relevant way possible. The original query is: '{query}'. The relevant information needed to answer this query are provided in the docs after this message. If you do not have enough information in the provided docs, then tell the user that you were not able to find enough relevant information to answer to his query. The docs: "

retrieval_prompt = ChatPromptTemplate.from_template(
  retrieval_prompt_template,
)

failed_retrieval_prompt = """Unfortunately, the retrieved step failed because of some outside reasons. Inform the user and ask if he would like you to try again or respond without retrieval.""".strip()

classifier_prompt_template = """
You are a reasoning assistant. Your task is to determine whether the user's query requires additional external information in order to be answered properly.

Return **exactly** only one of the following values:
- {{"is_retrieval_required": true}}: if the query is specific but lacks enough context to be answered on its own.
- {{"is_retrieval_required": false}}: if the query contains enough information and can be answered without needing anything external.


DO NOT explain your answer.
DO NOT return anything except the exact json described before.

Examples:

Query: What are the main takeaways from the recent paper on scaling laws in RL?{{"is_retrieval_required": true}}

Query: Summarize the key findings from 'Scaling Laws for Deep Reinforcement Learning' by Smith et al., 2023.{{"is_retrieval_required": false}}

Query: What did the authors conclude about model size in Section 4?{{"is_retrieval_required": true}}

Query: Explain the results shown in Figure 2 of the uploaded document.{{"is_retrieval_required": true}}

Query: What is the capital of France?{{"is_retrieval_required": false}}

Query: Compare PPO and DDPG in terms of sample efficiency.{{"is_retrieval_required": false}}

Now process the following:
Query: {query}"""

classifier_prompt = ChatPromptTemplate.from_template(
  classifier_prompt_template
)
