from typing import Annotated
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages


class AgentState(MessagesState):
    memory_context: str  # injected from memory layer before LLM call
