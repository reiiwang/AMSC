from langchain_core.messages import SystemMessage, trim_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.tools import TOOLS
from memory.base import BaseMemory, DummyMemory

HISTORY_LIMIT = 5
MAX_TOOL_ITERATIONS = 3

SYSTEM_PROMPT = """你是一位專業的健康顧問，能夠根據用戶的健康狀況提供個人化建議。
請以繁體中文回答，語氣親切專業。

當你需要查詢藥物、疾病或檢驗相關資訊時，請使用 search_knowledge_base 工具。
當你需要了解用戶的健康背景時，請使用 get_user_health_profile 工具。"""


def build_graph(memory: BaseMemory = None, user_id: str = "default") -> StateGraph:
    if memory is None:
        memory = DummyMemory()

    llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(TOOLS)
    tool_node = ToolNode(TOOLS)

    def retrieve_memory(state: AgentState) -> dict:
        last_user_msg = next(
            (m.content for m in reversed(state["messages"]) if m.type == "human"),
            "",
        )
        context = memory.retrieve(user_id, last_user_msg)
        return {"memory_context": context}

    def call_agent(state: AgentState) -> dict:
        system_content = SYSTEM_PROMPT
        if state.get("memory_context"):
            system_content += f"\n\n## 用戶長期記憶\n{state['memory_context']}"

        recent_messages = trim_messages(
            state["messages"],
            max_tokens=HISTORY_LIMIT,
            token_counter=len,
            strategy="last",
            include_system=False,
        )

        response = llm.invoke([SystemMessage(content=system_content)] + recent_messages)
        return {"messages": [response]}

    def save_memory(state: AgentState) -> dict:
        serialized = [
            {"role": m.type, "content": m.content} for m in state["messages"]
        ]
        memory.save(user_id, serialized)
        return {}

    def should_use_tools(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        # count how many tool call rounds have happened
        tool_call_count = sum(
            1 for m in state["messages"] if m.type == "tool"
        )
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            if tool_call_count < MAX_TOOL_ITERATIONS:
                return "tools"
        return "save_memory"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve_memory", retrieve_memory)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", tool_node)
    graph.add_node("save_memory", save_memory)

    graph.add_edge(START, "retrieve_memory")
    graph.add_edge("retrieve_memory", "agent")
    graph.add_conditional_edges("agent", should_use_tools, ["tools", "save_memory"])
    graph.add_edge("tools", "agent")  # loop back after tool execution
    graph.add_edge("save_memory", END)

    return graph.compile()
