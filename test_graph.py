"""Quick smoke test for the agent graph."""
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from agent.graph import build_graph

graph = build_graph(user_id="test_user")

state = {"messages": [HumanMessage(content="我最近頭很痛，早上起床的時候特別嚴重。")]}
result = graph.invoke(state)
print(result["messages"][-1].content)
