"""Interactive chat loop. Usage: python scripts/chat_loop.py [dummy|langmem|mem0]"""
import sys
from pathlib import Path

# ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage

adapter_name = sys.argv[1] if len(sys.argv) > 1 else "dummy"

if adapter_name == "langmem":
    from memory.langmem_adapter import LangMemAdapter
    memory = LangMemAdapter()
elif adapter_name == "mem0":
    from memory.mem0_adapter import Mem0Adapter
    memory = Mem0Adapter()
elif adapter_name == "memsearch":
    from memory.memsearch_adapter import MemsearchAdapter
    memory = MemsearchAdapter()
else:
    from memory.base import DummyMemory
    memory = DummyMemory()

from agent.graph import build_graph

user_id = input(f"User ID (default: dev): ").strip() or "dev"
graph = build_graph(memory=memory, user_id=user_id)

print(f"\nHealth Advisor [{adapter_name}] — user: {user_id}")
print("輸入 quit 離開，reset 清除對話歷史\n")

messages = []
while True:
    try:
        user_input = input("你：").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再見！")
        break

    if not user_input:
        continue
    if user_input.lower() == "quit":
        print("再見！")
        break
    if user_input.lower() == "reset":
        messages = []
        print("(對話歷史已清除)")
        continue

    messages.append(HumanMessage(content=user_input))
    result = graph.invoke({"messages": messages})
    messages = result["messages"]
    reply = messages[-1].content
    print(f"\n顧問：{reply}\n")
