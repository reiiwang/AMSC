"""Smoke test for AMemAdapter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.graph import build_graph
from memory.amem_adapter import AMemAdapter

USER_ID = "test_amem_user"

messages = [
    {
        "role": "human",
        "content": "我最近血壓偏高，大概 140/90，有點擔心。",
    },
    {
        "role": "ai",
        "content": "您的血壓確實偏高，建議減少鹽分攝取並適度運動。有在服用任何藥物嗎？",
    },
]

print("=== A-MEM Smoke Test ===")

memory = AMemAdapter()

print("\n[1] Testing save()...")
memory.save(USER_ID, messages)
system = memory._get_system(USER_ID)
print(f"    memories stored: {len(system.memories)}")
for note_id, note in system.memories.items():
    print(f"    - [{note_id[:8]}] keywords: {note.keywords[:3]}")
    print(f"      context: {note.context}")

print("\n[2] Testing retrieve()...")
result = memory.retrieve(USER_ID, "血壓用藥建議")
print(f"    retrieved:\n{result}")

print("\n[3] Testing via LangGraph...")
graph = build_graph(memory, USER_ID)
response = graph.invoke({
    "messages": [{"role": "human", "content": "我的血壓狀況有改善嗎？"}]
})
final = response["messages"][-1]
print(f"    agent response: {final.content[:200]}")

print("\n✅ A-MEM smoke test passed.")
