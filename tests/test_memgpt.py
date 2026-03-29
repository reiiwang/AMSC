"""Smoke test for MemGPTAdapter."""
from dotenv import load_dotenv
load_dotenv()

from memory.memgpt_adapter import MemGPTAdapter

mem = MemGPTAdapter(user_id="user_001")

print("=== Initial core memory ===")
print(mem.retrieve("user_001", ""))

print("\n=== Testing core_memory_append ===")
tools = {t.name: t for t in mem.get_tools()}
result = tools["core_memory_append"].invoke(
    {"label": "human", "content": "用戶名為陳小明，45歲，有高血壓，服用氨氯地平。"}
)
print(result)

print("\n=== Core memory after append ===")
print(mem.retrieve("user_001", ""))

print("\n=== Testing archival_memory_insert ===")
result = tools["archival_memory_insert"].invoke(
    {"content": "2026-03-29：用戶回報早晨頭痛，量血壓 155/95，建議記錄一週血壓數值。"}
)
print(result)

print("\n=== Testing archival_memory_search ===")
result = tools["archival_memory_search"].invoke({"query": "血壓頭痛"})
print(result)
