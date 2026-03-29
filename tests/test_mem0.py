"""Smoke test for Mem0Adapter."""
from dotenv import load_dotenv
load_dotenv()

from memory.mem0_adapter import Mem0Adapter

mem = Mem0Adapter()
user_id = "user_001"

messages = [
    {"role": "human", "content": "我最近頭很痛，每天早上都這樣。"},
    {"role": "ai", "content": "了解，請問您有在服用任何藥物嗎？"},
    {"role": "human", "content": "有，我在吃氨氯地平治療高血壓，每天早上一顆。"},
    {"role": "ai", "content": "建議您記錄早晨血壓，確認頭痛是否與晨峰高血壓有關。"},
]

print("Saving memories...")
mem.save(user_id, messages)

print("\nRetrieving: '用戶的藥物'")
print(mem.retrieve(user_id, "用戶的藥物") or "(no results)")

print("\nRetrieving: '頭痛症狀'")
print(mem.retrieve(user_id, "頭痛症狀") or "(no results)")
