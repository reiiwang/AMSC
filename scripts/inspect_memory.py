"""Inspect stored memories for a given user and adapter.

Usage:
    python scripts/inspect_memory.py <adapter> <user_id>
    adapter: langmem | mem0
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

adapter_name = sys.argv[1] if len(sys.argv) > 1 else "langmem"
user_id = sys.argv[2] if len(sys.argv) > 2 else "dev"

print(f"\n{'═'*50}")
print(f"  Adapter : {adapter_name}")
print(f"  User ID : {user_id}")
print(f"{'═'*50}\n")

if adapter_name == "langmem":
    import json
    store_file = Path(f".langmem_store/{user_id}.json")
    if not store_file.exists():
        print("(no memories found)")
    else:
        memories = json.loads(store_file.read_text())
        if not memories:
            print("(no memories found)")
        else:
            for i, (mem_id, content) in enumerate(memories.items(), 1):
                print(f"[{i}] {content}")
                print(f"    id: {mem_id}\n")

elif adapter_name == "mem0":
    from memory.mem0_adapter import Mem0Adapter
    mem = Mem0Adapter()
    results = mem._mem.get_all(user_id=user_id)
    memories = results.get("results", []) if isinstance(results, dict) else results
    if not memories:
        print("(no memories found)")
    else:
        for i, m in enumerate(memories, 1):
            print(f"[{i}] {m['memory']}")
            print(f"    id: {m['id']}  score: {m.get('score', 'N/A')}\n")

else:
    print(f"Unknown adapter: {adapter_name}. Use 'langmem' or 'mem0'.")
    sys.exit(1)
