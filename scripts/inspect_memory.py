"""Inspect stored memories for a given user and adapter.

Usage:
    python scripts/inspect_memory.py <adapter> <user_id>
    adapter: langmem | mem0 | memsearch
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

elif adapter_name == "memsearch":
    import asyncio
    from memsearch import MemSearch
    user_dir = Path(f".memsearch_store/{user_id}")
    md_files = sorted(user_dir.glob("*.md")) if user_dir.exists() else []
    if not md_files:
        print("(no memory files found)")
    else:
        for i, f in enumerate(md_files, 1):
            print(f"[{i}] {f.name}")
            print(f.read_text(encoding="utf-8"))
            print()

elif adapter_name == "memgpt":
    import json
    store_file = Path(f".memgpt_store/{user_id}.json")
    if not store_file.exists():
        print("(no core memory found)")
    else:
        blocks = json.loads(store_file.read_text())
        for label, value in blocks.items():
            print(f"[{label}] ({len(value)} chars)")
            print(value)
            print()

elif adapter_name == "amem":
    import json
    store_file = Path(f".amem_store/{user_id}.json")
    if not store_file.exists():
        print("(no memories found)")
    else:
        notes = json.loads(store_file.read_text())
        if not notes:
            print("(no memories found)")
        else:
            for i, note in enumerate(notes, 1):
                print(f"[{i}] {note['content'][:120]}{'...' if len(note['content']) > 120 else ''}")
                print(f"    id      : {note['id']}")
                print(f"    keywords: {', '.join(note.get('keywords', [])[:6])}")
                print(f"    tags    : {', '.join(note.get('tags', [])[:4])}")
                print(f"    context : {note.get('context', '')[:100]}")
                links = note.get('links', [])
                if links:
                    print(f"    links   : {links}")
                print()

else:
    print(f"Unknown adapter: {adapter_name}. Use 'langmem', 'mem0', 'memsearch', 'memgpt', or 'amem'.")
    sys.exit(1)
