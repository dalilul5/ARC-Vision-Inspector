import json
import os
from typing import List, Dict, Any

def load_arc_task(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_arc_tasks_from_dir(directory: str) -> List[Dict[str, Any]]:
    tasks = []
    if not os.path.exists(directory):
        return tasks

    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".json"):
            full_path = os.path.join(directory, fname)
            task = load_arc_task(full_path)
            task["_task_name"] = fname
            tasks.append(task)
    return tasks

def iter_all_pairs(task: Dict[str, Any]):
    task_name = task.get("_task_name", "unknown_task")

    for split in ["train", "test"]:
        for idx, pair in enumerate(task.get(split, [])):
            yield {
                "task_name": task_name,
                "split": split,
                "index": idx,
                "input": pair["input"],
                "output": pair["output"]
            }
