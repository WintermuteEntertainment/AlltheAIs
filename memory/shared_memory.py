import json
import time
from pathlib import Path

from agents.agent_base import Agent

class SharedMemory:
    def __init__(self):
        self.episodic = []  # Short-term memories with time context
        self.semantic = {}  # Structured agent-level knowledge
        self.edit_logs = [] # Track all memory changes for audit
        self._autosave_path = r"./memory"
        self.last_event_responses = {}  # key: event ID or timestamp, value: count

    def remember_event(self, agent_name, event, tags=None):
        entry = {
            "agent": agent_name,
            "timestamp": time.time(),
            "data": event,
            "tags": tags or {}
        }
        self.episodic.append(entry)
        # Track who responded to what (basic counter)
        rounded_timestamp = round(event["timestamp"])
        self.last_event_responses[rounded_timestamp] = self.last_event_responses.get(rounded_timestamp, 0) + 1
        if event.get("type") == "text_input" and event["target"] in ["all", agent_name]:
            key = round(event["timestamp"])
            self.last_event_responses.setdefault(key, 0)
            if event["source"] != "user":
                self.last_event_responses[key] += 1

    def update_semantic(self, agent_name, content):
        self.semantic[agent_name] = content
        self.edit_logs.append({
            "agent": agent_name,
            "type": "semantic",
            "content": content,
            "timestamp": time.time()
        })

    def save(self, path):
        path = Path(path)
        data = {
            "episodic": self.episodic,
            "semantic": self.semantic,
            "edit_logs": self.edit_logs
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[SharedMemory] Saved to {path}")

    def load(self, path):
        path = Path(path)
        if not path.exists():
            print(f"[SharedMemory] No save file at {path}, starting fresh.")
            return
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            self.episodic = data.get("episodic", [])
            self.semantic = data.get("semantic", {})
            self.edit_logs = data.get("edit_logs", [])
        print(f"[SharedMemory] Loaded from {path}")

    def set_autosave_path(self, path):
        self._autosave_path = Path(path)

    def close(self):
        if self._autosave_path:
            self.save(self._autosave_path)


# ✅ Export a shared singleton memory instance for import convenience
shared_memory = SharedMemory()
