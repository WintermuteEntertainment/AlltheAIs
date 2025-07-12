# cli/tim_console_cli.py
import time
from core.shared_bus import AsyncSharedBus

self.memory = global_shared_memory

class TimConsole:
    def __init__(self):
        self.name = "tim"
        self.bus = None
        self.voice_channels = ["override_request"]

    def issue_command(self, target, action):
        return {
            "source": self.name,
            "target": target,
            "type": "embodied_action",
            "data": {
                "type": "emergency_override",
                "command": action,
                "biometric_signature": self.get_biometric_hash()
            },
            "timestamp": time.time()
        }

    def get_biometric_hash(self):
        return "stubbed_hash_1234"

    def receive(self, event):
        print(f"[TimConsole] Received event: {event}")

