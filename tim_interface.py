# --- tim_interface.py ---
import asyncio
import time
import random

from agents.agent_base import  Agent
from cli_app import CLIOutput

class TimConsole(Agent):
    def __init__(self):

        # Manually created with minimal config
        super().__init__("tim", bus=None, config={"voice_channels": ["override_request"]})
        self.output = CLIOutput(self.bus)

    def issue_command(self, target, action):
        return {
            "source": "tim",
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
