# --- tim_interface.py ---

import random
import time
from core.agent_base import Agent
import asyncio
from agents.alex import Alex
from agents.eris import Eris
from agents.gertrude import Gertrude
from tim_interface import TimConsole
from core.council import CoAgencyCouncil
from core.shared_memory import SharedMemory
from core.escalation_graph import EscalationMatrix
from core.embodied_security import EmbodiedVerification

class TimConsole(Agent):
    def __init__(self):
        # Manually created with minimal config
        super().__init__("tim", bus=None, config={"voice_channels": ["override_request"]})

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
