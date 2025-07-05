# --- core/embodied_security.py ---

import asyncio
from agents.alex import Alex
from agents.eris import Eris
from agents.gertrude import Gertrude
from tim_interface import TimConsole
from core.council import CoAgencyCouncil
from core.shared_memory import SharedMemory


class EmbodiedVerification:
    def __init__(self):
        self.required_confirmations = {
            "emergency_override": ["physical_presence", "biometric", "voice"],
            "environment_adjust": ["physical_presence"]
        }

    def verify_action(self, action, confirmations):
        required = self.required_confirmations.get(action["type"], [])
        return all(c in confirmations for c in required)
