# --- core/embodied_security.py ---

import asyncio
import time
import random
# from council import CoAgencyCouncil
# from escalation_graph import EscalationMatrix
# from embodied_security import EmbodiedVerification


class EmbodiedVerification:
    def __init__(self):
        self.required_confirmations = {
            "emergency_override": ["physical_presence", "biometric", "voice"],
            "environment_adjust": ["physical_presence"]
        }

    def verify_action(self, action, confirmations):
        required = self.required_confirmations.get(action["type"], [])
        return all(c in confirmations for c in required)
