# --- core/agent_base.py ---

import asyncio
import time
import random
import asyncio
from agents.alex import Alex
from agents.eris import Eris
from agents.gertrude import Gertrude
from tim_interface import TimConsole
from core.council import CoAgencyCouncil
from core.shared_memory import SharedMemory
from core.escalation_graph import EscalationMatrix
from core.embodied_security import EmbodiedVerification


class VoicePermissionError(Exception):
    pass

class Agent:
    def __init__(self, name, bus, config, escalation_matrix=None, shared_memory=None):
        self.name = name
        self.bus = bus
        self.config = config.get(name, {})
        self.voice_channels = self.config.get("voice_channels", [])
        self.escalation_matrix = escalation_matrix
        self.shared_memory = shared_memory
        self.verifier = EmbodiedVerification()
        self.journal = []
        if not self.voice_channels:
            raise VoicePermissionError(f"{name} created without voice channels - violates co-agency principle")

    async def receive(self, event):
        if event["target"] == "all":
            self.log_event(event)
        # To be implemented by each agent

    async def act(self):
        pass

    async def vote(self, proposal):
        return "approve" if random.random() > 0.2 else "deny"

    async def publish(self, event_type, data, target="all"):
        if event_type not in self.voice_channels:
            raise VoicePermissionError(f"{self.name} lacks voice channel for {event_type}")
        event = {
            "source": self.name,
            "target": target,
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        }
        if target == "all":
            self.log_event(event)
        await self.bus.publish(event)

    def log_event(self, event):
        self.journal.append(event)

    async def escalate(self, reason, to="Tim", concern_type=None):
        if self.escalation_matrix and concern_type:
            paths = self.escalation_matrix.get_escalation_paths(self.name, concern_type)
            for (target, _, weight) in paths:
                if weight >= 0.6:
                    await self.publish("escalation", {"reason": reason, "weight": weight}, target=target)
                    return
        await self.publish("escalation", {"reason": reason}, target=to)

    def check_permission(self, action):
        if action in self.config.get("cannot_act_on", []):
            raise PermissionError(f"{self.name} cannot act on '{action}'")
