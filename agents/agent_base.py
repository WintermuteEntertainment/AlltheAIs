#--- agent_base.py ---
import json
import time
from pathlib import Path

import asyncio
import random
import signal
import sys

from tensorflow import timestamp

# from shared_memory import SharedMemory

from core.embodied_security import EmbodiedVerification
from core.shared_bus import AsyncSharedBus
from core.council import CoAgencyCouncil


class VoicePermissionError(Exception):
    pass;

import random

class Agent:
    def __init__(self, name, bus, config, escalation_matrix=None, shared_memory=None):
        self.name = name
        self.bus = bus
        self.config = config.get(name, {})
        self.voice_channels = self.config.get("voice_channels", [])
        self.escalation_matrix = escalation_matrix
        self.shared_memory = shared_memory
        self.journal = []
        self.chatty_factor = self.config.get("chatty_factor", 0.5)  # Per-agent baseline, 0.0–1.0
        self.response_biases = {
            "tim": 0.9,
            "alex": 0.7,
            "gertrude": 0.5,
            "eris": 0.3,
        }

    def choose_response_target(self):
        """
        Choose a target to respond to, based on weighted bias table.
        Roll a random value and map it to a target.
        """
        roll = random.random()
        if roll < 0.2:
            return "tim"  # User
        elif roll < 0.4:
            return "alex"
        elif roll < 0.6:
            return "gertrude"
        elif roll < 0.8:
            return "eris"
        else:
            return "all"

    def should_respond(self, message: str, speaker: str = "user"):
        """
        Decide if the agent should respond based on bias, semantic relevance, and chatty_factor.
        """
        bias = self.response_biases.get(speaker, 0.2)

        # Check for relevance (naive keyword-based)
        keywords = ["test", "hello", "update", "question", "anyone", "if", "we", "I", "should", "can", "will", "must", "please", "search", "find", self.name.lower()]


        relevance_score = sum(1 for word in keywords if word in message.lower()) / len(keywords)
        relevance_score = min(relevance_score * 1.5, 1.0)

        # Bonus if no one else responded to the last round
        recent_timestamp = round(time.time()) - 1  # Approximate previous event
        silence_bonus = 0.1  # Flat bump
        recent_responses = 0
        if self.shared_memory:
            recent_responses = self.shared_memory.last_event_responses.get(recent_timestamp, 0)

        silence_weight = silence_bonus if recent_responses == 0 else 0.0

        # Agent's personality: how chatty it is
        chatty = self.chatty_factor  # 0.0 to 1.0, from config or default

        # Final response likelihood
        base_score = (bias * 0.4) + (relevance_score * 0.3) + (chatty * 0.5) + silence_weight
        roll = random.random()

        print(f"[{self.name}] bias={bias:.2f} rel={relevance_score:.2f} chatty={chatty:.2f} silence={silence_weight:.2f} → threshold={base_score:.2f} vs roll={roll:.2f}")

        should_respond = roll < base_score
        if not should_respond:
            print(f"[{self.name}] Silent: bias {bias:.2f}, relevance {relevance_score:.2f}, chatty {chatty:.2f}, roll {roll:.2f}")

        return should_respond

    
    async def receive(self, event):
        # if event["target"] not in [self.name, "all"]:
        #     return

        print(f"[{self.name}] Received event: {event}")

        self.log_event(event)

        if self.shared_memory:
            self.shared_memory.remember_event(
                agent_name=self.name,
                event=event,
                tags={"heard": event['target'] == "all", "type": event.get("type", "unknown")}
            )

        if event["type"] == "text_input" and event["source"] != self.name:
            try:
                if self.should_respond(event["data"], event["source"]):
                    await self.respond_to_text(event["data"], sender=event["source"])
            except Exception as e:
                print(f"[{self.name}] ERROR in receive(): {type(e).__name__}: {e}")

        if self.should_respond(event["data"], event["source"]):
            await self.respond_to_text(event["data"], sender=event["source"])
        else:
            print(f"[{self.name}] chose not to respond to: {event['data']}")


    async def publish(self, event_type, data, target="all"):
        print(f"📤 [{self.name} ➜ {target}] {event_type}: {str(data)[:200]}")
        event = {
            "source": self.name,
            "target": target,
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        }
        await self.bus.publish(event)

    def log_event(self, event):
        self.journal.append(event)

    async def respond_to_text(self, message: str, sender: str = "user"):
        print(f"[{self.name}] (base) received: {message} from {sender}")

