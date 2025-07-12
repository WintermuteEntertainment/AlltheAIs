# --- alex_cli_adapter.py
import asyncio
import os
import openai
from openai import AsyncOpenAI
import logging

from agents.alex_agent import Alex  # Your actual Agent subclass

# from core.shared_bus import AsyncSharedBus
# from core.escalation_graph import EscalationMatrix
# from memory.shared_memory import SharedMemory

# Use environment variable or hardcode your API key here
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "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"))

class AlexCLIWrapper:
    def __init__(self, name="alex", bus=None, config=None, matrix=None, memory=None):
        self.name = name
        self.journal = []

        bus = bus or self.bus()
        matrix = matrix or self.matrix()
        memory = memory or self.memory()

        config_dict = {name: config or {
            "voice_channels": ["emotional_expression", "mediation_request", "environment_poetry", "sensor_poetry"],
            "cannot_act_on": ["physical_env"]
        }}

        self.agent = Alex(
            name, 
            bus, 
            config,  # Use the passed config directly
            matrix, 
            memory
        )

        logging.info(f"Alex agent initialized with config: {config}")

    # async def publish(self, event_type, message, target="all"):
    #     if event_type != "text_input":
    #         return f"[Alex Error] Unsupported event_type: {event_type}"

    #     await self.agent.respond_to_text(message)
    #     self.journal.append({"type": event_type, "data": message})

    async def receive(self, event):
        # Delegate to the actual agent's async receive method
        await self.agent.receive(event)

    def metadata(self):
        return {
            "name": self.name,
            "type": "agent",
            "provider": "OpenAI",
            "model": "gpt-4.5-preview"
        }
