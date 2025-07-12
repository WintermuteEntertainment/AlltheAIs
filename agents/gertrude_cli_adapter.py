# agents/gertrude_cli_adapter.py
import time
import asyncio

from agents.gertrude_agent import Gertrude


class GertrudeCLIWrapper:
    def __init__(self, name="gertrude", bus=None, config=None, matrix=None, memory=None):
        self.name = name
        self.journal = []

        self.agent = Gertrude(name, bus, config, matrix, memory)

    # async def publish(self, event_type, message, target="all"):
    #     if event_type != "text_input":
    #         return f"[Gertrude Error] Unsupported event_type: {event_type}"
    #     await self.agent.respond_to_text(message)
    #     self.journal.append({"type": event_type, "data": message})

    async def receive(self, event):
        # Delegate to the actual agent's async receive method
        await self.agent.receive(event)

    def metadata(self):
        return self.agent.metadata()


