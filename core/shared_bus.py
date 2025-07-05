#shared_bus.py

import asyncio
from agents.alex import Alex
from agents.eris import Eris
from agents.gertrude import Gertrude
from tim_interface import TimConsole
from core.council import CoAgencyCouncil
from core.shared_memory import SharedMemory

class AsyncSharedBus:
    def __init__(self):
        self.subscribers = []
        self.event_queue = asyncio.Queue()

    def subscribe(self, agent):
        self.subscribers.append(agent)

    async def publish(self, event):
        await self.event_queue.put(event)

    async def event_dispatcher(self):
        while True:
            event = await self.event_queue.get()
            await asyncio.gather(*[
                sub.receive(event)
                for sub in self.subscribers
                if event['target'] in (sub.name, "all")
            ])