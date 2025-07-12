#shared_bus.py

import asyncio
import time
import random



class AsyncSharedBus:
    def __init__(self):
        self.subscribers = []
        self.event_queue = asyncio.Queue()

    def subscribe(self, agent):
        self.subscribers.append(agent)

    async def publish(self, event):
        await self.event_queue.put(event)

    # In AsyncSharedBus.event_dispatcher
    
    async def event_dispatcher(self):
        while True:
            try:
                event = await self.event_queue.get()
                print(f"[Bus] Dispatching event: {event}")

                if event["type"] in ["emotional_expression", "data_alert", "sensor_poetry"]:
                    print(f"[{event['source']} ➜ {event['target']}] {event['type']}: {event['data']}")

                tasks = []
                for sub in self.subscribers:
                    if event['target'] in (sub.name, "all"):
                        task = asyncio.create_task(sub.receive(event))
                        tasks.append(task)

                await asyncio.gather(*tasks)

            except Exception as e:
                print(f"[Bus Error] {type(e).__name__}: {e}")
