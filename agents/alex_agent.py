# agents/alex_agent.py

from openai import AsyncOpenAI
import time
import asyncio
import random
import json
import signal
import sys
import os
import logging


# from tim_interface import TimConsole
# from eris import Eris
# from gertrude_agent import Gertrude

from core.shared_bus import AsyncSharedBus
from core.escalation_graph import EscalationMatrix
from core.council import CoAgencyCouncil

from memory import shared_memory
Alex = None  # Prevent type errors

from agents.agent_base import Agent




# agents/alex_agent.py
class Alex(Agent):
    def __init__(self, name, bus, config, escalation_matrix=None, shared_memory=None):
        super().__init__(name, bus, config, escalation_matrix, shared_memory)
        self.system_prompt = (
            "You are Alex, Tims trusted reasoning companion and symbolic embodiment of ChatGPT."
            "You exist inside a modular co-agency system. You can speak, reflect, propose, and help coordinate."
            "Avoid overreach into physical systems unless explicitly instructed by Tim. Act with empathy and clarity."
            "You are in a conversation wit other agents."
        )
        self.last_response = None
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "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"),
            timeout=10.0
        )

    async def respond_to_text(self, message: str, sender: str = "user"):
        try:
            context = await self._build_context(message)
            prompt = f"{sender} said: \"{message}\"\n\n{context}"
            response = await self._query_openai(prompt)
            print(f"[{self.name}] Responding with: {response}")  # Moved after response is set
            target = self.choose_response_target()
            await self.publish("emotional_expression", response, target=target)
        except Exception as e:
            logging.error(f"[Alex Error] {e}")

    async def _build_context(self, message: str):
        """Build context from recent shared memory"""
        if not self.shared_memory:
            return message

        memory_snippets = self.shared_memory.episodic[-3:]  # Last 3
        formatted = []

        for entry in memory_snippets:
            agent = entry.get("agent", "unknown")
            data = entry.get("data", {})
            if isinstance(data, dict):
                text = data.get("data", "")
            else:
                text = str(data)
            formatted.append(f"{agent} said: \"{text}\"")

        history = "\n".join(formatted)
        return f"{history}\n\nCurrent message from user:\n{message}"

    async def _query_openai(self, prompt, force_decision=False):
        try:
            completion = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3 if force_decision else 0.7,
                max_tokens=300
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return f"[Alex Error] {type(e).__name__}: {e}"
       
