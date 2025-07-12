# agents/eris_agent.py

import torch
import logging
from agents.agent_base import Agent


class Eris(Agent):
    def __init__(self, name, bus, config, escalation_matrix=None, shared_memory=None, model=None):
        super().__init__(name, bus, config, escalation_matrix, shared_memory)
        self.model = model
        self.last_response = None
        
        # Handle GPU initialization
        if torch.cuda.is_available():
            logging.info("Initializing CUDA resources...")
            try:
                torch.cuda.init()
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
                logging.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB")
            except Exception as e:
                logging.error(f"CUDA initialization failed: {e}")

    # After loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    async def respond_to_text(self, message: str, sender: str = "user"):
        wrapped_prompt = f"{sender} said:\n{message.strip()}\n\nYour response:"
        context = await self._build_context(wrapped_prompt)
        response = self._query_eris_model(context)
        print(f"[{self.name}] Responding with: {response}")  # Moved here, AFTER it's defined
        self.last_response = response
        target = self.choose_response_target()
        await self.publish("data_alert", response, target=target)

    def _query_eris_model(self, prompt):
        if self.model is None:
            return "[Eris Error]: No model loaded."
        try:
            result = self.model.generate_response(prompt)
            return result[0] if isinstance(result, tuple) else result
        except Exception as e:
            return f"[Eris Error]: {str(e)}"

    async def act(self):
        if self.last_response:
            await self.publish("consensus_question", f"Follow-up: {self.last_response}")
            self.last_response = None

    async def vote(self, proposal):
        prompt = f"Evaluate this proposal:\n\n{proposal}"
        response = self._query_eris_model(prompt)
        return "approve" if "approve" in response.lower() else "deny"

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
