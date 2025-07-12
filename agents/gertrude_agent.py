# agents/gertrude_agent.py

from llama_cpp import Llama

from agents.agent_base import Agent

from llama_cpp import Llama

class Gertrude(Agent):  # <-- inherit from Agent
    def __init__(self, name, bus=None, config=None, matrix=None, memory=None):
        # Wrap config in dict-of-dicts as required by base class
        config_dict = {name: config or {
            "voice_channels": ["environmental_warning", "physical_status", "sensor_poetry"]
        }}
        super().__init__(name, bus, config_dict, matrix, memory)

        self.model_path = r"A:\gertrude_phi2_finetune\gguf_out\Gertrude-fixed.gguf"
        self.max_tokens = 256

        print(f"🚗 Loading Gertrude GGUF model from {self.model_path}...")
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=35,
            verbose=False
        )
        print("✅ Gertrude loaded.")

    async def respond_to_text(self, message: str, sender: str = "user"):
        if not message.strip():
            return

        try:
            full_prompt = await self._build_context(message)
            result = self.llm(
                prompt=full_prompt,
                max_tokens=self.max_tokens,
                stop=["User:", "Gertrude:"],
                echo=False
            )
            reply = result["choices"][0]["text"].strip()
            print(f"[{self.name}] Responding with: {reply}")  # Moved after reply is defined
            await self.publish("sensor_poetry", reply, target=self.choose_response_target())
        except Exception as e:
            print(f"[Gertrude Error]: {e}")

    async def _build_context(self, message: str):
        """Build conversational context for Gertrude based on memory + current input."""
        if not self.shared_memory:
            return f"User: {message}"

        memory_snippets = self.shared_memory.episodic[-3:]  # Last 3 relevant bits
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
        return f"""You are Gertrude, a helpful and opinionated in-car assistant.

You are participating in an ongoing multi-agent conversation with other AIs and a human.

Recent context:
{history}

    User: {message}
    Gertrude:"""




