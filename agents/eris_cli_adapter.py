# agents/eris_cli_adapter.py

from gtts import tokenizer
from agents.eris_agent import Eris
import torch
import logging


logger = logging.getLogger(__name__)

class ErisCLIWrapper:
    def __init__(self, name="eris", bus=None, config=None, matrix=None, memory=None):
        self.name = name
        self.journal = []
        model = None
        
        try:
            from agents.Eris import load_model, ERISEngine
            
            logger.info("Loading Eris model...")
            tokenizer, base_model = load_model()
            model = ERISEngine(tokenizer, base_model)
            logger.info("Model loaded successfully")
            
            if torch.cuda.is_available():
                logger.info("Moving model to GPU...")
                model = model.to(torch.device("cuda"))  # Use custom to() method
                logger.info("Model moved to GPU")
            else:
                logger.info("Using CPU for Eris model")
        except Exception as e:
            logger.exception(f"Error loading Eris model: {e}")
        
         # Ensure config is properly set
        self.config = config or {
            "voice_channels": ["hypothesis_proposal", "data_alert", "consensus_question", "sensor_poetry", "environmental_warning", "driver_questions"]
        }

        self.agent = Eris(name, bus, config, matrix, memory, model=model)

        try:
            self.agent = Eris(name, bus, config, matrix, memory, model=model)
            logger.info(f"Eris agent initialized. Model available: {model is not None}")
        except Exception as e:
            logger.exception(f"Error creating Eris agent: {e}")
            raise
        
        # # Provide default voice_channels if not passed in
        # config = config or {
        #     "voice_channels": ["hypothesis_proposal", "data_alert", "consensus_question"]
        # }

        print(f"[DEBUG] Model in Eris agent? {self.agent.model is not None}")  # ✅ This should now say True
        self.name = name
        self.journal = []

    # async def publish(self, event_type, message, target="all"):
    #     if event_type != "text_input":
    #         print(f"[Eris Warning] Unsupported event_type: {event_type}")
    #         return

    #     if not hasattr(self.agent, "model") or self.agent.model is None:
    #         print("[Eris Error] Model not available.")
    #         return

    #     try:
    #         response = self.agent._query_eris_model(message)
    #     except Exception as e:
    #         response = f"[Eris Error]: {str(e)}"

    #     self.journal.append({"type": event_type, "data": message})
    #     print(f"[Eris] {response}")
    #     return response

    async def receive(self, event):
        # Delegate to the actual agent's async receive method
        await self.agent.receive(event)

    def metadata(self):
        return {
            "name": self.name,
            "type": "agent",
            "provider": "local",
            "device": "cuda" if hasattr(self.agent.model, "device") and self.agent.model.device == "cuda" else "cpu"
        }
