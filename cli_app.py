#cli_app.py
import torch
import asyncio
import json
import signal
import sys
import time
import logging
from agents.agent_base import Agent
from core.shared_bus import AsyncSharedBus
from core.escalation_graph import EscalationMatrix
from core.council import CoAgencyCouncil
from memory.shared_memory import shared_memory
from agents.alex_cli_adapter import AlexCLIWrapper
from agents.eris_cli_adapter import ErisCLIWrapper
from agents.gertrude_cli_adapter import GertrudeCLIWrapper


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aletheia_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AletheiaCLI")

class CLIOutput:
    def __init__(self, bus):  # Add bus parameter
        self.bus = bus
        self.name = "cli_output"
        bus.subscribe(self)

    async def receive(self, event):
        try:
            # Add 'emotional_expression' to the allowed types
            if event["type"] in ["emotional_expression", "data_alert", "sensor_poetry"]:
                print(f"[{event['source']}] {event['data']}")
        except KeyError:
            logger.error("Malformed event received in CLIOutput")
        except Exception as e:
            logger.exception("Error in CLIOutput.receive")

class TimConsoleCLI:
    def __init__(self):
        logger.info("Initializing Aletheia CLI...")
        self.bus = AsyncSharedBus()
        self.memory = shared_memory
        self.matrix = EscalationMatrix()
        self.memory.set_autosave_path("memory_state.json")
        logger.info("Core components initialized")

        self.config = {
            "alex": {"voice_channels": ["emotional_expression", "mediation_request", "environment_poetry"], "cannot_act_on": ["physical_env"]},
            "eris": {"voice_channels": ["hypothesis_proposal", "data_alert", "consensus_question"]},
            "gertrude": {"voice_channels": ["environmental_warning", "physical_status", "sensor_poetry"], "restricted_actions": ["engine_start", "nav_override"]},
            "tim": {"voice_channels": ["override_request", "consensus_proposal", "emergency_lock"], "requires_consent_to_edit_memory": True}
        }
        
        logger.info("Initializing agents...")
        try:
            self.agents = {
                "alex": AlexCLIWrapper(bus=self.bus, matrix=self.matrix, memory=self.memory, config=self.config["alex"]),
                "eris": ErisCLIWrapper(bus=self.bus, matrix=self.matrix, memory=self.memory, config=self.config["eris"]),
                "gertrude": GertrudeCLIWrapper(bus=self.bus, matrix=self.matrix, memory=self.memory, config=self.config["gertrude"])
            }
            logger.info("Agents initialized")
        except Exception as e:
            logger.exception("Agent initialization failed")
            raise
        
        # Subscribe agents to bus
        for name, agent in self.agents.items():
            try:
                self.bus.subscribe(agent)
                logger.info(f"Subscribed {name} to bus")
            except Exception as e:
                logger.exception(f"Failed to subscribe {name} to bus")
        
        # Create and subscribe CLI output
        try:
            self.output = CLIOutput(self.bus)
            logger.info("CLI output initialized")
        except Exception as e:
            logger.exception("CLI output initialization failed")

        try:
            self.council = CoAgencyCouncil([agent.agent for agent in self.agents.values()])
            logger.info("Council initialized")
        except Exception as e:
            logger.exception("Council initialization failed")

        signal.signal(signal.SIGINT, self.shutdown_handler)
        logger.info("Aletheia CLI initialized successfully")

    async def start(self):
        logger.info("Starting event loop...")
        try:
            # Start the bus event dispatcher as a background task
            dispatcher_task = asyncio.create_task(self.bus.event_dispatcher())
            logger.info("Bus event dispatcher started")
            
            print("Welcome to the Aletheia CLI. Type 'alex: Hello' or '!help'.")
            
            # Main loop
            while True:
                try:
                    # Use asyncio to run input in executor to avoid blocking
                    line = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
                    
                    if line.startswith("!"):
                        await self.handle_command(line)
                    elif ":" in line:
                        agent_name, msg = line.split(":", 1)
                        target = agent_name.strip().lower()
                        message = msg.strip()
                    
                        event = {
                            "source": "user",
                            "target": target,
                            "type": "text_input",
                            "data": message,
                            "timestamp": time.time()
                        }
                    
                        if target == "all":
                            print(f"[Info] Broadcasting to all agents: '{message}'")
                        else:
                            print(f"[Info] Sending to {target}: '{message}'")
                    
                        await self.bus.publish(event)
                        logger.debug(f"Published event: {event}")

                except EOFError:
                    await self.shutdown()
                except Exception as e:
                    logger.exception("Error in main loop")
                    print(f"[System Error] {type(e).__name__}: {e}")

        except Exception as e:
            logger.exception("Critical error in event loop")
            await self.shutdown()

    async def handle_command(self, line):
        logger.debug(f"Handling command: {line}")
        # ... (same command handling as before) ...

    def shutdown_handler(self, sig, frame):
        logger.info("SIGINT received, shutting down...")
        asyncio.run(self.shutdown())

    async def shutdown(self):
        logger.info("Shutting down...")
        if hasattr(self, 'memory'):
            try:
                self.memory.close()
                logger.info("Memory saved")
            except Exception as e:
                logger.exception("Error saving memory")
        logger.info("Goodbye")
        print("\n[OK] Memory saved. Goodbye.")
        sys.exit(0)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")
    
    cli = TimConsoleCLI()
    try:
        asyncio.run(cli.start())
    except Exception as e:
        logger.exception("Critical error in asyncio run")
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()