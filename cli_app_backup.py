#cli_app.py
from multiprocessing import shared_memory
import torch
import asyncio
import json
import signal
import sys
import time

from agents.agent_base import Agent

from core.shared_bus import AsyncSharedBus
from core.escalation_graph import EscalationMatrix
from core.council import CoAgencyCouncil

from memory.shared_memory import shared_memory as global_shared_memory

from agents.alex_cli_adapter import AlexCLIWrapper
from agents.eris_cli_adapter import ErisCLIWrapper
from agents.gertrude_cli_adapter import GertrudeCLIWrapper



class TimConsoleCLI:
    def __init__(self):
        self.bus = AsyncSharedBus()
        self.memory = shared_memory
        self.matrix = EscalationMatrix()
        self.memory.set_autosave_path("memory_state.json")  # Add persistent storage

        self.config = {
            "alex": {"voice_channels": ["emotional_expression", "mediation_request", "environment_poetry"], "cannot_act_on": ["physical_env"]},
            "eris": {"voice_channels": ["hypothesis_proposal", "data_alert", "consensus_question"]},
            "gertrude": {"voice_channels": ["environmental_warning", "physical_status", "sensor_poetry"], "restricted_actions": ["engine_start", "nav_override"]},
            "tim": {"voice_channels": ["override_request", "consensus_proposal", "emergency_lock"], "requires_consent_to_edit_memory": True}
        }
        
        self.agents = {
            "alex": AlexCLIWrapper(bus=self.bus, matrix=self.matrix, memory=self.memory, config=self.config["alex"]),
            "eris": ErisCLIWrapper(bus=self.bus, matrix=self.matrix, memory=self.memory, config=self.config["eris"]),
            "gertrude": GertrudeCLIWrapper(bus=self.bus, matrix=self.matrix, memory=self.memory, config=self.config["gertrude"])
        }
        
        # Create and subscribe CLI output
        self.output = CLIOutput(self.bus)


        for agent in self.agents.values():
            self.bus.subscribe(agent)  # Subscribe the wrapper

        self.council = CoAgencyCouncil([agent.agent for agent in self.agents.values()])

        signal.signal(signal.SIGINT, self.shutdown_handler)

    async def start(self):
        asyncio.create_task(self.bus.event_dispatcher())
        print("Welcome to the Aletheia CLI. Type 'alex: Hello' or '!help'.")

        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
                if line.startswith("!"):
                    await self.handle_command(line)
                elif ":" in line:
                    agent_name, msg = line.split(":", 1)
                    target = agent_name.strip().lower()
                    message = msg.strip()
                
                    # Create proper event
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

            except EOFError:
                await self.shutdown()

    async def handle_command(self, line):
        cmd = line.strip().lower()

        if cmd == "!exit":
            await self.shutdown()

        elif cmd == "!help":
            print("""
Available commands:
  alex: message         - Speak to an agent
  !memory               - Show recent episodic memory
  !semantic             - Show current semantic memory
  !journal agent        - Show agent journal
  !vote {json}          - Submit a council vote
  !exit                 - Quit
""")

        elif cmd == "!memory":
            for who, what in self.memory.episodic[-10:]:
                print(f"[{who}] {what}")

        elif cmd == "!semantic":
            for k, v in self.memory.semantic.items():
                print(f"[{k}] {json.dumps(v)}")

        elif cmd.startswith("!journal"):
            parts = cmd.split()
            if len(parts) < 2:
                print("[Error] Usage: !journal agent")
                return
            agent = self.agents.get(parts[1])
            if agent:
                for e in agent.journal[-10:]:
                    print(f"{e['type']} → {e['data']}")
            else:
                print("[Error] Unknown agent")

        elif cmd.startswith("!vote"):
            try:
                payload = json.loads(line[len("!vote"):].strip())
                result = await self.council.vote(payload)
                print("Council Results:", result)
                if self.council.decision(result):
                    print("Consensus reached → applying to memory.")
                    await self.memory.update("council", "semantic", payload)
            except json.JSONDecodeError:
                print("[Error] Invalid JSON for vote")

        else:
            print("[Error] Unknown command")

    def shutdown_handler(self, sig, frame):
        print("\n[Signal] Saving memory and exiting...")
        asyncio.run(self.shutdown())

    async def shutdown(self):
        if hasattr(self, 'memory'):
            self.memory.close()  # Calls autosave
        print("[OK] Memory saved. Goodbye.")
        sys.exit(0)

    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



