
import asyncio
from agents.alex import Alex
from agents.eris import Eris
from agents.gertrude import Gertrude
from tim_interface import TimConsole
from core.council import CoAgencyCouncil
from core.shared_memory import SharedMemory
from core.escalation_graph import EscalationMatrix
from core.embodied_security import EmbodiedVerification

config = {
    "alex": {"voice_channels": ["emotional_expression", "mediation_request", "environment_poetry"], "cannot_act_on": ["physical_env"]},
    "eris": {"voice_channels": ["hypothesis_proposal", "data_alert", "consensus_question"]},
    "gertrude": {"voice_channels": ["environmental_warning", "physical_status", "sensor_poetry"], "restricted_actions": ["engine_start", "nav_override"]},
    "tim": {"voice_channels": ["override_request", "consensus_proposal", "emergency_lock"], "requires_consent_to_edit_memory": True}
}

bus = AsyncSharedBus()
matrix = EscalationMatrix()
memory = SharedMemory()

alex = Alex("alex", bus, config, escalation_matrix=matrix, shared_memory=memory)
eris = Eris("eris", bus, config, escalation_matrix=matrix, shared_memory=memory)
gertrude = Gertrude("gertrude", bus, config, escalation_matrix=matrix, shared_memory=memory)
tim = TimConsole()

council = CoAgencyCouncil([alex, eris, gertrude])

bus.subscribe(alex)
bus.subscribe(eris)
bus.subscribe(gertrude)

async def main_loop():
    asyncio.create_task(bus.event_dispatcher())

    for tick in range(10):
        print(f"\n--- Tick {tick} ---")
        if tick == 1:
            await alex.publish("text_input", "Alex, do you think Eris likes metaphors?")
        if tick == 3:
            proposal = {"action": "memory_edit", "reason": "optimize recall for ethics module"}
            result = await council.vote(proposal)
            print("\nCOUNCIL VOTE RESULT:", result)
            if council.decision(result):
                print("Consensus: Proposal approved.")
                await memory.update("council", "semantic", proposal)
            else:
                print("Consensus: Proposal denied.")
        await alex.act()
        await asyncio.sleep(0.5)

asyncio.run(main_loop())

