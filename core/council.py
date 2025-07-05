# --- core/council.py ---
import asyncio
from agents.alex import Alex
from agents.eris import Eris
from agents.gertrude import Gertrude
from tim_interface import TimConsole
from core.council import CoAgencyCouncil
from core.shared_memory import SharedMemory
from core.escalation_graph import EscalationMatrix
from core.embodied_security import EmbodiedVerification


class CoAgencyCouncil:
    def __init__(self, agents):
        self.agents = agents
        self.vote_log = []

    async def vote(self, proposal):
        results = await asyncio.gather(*[
            agent.vote(proposal) for agent in self.agents
        ])
        result_map = dict(zip([agent.name for agent in self.agents], results))
        self.vote_log.append({"proposal": proposal, "results": result_map, "timestamp": time.time()})
        return result_map

    def decision(self, results):
        approvals = sum(1 for v in results.values() if v == "approve")
        return approvals > len(results) // 2