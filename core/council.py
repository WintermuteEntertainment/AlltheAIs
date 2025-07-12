# --- core/council.py ---
import asyncio
import time
import random

import json
import signal
import sys






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