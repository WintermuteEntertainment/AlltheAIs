# --- core/escalation_graph.py ---
import asyncio
import time
import random


class EscalationMatrix:
    def __init__(self):
        self.trust_paths = {
            "gertrude": [
                ("alex", "environment", 0.7),
                ("tim", "emergency", 0.9),
                ("eris", "data_validation", 0.5)
            ],
            "eris": [
                ("alex", "interpretation", 0.6),
                ("gertrude", "reality_check", 0.8),
                ("tim", "ethical_audit", 0.85)
            ],
            "alex": [
                ("eris", "analysis", 0.75),
                ("gertrude", "embodiment", 0.9),
                ("tim", "existential", 0.65)
            ],
            "tim": [
                ("alex", "mediation", 0.8),
                ("eris", "strategic_audit", 0.7),
                ("gertrude", "embodiment", 0.85)
            ]
        }
        self.bidirectional_paths = {
            ("alex", "gertrude"): "embodiment_expression",
            ("eris", "gertrude"): "data_physicalization",
            ("tim", "gertrude"): "human_embodiment"
        }

    def get_escalation_paths(self, source, concern_type=None):
        paths = self.trust_paths.get(source, [])
        if concern_type:
            return [p for p in paths if p[1] == concern_type]
        return paths

    def add_bidirectional_trust(self, agent1, agent2, relationship_type):
        self.bidirectional_paths[(agent1, agent2)] = relationship_type
        self.bidirectional_paths[(agent2, agent1)] = relationship_type + "_reciprocal"

    def get_relationship(self, agent1, agent2):
        return self.bidirectional_paths.get((agent1, agent2)) or \
               self.bidirectional_paths.get((agent2, agent1))
