"""A small team of agents that answer questions together.

Planner -> Researcher -> Writer -> Verifier (-> Writer again if the Verifier finds unsupported claims).
Each agent has one job and hands a typed result to the next; the Orchestrator runs them and records a trace.
"""

from rag.agents.base import Trace, TraceStep
from rag.agents.planner import Plan, Planner

__all__ = ["Plan", "Planner", "Trace", "TraceStep"]
