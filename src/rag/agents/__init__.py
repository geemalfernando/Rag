"""A small team of agents that answer questions together.

Planner -> Researcher -> Writer -> Verifier (-> Writer again if the Verifier finds unsupported claims).
Each agent has one job and hands a typed result to the next; the Orchestrator runs them and records a trace.
"""

from rag.agents.base import Trace, TraceStep
from rag.agents.orchestrator import Orchestrator, TeamAnswer
from rag.agents.planner import Plan, Planner
from rag.agents.researcher import Researcher
from rag.agents.verifier import Verdict, Verifier
from rag.agents.writer import Writer

__all__ = ["Orchestrator", "Plan", "Planner", "Researcher", "TeamAnswer", "Trace", "TraceStep", "Verdict", "Verifier", "Writer"]
