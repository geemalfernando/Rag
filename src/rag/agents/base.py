import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field


@dataclass
class TraceStep:
    agent: str
    summary: str
    detail: dict = field(default_factory=dict)
    ms: int = 0


@dataclass
class Trace:
    """What each agent did, in order. Shown in the CLI and web UI so the team's work is visible."""

    steps: list[TraceStep] = field(default_factory=list)

    @contextmanager
    def step(self, agent: str):
        step = TraceStep(agent=agent, summary="")
        start = time.perf_counter()
        try:
            yield step
        finally:
            step.ms = round((time.perf_counter() - start) * 1000)
            self.steps.append(step)

    def to_list(self) -> list[dict]:
        return [asdict(s) for s in self.steps]
