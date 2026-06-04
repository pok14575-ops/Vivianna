from dataclasses import dataclass

@dataclass
class Tool:
    name: str
    description: str
    confidence_threshold: float
    priority: int

TOOLS = [
    Tool(
        name="web",
        description="requires live internet data: news, weather, prices, current events, recent releases, unknown people or companies",
        confidence_threshold=0.90,
        priority=1,
    ),
    Tool(
        name="chat",
        description="general conversation, knowledge, reasoning, tasks that do not need live or real-time data",
        confidence_threshold=0.0,
        priority=99,
    ),
]
