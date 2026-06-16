from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from .types import CognitiveResult, EmotionResult, EngagementLevel


@dataclass
class TemporalRecord:
    timestamp: datetime
    frameId: int
    face: Any
    emotion: EmotionResult
    cognitive: CognitiveResult
    engagement: EngagementLevel

    def toJSON(self) -> dict:
        return asdict(self)
