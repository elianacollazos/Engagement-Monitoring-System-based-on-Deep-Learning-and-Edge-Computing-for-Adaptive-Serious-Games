from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class SystemConfig:
    cameraSource: int = 0
    frameRate: int = 30
    width: int = 640
    height: int = 480
    modelPath: Optional[str] = None
    modelType: Optional[str] = None
    confidenceThreshold: float = 0.5


@dataclass
class Session:
    sessionId: str
    userId: str
    startTime: datetime
    endTime: Optional[datetime] = None


@dataclass
class SessionInfo:
    sessionId: str
    userId: str
    startTime: Optional[datetime]
    endTime: Optional[datetime]
    active: bool


@dataclass
class Resolution:
    width: int
    height: int


@dataclass
class Frame:
    data: Any
    frameId: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Face:
    data: Any
    bbox: Optional[tuple] = None


@dataclass
class Landmarks:
    points: Any


@dataclass
class ProcessedFrame:
    data: Any


@dataclass
class EmotionResult:
    label: str
    confidence: float
    scores: dict = field(default_factory=dict)
    raw: Any = None


@dataclass
class AttentionResult:
    score: float
    details: dict = field(default_factory=dict)


@dataclass
class PostureResult:
    score: float
    details: dict = field(default_factory=dict)


@dataclass
class GazeResult:
    score: float
    details: dict = field(default_factory=dict)


@dataclass
class CognitiveResult:
    attention: AttentionResult
    posture: PostureResult
    gaze: GazeResult
    score: float


@dataclass
class EngagementLevel:
    label: str
    score: float


@dataclass
class MappingConfig:
    levels: tuple = ("Disengaged", "Engaged", "Highly Engaged", "Fully Engaged")


@dataclass
class Report:
    session: Session
    records: list
    summary: dict = field(default_factory=dict)


@dataclass
class File:
    path: str
