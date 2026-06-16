from datetime import datetime
from uuid import uuid4

from .types import Session, SessionInfo, SystemConfig


class SessionManager:
    def __init__(self, config: SystemConfig | None = None):
        self.sessionId = ""
        self.userId = ""
        self.config = config or SystemConfig()
        self.startTime = None
        self.endTime = None
        self._session = None

    def isActive(self) -> bool:
        return self._session is not None and self.endTime is None

    def closed(self) -> None:
        self.endSession()

    def startSession(self, userId: str) -> Session:
        self.sessionId = str(uuid4())
        self.userId = userId
        self.startTime = datetime.now()
        self.endTime = None
        self._session = Session(self.sessionId, userId, self.startTime)
        return self._session

    def endSession(self) -> None:
        self.endTime = datetime.now()
        if self._session is not None:
            self._session.endTime = self.endTime

    def getSessionInfo(self) -> SessionInfo:
        return SessionInfo(
            sessionId=self.sessionId,
            userId=self.userId,
            startTime=self.startTime,
            endTime=self.endTime,
            active=self.isActive(),
        )
