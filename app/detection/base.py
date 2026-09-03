from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.models import DetectionSnapshot, FramePacket


class Detector(ABC):
    @abstractmethod
    def detect(self, packet: FramePacket) -> DetectionSnapshot:
        raise NotImplementedError

    def reset(self) -> None:
        pass

