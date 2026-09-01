from __future__ import annotations

from abc import ABC, abstractmethod
from src.models.domain import ActivityDiagram, Defect, RepairResult


class BaseRepair(ABC):
    repair_type = "BASE"

    @abstractmethod
    def repair(self, diagram: ActivityDiagram, defects: list[Defect]) -> RepairResult:
        raise NotImplementedError
