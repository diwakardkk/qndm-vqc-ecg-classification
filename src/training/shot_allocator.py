from __future__ import annotations

from dataclasses import dataclass


class ShotAllocator:
    def get_shots(self, *, method: str, epoch: int, parameter: int | None = None) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class FixedShotAllocator(ShotAllocator):
    shots: int

    def get_shots(self, *, method: str, epoch: int, parameter: int | None = None) -> int:
        return self.shots


class CollaboratorAdaptiveShotAllocator(ShotAllocator):
    def get_shots(self, *, method: str, epoch: int, parameter: int | None = None) -> int:
        raise NotImplementedError("Collaborator adaptive-shot allocation rule has not been supplied.")


@dataclass(frozen=True)
class HeuristicAdaptiveShotAllocator(ShotAllocator):
    base_shots: int

    label = "EXPLORATORY HEURISTIC - NOT COLLABORATOR METHOD"

    def get_shots(self, *, method: str, epoch: int, parameter: int | None = None) -> int:
        return int(self.base_shots * (1 + epoch // 5))

