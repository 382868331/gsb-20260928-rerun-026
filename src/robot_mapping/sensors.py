from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .models import SensorRecord


@dataclass(frozen=True)
class SensorSpec:
    sensor_id: str
    sensor_type: str


class SensorRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, SensorSpec] = {}

    def register(self, sensor_id: str, sensor_type: str) -> None:
        existing = self._specs.get(sensor_id)
        if existing is not None and existing.sensor_type != sensor_type:
            raise ValueError(f"Sensor {sensor_id} already registered as {existing.sensor_type}")
        self._specs[sensor_id] = SensorSpec(sensor_id, sensor_type)

    def require(self, sensor_id: str, sensor_type: str) -> SensorSpec:
        spec = self._specs.get(sensor_id)
        if spec is None:
            spec = SensorSpec(sensor_id, sensor_type)
            self._specs[sensor_id] = spec
        if spec.sensor_type != sensor_type:
            raise ValueError(f"Sensor {sensor_id} has type {spec.sensor_type}, expected {sensor_type}")
        return spec

    def specs(self) -> tuple[SensorSpec, ...]:
        return tuple(self._specs.values())


class AppendOnlyStore:
    """Append-only raw sensor log.

    The log is deliberately independent from optimized poses. Correcting the
    map never mutates a record already stored here.
    """

    def __init__(self, registry: SensorRegistry | None = None) -> None:
        self.registry = registry or SensorRegistry()
        self._records: list[SensorRecord] = []
        self._by_type: dict[str, list[SensorRecord]] = defaultdict(list)
        self._next_id = 0

    def append(self, sensor_id: str, sensor_type: str, timestamp: float, payload: object) -> SensorRecord:
        self.registry.require(sensor_id, sensor_type)
        record_id = f"record-{self._next_id:05d}"
        record = SensorRecord(record_id, sensor_id, sensor_type, timestamp, payload)
        self._records.append(record)
        self._by_type[sensor_type].append(record)
        self._next_id += 1
        return record

    def records(self, sensor_type: str | None = None) -> tuple[SensorRecord, ...]:
        if sensor_type is None:
            return tuple(self._records)
        return tuple(self._by_type[sensor_type])

    def payloads(self, sensor_type: str) -> Iterable[object]:
        return (record.payload for record in self._by_type[sensor_type])

    def __len__(self) -> int:
        return len(self._records)
