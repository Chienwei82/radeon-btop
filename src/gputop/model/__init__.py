"""Data model for gputop.

Every module in this package is pure data: no I/O, no UI, no logging.  All dataclasses
are frozen and slotted so that snapshots can be safely handed between the sampler
thread and the UI thread on both the standard and the free-threaded CPython builds.
"""

from gputop.model.aliases import (
    Bdf,
    Bytes,
    Celsius,
    ClientId,
    Mhz,
    Nanoseconds,
    Percent,
    Rpm,
    Watts,
)
from gputop.model.device import AmdgpuDevice, DeviceKind
from gputop.model.history import RingBuffer
from gputop.model.metrics import (
    AmdgpuMetrics,
    Clock,
    Fan,
    MemoryPool,
    PcieLink,
    Power,
    Temperature,
    ThrottleInfo,
)
from gputop.model.process import EngineUsage, GpuProcess
from gputop.model.snapshot import GpuSnapshot, SamplerStats, SourceReport

__all__ = [
    "AmdgpuDevice",
    "AmdgpuMetrics",
    "Bdf",
    "Bytes",
    "Celsius",
    "ClientId",
    "Clock",
    "DeviceKind",
    "EngineUsage",
    "Fan",
    "GpuProcess",
    "GpuSnapshot",
    "MemoryPool",
    "Mhz",
    "Nanoseconds",
    "PcieLink",
    "Percent",
    "Power",
    "RingBuffer",
    "Rpm",
    "SamplerStats",
    "SourceReport",
    "Temperature",
    "ThrottleInfo",
    "Watts",
]
