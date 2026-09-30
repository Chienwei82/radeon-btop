"""Data-layer readers.

Every module here is a *total* reader: it degrades to ``None`` or an empty result rather
than raising, because sysfs entries vanish and permissions vary between kernels.
"""

from gputop.readers.discovery import discover_devices
from gputop.readers.dpm import read_clock, read_performance_level, read_power_profile
from gputop.readers.hwmon import read_fan, read_power, read_temperatures
from gputop.readers.metrics_bin import GpuMetricsParser, ParsedGpuMetrics
from gputop.readers.procfs import ProcessCollector
from gputop.readers.throttle import ThrottleStatus, decode_throttle

__all__ = [
    "GpuMetricsParser",
    "ParsedGpuMetrics",
    "ProcessCollector",
    "ThrottleStatus",
    "decode_throttle",
    "discover_devices",
    "read_clock",
    "read_fan",
    "read_performance_level",
    "read_power",
    "read_power_profile",
    "read_temperatures",
]
