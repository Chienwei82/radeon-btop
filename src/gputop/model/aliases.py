"""Type aliases shared across the data layer.

These are PEP 695 ``type`` aliases: they are lazily evaluated under PEP 649 and are
usable both as annotations and inside ``isinstance`` narrowing checks.
"""

type Mhz = int
type Bytes = int
type Rpm = int

#: hwmon reports power in microwatts and temperature in millidegrees, so both genuinely
#: carry sub-unit resolution.  Typing them as ``int`` would force a lossy round-trip.
type Celsius = float
type Watts = float

type Percent = float
type Bdf = str
type ClientId = int
type Nanoseconds = int

__all__ = [
    "Bdf",
    "Bytes",
    "Celsius",
    "ClientId",
    "Mhz",
    "Nanoseconds",
    "Percent",
    "Rpm",
    "Watts",
]
