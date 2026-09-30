"""Total, never-raising read primitives for sysfs and procfs.

Every function in this module returns ``None`` on any kind of failure -- missing file,
permission denied, garbage content, or a race where the file vanished between ``stat``
and ``open``.  The read layer must never raise; that policy is what lets the sampler
thread and the UI treat "unavailable" as ordinary data rather than an error path.
"""

import re
from pathlib import Path

_WHITESPACE = " \t\r\n\x0b\x0c"


def read_text(path: Path) -> str | None:
    """Read a small text file, returning ``None`` if it cannot be read."""
    try:
        with path.open("rb") as handle:
            return handle.read(4096).decode("utf-8", errors="replace").strip()
    except OSError, ValueError:
        return None


def read_int(path: Path) -> int | None:
    """Read a file whose entire content is a single integer."""
    text = read_text(path)
    return parse_int(text)


def parse_int(text: str | None) -> int | None:
    """Parse a stripped integer string, tolerating trailing whitespace and junk."""
    if not text:
        return None
    try:
        return int(text.strip(), 10)
    except TypeError, ValueError:
        return None


def read_bytes(path: Path, limit: int = 65_536) -> bytes | None:
    """Read up to ``limit`` bytes from a binary file."""
    try:
        with path.open("rb") as handle:
            return handle.read(limit)
    except OSError, ValueError:
        return None


def read_hex_int(path: Path) -> int | None:
    """Read a hexadecimal integer such as ``0x1002`` as used by ``vendor``/``device``."""
    text = read_text(path)
    if not text:
        return None
    try:
        return int(text, 16)
    except ValueError:
        return None


def parse_first_int(text: str | None) -> int | None:
    """Return the first integer appearing anywhere in a whitespace-formatted file.

    Sysfs files such as ``pp_dpm_sclk`` are tables rather than single values, so this
    extracts the first number instead of demanding the whole content be one integer.
    """
    if not text:
        return None
    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else None


def read_first_int(path: Path) -> int | None:
    """Read a file and return the first integer found in it."""
    return parse_first_int(read_text(path))


def link_target(path: Path) -> str | None:
    """Resolve a symlink without raising."""
    try:
        return path.readlink().as_posix()
    except OSError:
        return None


def resolve(path: Path) -> Path | None:
    """Resolve a path without raising, returning ``None`` on failure."""
    try:
        return path.resolve(strict=False)
    except OSError, RuntimeError:
        return None


def strip_padding(text: str | None) -> str | None:
    """Strip surrounding whitespace from an optional string."""
    if text is None:
        return None
    return text.strip(_WHITESPACE)


def parse_power_microwatts(text: str | None) -> float | None:
    """Convert a hwmon micro-watt reading into watts."""
    value = parse_int(text)
    if value is None:
        return None
    return value / 1_000_000.0


def parse_millidegrees(text: str | None) -> float | None:
    """Convert a hwmon milli-degree reading into whole degrees Celsius."""
    value = parse_int(text)
    if value is None:
        return None
    return value / 1000.0


#: Multipliers for the units the DRM fdinfo keys use.  The amdgpu driver writes the
#: *binary* IEC prefixes (``KiB``/``MiB``/``GiB``), while ``btop``-style tables assume
#: powers of 1000 -- getting this wrong would misreport every per-process memory figure
#: by 2.4%, so both spellings are accepted and the IEC ones use powers of 1024.
_SIZE_UNITS: dict[str, int] = {
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}


def parse_drm_size(text: str | None) -> int | None:
    """Parse a DRM fdinfo size value such as ``4116 KiB`` into bytes.

    Handles the optional trailing space in ``drm-memory-gtt: \\t0 KiB`` that the amdgpu
    driver emits.
    """
    if not text:
        return None
    parts = text.replace("\t", " ").split()
    if not parts:
        return None
    number = parse_int(parts[0])
    if number is None:
        return None
    unit = parts[1].lower() if len(parts) > 1 else "b"
    factor = _SIZE_UNITS.get(unit)
    if factor is None:
        return None
    return number * factor
