"""Configuration loaded with the stdlib ``tomllib`` module.

Precedence, lowest to highest: dataclass defaults -> packaged config file ->
``$GPUTOP_CONFIG`` -> ``~/.config/gputop/gputop.toml`` -> ``~/.gputoprc`` -> CLI flags.

Unknown keys are collected as warnings rather than raising, so a config written for a
future release still works today.
"""

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from gputop.model.history import DEFAULT_HISTORY_LENGTH
from gputop.ui.theme import DEFAULT_THEME_NAME, THEMES

MIN_INTERVAL_MS = 100
MAX_INTERVAL_MS = 10_000


@dataclass(frozen=True, slots=True, kw_only=True)
class GeneralConfig:
    """Top level behaviour."""

    interval_ms: int = 1000
    history_points: int = DEFAULT_HISTORY_LENGTH


@dataclass(frozen=True, slots=True, kw_only=True)
class GpuConfig:
    """Device selection and classification."""

    devices: tuple[str, ...] = ()
    kind: str = "auto"
    default_index: int = 0
    names: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProcessConfig:
    """Per-process table behaviour."""

    show: bool = True
    max_rows: int = 20
    sort: str = "engine"
    min_usage_percent: float = 0.0
    hide_kernel_threads: bool = True
    include_all_users: bool = False


DEFAULT_DUMP_DIR = Path("~/.local/state/gputop").expanduser()


@dataclass(frozen=True, slots=True, kw_only=True)
class DiagnosticsConfig:
    """Diagnostic dump behaviour."""

    dump_format: str = "json"
    # default_factory rather than a bare default: the call must run per instance so that
    # changing $HOME after import is still honoured.
    dump_dir: Path = field(default_factory=lambda: DEFAULT_DUMP_DIR)


@dataclass(frozen=True, slots=True, kw_only=True)
class UiConfig:
    """Appearance and layout."""

    theme: str = DEFAULT_THEME_NAME
    no_color: bool = False
    graph_height: int = 10
    graph_history: int = 300
    hide_on_zero: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class Config:
    """The complete, validated configuration."""

    general: GeneralConfig = GeneralConfig()
    gpu: GpuConfig = GpuConfig()
    process: ProcessConfig = ProcessConfig()
    ui: UiConfig = UiConfig()
    diagnostics: DiagnosticsConfig = DiagnosticsConfig()
    warnings: tuple[str, ...] = ()

    @property
    def interval_s(self) -> float:
        """The sampling interval in seconds."""
        return self.general.interval_ms / 1000.0

    @property
    def history_length(self) -> int:
        """Number of samples retained for graph history."""
        return max(1, self.general.history_points)


def config_search_paths() -> tuple[Path, ...]:
    """Return the config file locations searched when no explicit path is given."""
    paths: list[Path] = []
    env_path = os.environ.get("GPUTOP_CONFIG")
    if env_path:
        paths.append(Path(env_path))
    paths.append(Path.home() / ".config" / "gputop" / "gputop.toml")
    paths.append(Path.home() / ".gputoprc")
    return tuple(paths)


def _as_section(value: object) -> dict[str, Any]:
    """Coerce a TOML value into a mapping, treating anything else as empty."""
    return dict(value) if isinstance(value, dict) else {}


def _build(cls: type[Any], values: dict[str, Any], where: str, warnings: list[str]) -> Any:
    """Instantiate a config section, rejecting unknown keys and bad types."""
    known = {f.name for f in fields(cls)}
    for key in values:
        if key not in known:
            warnings.append(f"{where}.{key}: unknown key ignored")
    kwargs: dict[str, Any] = {}
    for name in known & values.keys():
        kwargs[name] = values[name]
    try:
        return cls(**kwargs)
    except TypeError as exc:
        warnings.append(f"{where}: invalid configuration ({exc})")
        return cls()


def _validate(config: Config, warnings: list[str]) -> Config:
    """Clamp ranges and normalise enum-ish fields in place-safe fashion."""
    general = config.general
    if not MIN_INTERVAL_MS <= general.interval_ms <= MAX_INTERVAL_MS:
        clamped = min(MAX_INTERVAL_MS, max(MIN_INTERVAL_MS, general.interval_ms))
        warnings.append(
            f"general.interval_ms={general.interval_ms} out of range, using {clamped}"
        )
        general = replace(general, interval_ms=clamped)
    if general.history_points < 1:
        warnings.append(f"general.history_points={general.history_points} below 1, using 1")
        general = replace(general, history_points=1)

    process = config.process
    if process.max_rows < 1:
        warnings.append(f"process.max_rows={process.max_rows} below 1, using 1")
        process = replace(process, max_rows=1)

    gpu = config.gpu
    if gpu.kind not in ("auto", "igpu", "dgpu"):
        warnings.append(f"gpu.kind={gpu.kind!r} invalid, using 'auto'")
        gpu = replace(gpu, kind="auto")

    ui = config.ui
    if ui.theme not in THEMES:
        warnings.append(
            f"ui.theme={ui.theme!r} unknown, using {DEFAULT_THEME_NAME!r} "
            f"(available: {', '.join(THEMES)})"
        )
        ui = replace(ui, theme=DEFAULT_THEME_NAME)
    if ui.graph_height < 3:
        warnings.append(f"ui.graph_height={ui.graph_height} below 3, using 3")
        ui = replace(ui, graph_height=3)
    if ui.graph_history < 2:
        warnings.append(f"ui.graph_history={ui.graph_history} below 2, using 2")
        ui = replace(ui, graph_history=2)

    return replace(config, general=general, process=process, gpu=gpu, ui=ui)


def load_config(path: Path | None = None) -> Config:
    """Load configuration from TOML.

    Args:
        path: An explicit config file.  When ``None`` the search path is used and a
            missing file simply yields defaults.

    Returns:
        A validated :class:`Config`.
    """
    warnings: list[str] = []
    raw: dict[str, Any] = {}
    chosen: Path | None = None

    candidates = (path,) if path is not None else config_search_paths()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with candidate.open("rb") as handle:
                raw = tomllib.load(handle)
            chosen = candidate
        except (OSError, tomllib.TOMLDecodeError) as exc:
            warnings.append(f"{candidate}: could not be read ({exc})")
            return Config(warnings=tuple(warnings))
        break

    if chosen is None and path is not None:
        warnings.append(f"{path}: not found, using defaults")

    config = Config(
        general=_build(GeneralConfig, _as_section(raw.get("general")), "general", warnings),
        gpu=_build(GpuConfig, _as_section(raw.get("gpu")), "gpu", warnings),
        process=_build(ProcessConfig, _as_section(raw.get("process")), "process", warnings),
        ui=_build(UiConfig, _as_section(raw.get("ui")), "ui", warnings),
        diagnostics=_build(
            DiagnosticsConfig, _as_section(raw.get("diagnostics")), "diagnostics", warnings
        ),
    )
    config = _validate(config, warnings)
    return replace(config, warnings=tuple(warnings))
