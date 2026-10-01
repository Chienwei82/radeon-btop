"""Configuration loaded with the stdlib ``tomllib`` module.

Precedence, lowest to highest: dataclass defaults -> packaged config file ->
``$GPUTOP_CONFIG`` -> ``~/.config/gputop/gputop.toml`` -> ``~/.gputoprc`` -> CLI flags.

Unknown keys are collected as warnings rather than raising, so a config written for a
future release still works today.
"""

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, cast

from gputop.model.alerts import Thresholds as AlertThresholds
from gputop.model.history import DEFAULT_HISTORY_LENGTH
from gputop.model.process import SORT_COLUMNS
from gputop.readers.radeontop import RadeontopOptions
from gputop.ui.theme import DEFAULT_THEME_NAME, THEMES

MIN_INTERVAL_MS = 100
MAX_INTERVAL_MS = 10_000

#: The temperature the sensor panel draws its scale against, and so the ceiling for an alert
#: threshold: past this point every AMD part is throttling hard enough that the number stops
#: being informative.
MAX_TEMP_C = 125.0

#: Environment variable that overrides where the session state is read from and written
#: to.  It exists so a test, a container or a second profile can keep its state separate
#: from the real one instead of quietly inheriting whatever the last session chose.
STATE_ENV = "GPUTOP_STATE"

#: Where the session state lives by default.  The name is the same as a config file's
#: would be, because it *is* the same file: gputop writes its ``[state]`` section into
#: the user's config and leaves every other section alone.
DEFAULT_STATE_PATH = Path("~/.config/gputop/config.toml")


@dataclass(frozen=True, slots=True, kw_only=True)
class GeneralConfig:
    """Top level behaviour."""

    interval_ms: int = 1000
    history_points: int = DEFAULT_HISTORY_LENGTH


@dataclass(frozen=True, slots=True, kw_only=True)
class GpuConfig:
    """Device selection and classification."""

    #: Restrict monitoring to these cards, named by PCI address or by ``cardN``.  Empty
    #: means "every amdgpu device found", which is what a single-GPU machine wants and what
    #: a multi-GPU one has to narrow deliberately.
    devices: tuple[str, ...] = ()
    kind: str = "auto"
    names: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProcessConfig:
    """Per-process table behaviour."""

    show: bool = True
    max_rows: int = 20
    sort: str = "total"
    min_usage_percent: float = 0.0
    hide_kernel_threads: bool = True
    #: Show the process as a parent/child tree rather than a flat list.
    process_tree: bool = False
    #: Show the full ``/proc/<pid>/cmdline`` instead of the 15-character ``comm`` name.
    full_command: bool = False
    #: Show clients that are holding a descriptor but doing nothing.  Off by default:
    #: such a row answers none of the questions this table exists for.
    show_idle: bool = False
    #: Allow the ``k`` key to signal a selected process.  Off by default because
    #: terminating another process from a monitor is a decision that needs a second
    #: one, and most sessions have no reason to ever make it.
    allow_kill: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class StateConfig:
    """What the interface looked like last time, written by gputop itself.

    Every field is a *runtime* choice -- the keys are the ones a user flips while
    watching, not the ones they set once.  ``None`` means "not recorded", which is
    distinct from a recorded default: a field that was never written must not silently
    override the value the user put in the file by hand.

    This is the only section of the config that gputop writes.  See
    :mod:`gputop.state` for the file it lives in and how it is merged.
    """

    theme: str | None = None
    interval_ms: int | None = None
    sort: str | None = None
    sort_reverse: bool | None = None
    filter: str = ""
    show_idle: bool = False
    process_tree: bool = False
    full_command: bool = False
    overview: bool = False
    gpu_index: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class BlocksConfig:
    """The optional radeontop-backed per-block panel.

    **Off by default.**  Enabling it spawns a child process that, on most unprivileged
    systems, answers "are you root?" -- so a default-on panel would put a permanent
    complaint in the status line of every user who cannot use it.  Opt-in also stops gputop
    from launching a process that reaches for the GPU the moment it starts, which is not
    something a monitor should do unasked.

    ``--blocks`` or ``blocks.enabled = true`` turns it on; once it is on and still cannot
    run, the reason is stated rather than hidden.
    """

    enabled: bool = False
    #: Program name looked up on ``$PATH``, or an absolute path to the binary.
    binary: str = "radeontop"
    #: Samples per second asked of radeontop.  Higher is more accurate and costs more CPU.
    ticks: int = 120
    #: Seconds between dumps.  Whole seconds only: radeontop parses ``-i`` with ``atoi``.
    interval_s: int = 1
    #: Where DRM nodes live, so ``-p`` can be preferred over an ambiguous PCI bus number.
    drm_root: Path = Path("/dev/dri")
    #: Restart a child that dies mid-session.  Retries are bounded; see the reader.
    restart: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class AlertsConfig:
    """Threshold alerting.

    Thresholds are absolute rather than derived, because there is no honest way to derive
    "too hot" from what the card reports: the junction limit is a property of the silicon,
    not something sysfs states.  ``0`` disables an individual threshold, which is how a
    user turns off the power alert without editing three places.
    """

    enabled: bool = True
    #: Hottest sensor in Celsius; ``0`` disables.
    temp_c: float = 90.0
    #: Power draw as a percentage of the enforced cap; ``0`` disables.
    power_percent: float = 95.0
    #: VRAM in use as a percentage of the pool; ``0`` disables.
    vram_percent: float = 92.0
    #: Full flash cycles per second when a threshold is crossed.  0 holds a steady border.
    flash_hz: float = 1.0

    def thresholds(self) -> AlertThresholds:
        """The model's threshold record this config describes."""
        return AlertThresholds(
            temp_c=self.temp_c,
            power_percent=self.power_percent,
            vram_percent=self.vram_percent,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LogConfig:
    """Session recording.

    Only ``--log`` turns recording on; this section configures how, so a user who records
    every session from a wrapper script does not have to repeat the compression level.
    """

    #: zstd compression level for a ``.zst`` target.  1-19, or 0 for the library default.
    zstd_level: int = 3
    #: Seconds between records.  ``0`` records every sample.
    interval_s: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class UiConfig:
    """Appearance and layout."""

    theme: str = DEFAULT_THEME_NAME
    no_color: bool = False
    #: Rows in the utilisation graph.  ``0`` -- the default -- lets the graph fill whatever
    #: the responsive layout can spare, which is what the interface has always done and what
    #: a fluid layout wants.  A positive value pins it to that many rows, which is what a
    #: user who wants a tall graph on a big terminal and does not want to lose it on a
    #: small one asks for; it is clamped down rather than allowed to overflow.
    #:
    #: There is deliberately no second "points per graph" key here.  ``history_points``
    #: sizes the sampler's ring buffer, the graphs read from that buffer, and a graph
    #: cannot hold more samples than the buffer feeding it -- so a ``graph_history`` beside
    #: it could only ever have been a value silently capped by ``history_points``, or the
    #: other way round.  One key, one number.
    graph_height: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class Config:
    """The complete, validated configuration."""

    general: GeneralConfig = GeneralConfig()
    gpu: GpuConfig = GpuConfig()
    process: ProcessConfig = ProcessConfig()
    ui: UiConfig = UiConfig()
    blocks: BlocksConfig = BlocksConfig()
    alerts: AlertsConfig = AlertsConfig()
    log: LogConfig = LogConfig()
    state: StateConfig = StateConfig()
    warnings: tuple[str, ...] = ()

    @property
    def interval_s(self) -> float:
        """The sampling interval in seconds."""
        return self.general.interval_ms / 1000.0

    @property
    def history_length(self) -> int:
        """Number of samples retained for graph history."""
        return max(1, self.general.history_points)


#: The top-level tables :func:`load_config` reads, taken from the dataclass rather than
#: written out, so a section cannot be added without this knowing about it.
#:
#: This exists because the loader's promise is that a config written for a different
#: version still works *and says so*.  That promise was kept for keys within a section and
#: quietly broken for whole sections: a table nothing reads is passed over in silence, so
#: ``[diagnostics]`` could sit in the README, in the example file and in a user's config
#: for a year while being parsed cleanly and read by nothing.  Removing those keys without
#: this would have turned a section full of settings into a section that vanishes.
KNOWN_SECTIONS = frozenset(f.name for f in fields(Config) if f.name != "warnings")


def config_search_paths() -> tuple[Path, ...]:
    """Return the config file locations searched when no explicit path is given."""
    paths: list[Path] = []
    env_path = os.environ.get("GPUTOP_CONFIG")
    if env_path:
        paths.append(Path(env_path))
    paths.append(Path.home() / ".config" / "gputop" / "gputop.toml")
    paths.append(Path.home() / ".gputoprc")
    return tuple(paths)


def state_path(explicit: Path | None = None) -> Path:
    """Return the file the session state is read from and written to.

    Args:
        explicit: An explicit path, used verbatim.

    Resolution order is explicit path, then ``$GPUTOP_STATE``, then
    ``~/.config/gputop/config.toml``.  Resolved on every call rather than cached at
    import so that changing ``$HOME`` -- which the test suite does -- is honoured.
    """
    if explicit is not None:
        return explicit
    env_path = os.environ.get(STATE_ENV)
    if env_path:
        return Path(env_path)
    return DEFAULT_STATE_PATH.expanduser()


def _as_section(value: object) -> dict[str, Any]:
    """Coerce a TOML value into a mapping, treating anything else as empty."""
    return dict(value) if isinstance(value, dict) else {}


def build_section(
    cls: type[Any], values: dict[str, Any], where: str, warnings: list[str]
) -> Any:
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
    """Coerce types, clamp ranges and normalise enum-ish fields.

    Every comparison below is between a value that came from an untrusted TOML file and a
    number, so each section is first passed through :func:`_as_int`.  A frozen dataclass
    does not validate types -- ``interval_ms = "fast"`` arrives here intact -- and without
    that pass ``100 <= "fast"`` raises ``TypeError`` straight out of :func:`load_config`,
    which is the one thing this module promises never to do: a malformed file has to
    produce warnings and a working configuration, not a refusal to start.
    """
    general = _validated_general(config.general, warnings)
    process = _validated_process(config.process, warnings)
    ui = _validated_ui(config.ui, warnings)

    gpu = _validated_gpu(config.gpu, warnings)

    state = _validate_state(config.state, warnings)

    blocks = _validate_blocks(config.blocks, warnings)
    alerts = _validate_alerts(config.alerts, warnings)
    log = _validate_log(config.log, warnings)

    return replace(
        config,
        general=general,
        process=process,
        gpu=gpu,
        ui=ui,
        state=state,
        blocks=blocks,
        alerts=alerts,
        log=log,
    )


def _validated_general(general: GeneralConfig, warnings: list[str]) -> GeneralConfig:
    """Coerce and clamp the ``[general]`` section."""
    defaults = GeneralConfig()
    general = replace(
        general,
        interval_ms=_as_int(
            general.interval_ms, defaults.interval_ms, "general.interval_ms", warnings
        ),
        history_points=_as_int(
            general.history_points, defaults.history_points, "general.history_points", warnings
        ),
    )
    if not MIN_INTERVAL_MS <= general.interval_ms <= MAX_INTERVAL_MS:
        clamped = min(MAX_INTERVAL_MS, max(MIN_INTERVAL_MS, general.interval_ms))
        warnings.append(
            f"general.interval_ms={general.interval_ms} out of range, using {clamped}"
        )
        general = replace(general, interval_ms=clamped)
    if general.history_points < 1:
        warnings.append(f"general.history_points={general.history_points} below 1, using 1")
        general = replace(general, history_points=1)
    return general


def _validated_process(process: ProcessConfig, warnings: list[str]) -> ProcessConfig:
    """Coerce and clamp the ``[process]`` section."""
    defaults = ProcessConfig()
    process = replace(
        process,
        max_rows=_as_int(process.max_rows, defaults.max_rows, "process.max_rows", warnings),
        # Compared against every process' engine percentage, so a non-number here raised
        # TypeError from inside the filter on the first evaluation -- after start-up, with
        # no warning, and on every tick thereafter.
        min_usage_percent=_as_float(
            process.min_usage_percent,
            defaults.min_usage_percent,
            "process.min_usage_percent",
            warnings,
        ),
    )
    if process.max_rows < 1:
        warnings.append(f"process.max_rows={process.max_rows} below 1, using 1")
        process = replace(process, max_rows=1)
    return process


def _validated_ui(ui: UiConfig, warnings: list[str]) -> UiConfig:
    """Coerce and clamp the ``[ui]`` section."""
    defaults = UiConfig()
    ui = replace(
        ui,
        graph_height=_as_int(
            ui.graph_height, defaults.graph_height, "ui.graph_height", warnings
        ),
    )
    # Typed as Any because that is what it is: the section was built straight out of a TOML
    # document, so the dataclass's ``str`` annotation is a promise the loader has not yet
    # kept.  ``in`` against a tuple compares rather than hashes, so a TOML array here is
    # merely "unknown" -- but an inline table would raise, and this is the check that stops it.
    theme: Any = ui.theme
    if not isinstance(theme, str) or theme not in THEMES:
        warnings.append(
            f"ui.theme={ui.theme!r} unknown, using {DEFAULT_THEME_NAME!r} "
            f"(available: {', '.join(THEMES)})"
        )
        ui = replace(ui, theme=DEFAULT_THEME_NAME)
    # A negative height has no meaning -- Textual would reject it -- and ``0`` is the
    # documented "fill the space available" value, so the fallback is the one a user who
    # typed nonsense can still get a working interface from.
    if ui.graph_height < 0:
        warnings.append(f"ui.graph_height={ui.graph_height} below 0, using 0 (fill available)")
        ui = replace(ui, graph_height=0)
    return ui


def _validate_blocks(blocks: BlocksConfig, warnings: list[str]) -> BlocksConfig:
    """Clamp the radeontop options to what radeontop itself will accept.

    Clamping rather than rejecting is deliberate for the interval in particular: radeontop
    parses ``-i`` with ``atoi`` and silently floors it at 1 second, so a config asking for
    500 ms would otherwise be honoured by gputop and ignored by the child, producing a
    panel that updated four times less often than the user asked for with nothing saying so.
    """
    clamped = RadeontopOptions(
        enabled=bool(blocks.enabled),
        binary=str(blocks.binary) if blocks.binary else "radeontop",
        ticks=_as_int(blocks.ticks, 120, "blocks.ticks", warnings),
        interval_s=_as_int(blocks.interval_s, 1, "blocks.interval_s", warnings),
        # Tested as a string because a Path is always truthy: TOML can legitimately carry
        # `drm_root = ""` for "look in the default place", and `Path("")` is not that.
        drm_root=Path(blocks.drm_root) if str(blocks.drm_root) else Path("/dev/dri"),
        restart=bool(blocks.restart),
    ).validated()
    if clamped.interval_s != blocks.interval_s:
        warnings.append(
            f"blocks.interval_s={blocks.interval_s} out of range for radeontop "
            f"(whole seconds only), using {clamped.interval_s}"
        )
    if clamped.ticks != blocks.ticks:
        warnings.append(f"blocks.ticks={blocks.ticks} out of range, using {clamped.ticks}")
    if not blocks.binary:
        warnings.append("blocks.binary empty, using 'radeontop'")
    return replace(
        blocks,
        binary=clamped.binary,
        ticks=clamped.ticks,
        interval_s=clamped.interval_s,
        drm_root=clamped.drm_root,
    )


def _as_float(value: Any, default: float, where: str, warnings: list[str]) -> float:
    """Coerce a config value to a float, warning and falling back when it is not one.

    A TOML document is untrusted input and ``temp_c = "hot"`` is a typo someone will make.
    A frozen dataclass accepts it without complaint -- dataclasses do not validate types --
    so the value arrives here intact and would otherwise reach the comparison below and
    raise ``TypeError`` out of ``load_config``.  That would break the module's own contract,
    which is that a malformed file produces warnings and a working configuration rather than
    a refusal to start.

    ``bool`` is rejected explicitly: TOML's ``true`` is an ``int`` subclass, and
    ``alerts.temp_c = true`` becoming ``1.0`` is a silent, very confusing threshold.
    """
    if isinstance(value, bool):
        warnings.append(f"{where}: expected a number, got a boolean, using {default}")
        return default
    if isinstance(value, (int, float)):
        return float(value)
    warnings.append(f"{where}: expected a number, got {type(value).__name__}, using {default}")
    return default


def _as_int(value: Any, default: int, where: str, warnings: list[str]) -> int:
    """Coerce a config value to an int, warning and falling back when it is not one."""
    if isinstance(value, bool):
        warnings.append(f"{where}: expected an integer, got a boolean, using {default}")
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    warnings.append(
        f"{where}: expected an integer, got {type(value).__name__}, using {default}"
    )
    return default


def _as_str(value: Any, default: str, where: str, warnings: list[str]) -> str:
    """Coerce a config value to a string, warning and falling back when it is not one.

    A number is accepted, because ``process_filter = 42`` is a plausible thing to type and
    its intent -- filter on the text "42" -- is unambiguous.  Anything else is not: the
    filter is passed straight to ``str.strip`` and a list would raise an ``AttributeError``
    from inside the interface, some seconds after start-up, with no warning at all.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    warnings.append(
        f"{where}: expected a string, got {type(value).__name__}, using {default!r}"
    )
    return default


def _as_str_list(value: Any, where: str, warnings: list[str]) -> tuple[str, ...]:
    """Coerce a config value to a tuple of strings, warning about each bad element.

    A bare string is *not* accepted as a one-element list.  ``devices = "card0"`` is the
    kind of thing a user writes, and quietly reading it as ``["card0"]`` would be kind --
    except that it would also be the only list-shaped key in the file that accepts a bare
    scalar, so the next one they wrote would be rejected for no visible reason.

    Each element is reported against its own position: ``gpu.devices`` names the key, but
    it is the fourth entry that is the table, and a message naming only the section sends
    the reader looking in the wrong place.
    """
    if isinstance(value, str):
        warnings.append(f"{where}: expected a list of names, got a single string, ignoring it")
        return ()
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for position, item in enumerate(value):
            # A number is accepted on the same grounds as ``_as_str``: ``devices = [0, 1]``
            # cannot mean anything else, and dropping the whole list over it would be worse.
            if isinstance(item, str):
                text = item.strip()
                if text:
                    out.append(text)
                else:
                    warnings.append(f"{where}[{position}]: empty entry, ignoring it")
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                out.append(str(item))
            else:
                warnings.append(
                    f"{where}[{position}]: expected a name, got {type(item).__name__}, ignoring it"
                )
        return tuple(out)
    warnings.append(
        f"{where}: expected a list of names, got {type(value).__name__}, ignoring it"
    )
    return ()


def _validated_gpu(gpu: GpuConfig, warnings: list[str]) -> GpuConfig:
    """Coerce the ``[gpu]`` section.

    ``names`` is the one that bites: it goes to the sampler as a ``dict(...)``, so a
    mistyped table reached ``dict(5)`` and raised ``TypeError`` from inside
    ``Sampler.__init__`` -- after start-up, with no warning, replacing the monitor with a
    traceback.  Keys and values are both stringified, so ``names = {0 = "igpu"}`` from a
    bare integer key does the obvious thing rather than nothing.

    ``devices`` is the other untrusted shape: it is a TOML array that reaches the device
    matcher entry by entry, so a nested array or a table in one position would raise from
    inside discovery.  It is coerced here instead, and a non-string is reported against its
    own position so the message points at the element that is wrong rather than the section.
    """
    if gpu.kind not in ("auto", "igpu", "dgpu"):
        warnings.append(f"gpu.kind={gpu.kind!r} invalid, using 'auto'")
        gpu = replace(gpu, kind="auto")
    names: dict[str, str] = {}
    try:
        items = dict(gpu.names).items()
    except TypeError, ValueError:
        warnings.append(
            f"gpu.names: expected a table of PCI addresses, got "
            f"{type(gpu.names).__name__}, ignoring it"
        )
    else:
        for key, value in items:
            names[str(key)] = _as_str(value, str(key), "gpu.names value", warnings)
    return replace(gpu, devices=_as_str_list(gpu.devices, "gpu.devices", warnings), names=names)


def _validate_alerts(alerts: AlertsConfig, warnings: list[str]) -> AlertsConfig:
    """Coerce and clamp the thresholds.

    A temperature limit above the point where the part throttles hard is a limit that can
    never usefully fire, so it is clamped to 125 -- the same ceiling the sensor panel draws
    its temperature scale against, which keeps the two panels agreeing about what "full"
    means.  Percentages above 100 are clamped for the same reason.
    """
    clean = AlertsConfig(
        enabled=bool(alerts.enabled),
        temp_c=_as_float(alerts.temp_c, 90.0, "alerts.temp_c", warnings),
        power_percent=_as_float(alerts.power_percent, 95.0, "alerts.power_percent", warnings),
        vram_percent=_as_float(alerts.vram_percent, 92.0, "alerts.vram_percent", warnings),
        flash_hz=_as_float(alerts.flash_hz, 1.0, "alerts.flash_hz", warnings),
    )
    changed: dict[str, Any] = {}
    if clean.temp_c < 0 or clean.temp_c > MAX_TEMP_C:
        clamped = min(MAX_TEMP_C, max(0.0, clean.temp_c))
        warnings.append(f"alerts.temp_c={clean.temp_c:g} out of range, using {clamped:g}")
        changed["temp_c"] = clamped
    for name in ("power_percent", "vram_percent"):
        value = getattr(clean, name)
        if value < 0 or value > 100:
            clamped = min(100.0, max(0.0, value))
            warnings.append(f"alerts.{name}={value:g} out of range, using {clamped:g}")
            changed[name] = clamped
    if clean.flash_hz < 0 or clean.flash_hz > 20:
        clamped = min(20.0, max(0.0, clean.flash_hz))
        warnings.append(f"alerts.flash_hz={clean.flash_hz:g} out of range, using {clamped:g}")
        changed["flash_hz"] = clamped
    return replace(clean, **changed) if changed else clean


def _validate_log(log: LogConfig, warnings: list[str]) -> LogConfig:
    """Coerce and clamp the recording options.

    The zstd range is 1-19 with 0 meaning "library default"; anything above 19 is rejected
    by the library itself, and clamping here means a typo produces a working recording with
    a warning rather than a crash at the first write.
    """
    clean = LogConfig(
        zstd_level=_as_int(log.zstd_level, 3, "log.zstd_level", warnings),
        interval_s=_as_float(log.interval_s, 0.0, "log.interval_s", warnings),
    )
    changed: dict[str, Any] = {}
    if clean.zstd_level < 0 or clean.zstd_level > 19:
        clamped = 0 if clean.zstd_level < 0 else 19
        warnings.append(f"log.zstd_level={clean.zstd_level} out of range, using {clamped}")
        changed["zstd_level"] = clamped
    if clean.interval_s < 0:
        warnings.append(f"log.interval_s={clean.interval_s:g} below 0, using 0 (every sample)")
        changed["interval_s"] = 0.0
    return replace(clean, **changed) if changed else clean


def _validate_state(state: StateConfig, warnings: list[str]) -> StateConfig:
    """Reject a saved state that no longer matches this build.

    A state file outlives the release that wrote it, so every value is treated as
    untrusted input.  A theme or a sort column from a future version is dropped with a
    warning rather than being allowed to break start-up.
    """
    # Untrusted for the same reason as ui.theme above: a state file outlives the release
    # that wrote it, and both of these are strings by declaration rather than by provenance.
    theme: Any = state.theme
    sort: Any = state.sort
    if theme is not None and (not isinstance(theme, str) or theme not in THEMES):
        warnings.append(f"state.theme={state.theme!r} unknown, ignoring the saved value")
        state = replace(state, theme=None)
    if sort is not None and (not isinstance(sort, str) or sort not in SORT_COLUMNS):
        warnings.append(f"state.sort={state.sort!r} unknown, ignoring the saved value")
        state = replace(state, sort=None)
    # The filter text is the one state field that is not an enum, so there is nothing to
    # check it against -- but it is handed to str.strip by the table on the first repaint,
    # which is long after this function has returned and with no warning to point at the
    # file that caused it.
    clean_filter = _as_str(state.filter, "", "state.process_filter", warnings)
    if clean_filter != state.filter:
        state = replace(state, filter=clean_filter)
    interval = state.interval_ms
    if interval is not None:
        coerced = _as_int(interval, 0, "state.interval_ms", warnings)
        if coerced != interval:
            # A wrong type has already been reported; the placeholder it became carries no
            # further information, and warning twice about one mistake reads as noise.
            interval = None
        elif not MIN_INTERVAL_MS <= coerced <= MAX_INTERVAL_MS:
            warnings.append(
                f"state.interval_ms={interval} out of range, ignoring the saved value"
            )
            interval = None
    if interval != state.interval_ms:
        state = replace(state, interval_ms=interval)
    gpu_index = _as_int(state.gpu_index, 0, "state.gpu_index", warnings)
    if gpu_index < 0:
        warnings.append(f"state.gpu_index={state.gpu_index} below 0, using 0")
        gpu_index = 0
    if gpu_index != state.gpu_index:
        state = replace(state, gpu_index=gpu_index)
    return state


def apply_state(config: Config, state: StateConfig) -> Config:
    """Fold a saved session state over a hand-written configuration.

    Only the two values the rest of the program reads through the config are moved:
    the theme, which ``gputop --theme`` and every panel colour depend on, and the
    interval, which the sampler is constructed from.  Everything else in ``[state]``
    describes the interface, so the UI reads it from the state itself and applying it
    here would only create a second place for it to be wrong.

    Precedence, lowest to highest, is therefore: defaults, config file, saved state,
    command line -- and the CLI is applied after this, which is why a one-off
    ``gputop -i 200`` still wins over what the last session settled on.
    """
    general, ui = config.general, config.ui
    if state.interval_ms is not None:
        general = replace(general, interval_ms=state.interval_ms)
    if state.theme is not None:
        ui = replace(ui, theme=state.theme)
    return replace(config, general=general, ui=ui, state=state)


def load_config(path: Path | None = None, *, with_state: bool | None = None) -> Config:
    """Load configuration from TOML.

    Args:
        path: An explicit config file.  When ``None`` the search path is used and a
            missing file simply yields defaults.
        with_state: Whether to fold in the session state gputop saved for itself.
            Defaults to ``True`` for the search path and ``False`` when ``path`` is
            given: naming a file means naming the whole configuration, and layering a
            previous session on top of it would make a test -- or a scripted run --
            depend on whatever the last interactive session happened to leave behind.

    Returns:
        A validated :class:`Config`.
    """
    if with_state is None:
        with_state = path is None
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

    for name in raw:
        if name not in KNOWN_SECTIONS:
            warnings.append(f"{name}: unknown section ignored")

    config = Config(
        general=build_section(
            GeneralConfig, _as_section(raw.get("general")), "general", warnings
        ),
        gpu=build_section(GpuConfig, _as_section(raw.get("gpu")), "gpu", warnings),
        process=build_section(
            ProcessConfig, _as_section(raw.get("process")), "process", warnings
        ),
        ui=build_section(UiConfig, _as_section(raw.get("ui")), "ui", warnings),
        blocks=build_section(BlocksConfig, _as_section(raw.get("blocks")), "blocks", warnings),
        alerts=build_section(AlertsConfig, _as_section(raw.get("alerts")), "alerts", warnings),
        log=build_section(LogConfig, _as_section(raw.get("log")), "log", warnings),
        state=_state_section(_as_section(raw.get("state")), warnings),
    )
    config = _validate(config, warnings)
    if with_state:
        config = _merge_saved_state(config, chosen, warnings)
    return replace(config, warnings=tuple(warnings))


def _state_section(section: Mapping[str, Any], warnings: list[str]) -> StateConfig:
    """Build a :class:`StateConfig` from the ``[state]`` table *as written in the file*.

    ``[state]`` is the one section whose keys deliberately differ from its field names:
    the file says ``process_sort`` and ``process_filter`` where the dataclass says ``sort``
    and ``filter``, so that the file reads well next to ``[process]``.  The translation is
    :func:`gputop.state.table_to_fields`, and it has to be applied here as well as in
    ``load_state`` -- building the section from the raw keys instead matched nothing, and
    every key gputop itself writes came back as "unknown key ignored" with the session
    silently reset to its defaults.
    """
    from gputop.state import STATE_SECTION, table_to_fields

    return cast(
        StateConfig,
        build_section(StateConfig, table_to_fields(section), STATE_SECTION, warnings),
    )


def _merge_saved_state(config: Config, chosen: Path | None, warnings: list[str]) -> Config:
    """Overlay the state file's ``[state]`` section, if there is one.

    The state file is usually the same file as the hand-written config, so this used to
    reuse the document ``load_config`` had already parsed instead of reading it again.  It
    no longer does, and the reason is worth recording: the two sections are keyed
    differently.  ``gputop.state`` writes ``process_sort``, ``process_filter`` and
    ``show_idle_processes`` while the dataclass fields are ``sort``, ``filter`` and
    ``show_idle``, and only ``load_state`` applies that mapping.  Handing the raw section
    straight to ``build_section`` therefore matched on the wrong names and dropped every
    key gputop itself writes -- with an "unknown key ignored" warning for each, on the
    default configuration path, so a user's saved session was discarded on every start.

    Re-reading a file of a few hundred bytes once, at start-up, is not a cost worth a second
    code path that can disagree with the first.

    A malformed state file is a warning, not a failure: losing the last session's sort
    order is not a reason to refuse to start.

    The import is local because :mod:`gputop.state` needs this module for the section's
    schema, and a module-scope import in both directions would be a cycle.  The two are
    only ever used together from here.
    """
    from gputop.state import load_state

    path = state_path()
    del chosen  # kept in the signature for the caller's clarity about what it decided
    if not path.is_file():
        return config
    state = load_state(path)
    return apply_state(config, _validate_state(state, warnings))
