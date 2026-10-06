#compdef gputop
#
# zsh completion for gputop.
#
# Installed as /usr/share/zsh/site-functions/_gputop, so compinit picks it up from the
# fpath without any further setup.
#
# The keys gputop uses once the interface is up -- q, tab, 1-9, m, p, R, the process
# table's s/S/r/a/t/c/k -- are NOT listed here.  Completion runs before the interface
# exists, so there is nothing to complete them against, and a second copy of the key table
# would be a second thing to forget when a binding changes.  They are documented in
# gputop(1), which ships in the same package; run "man gputop" and read the KEYS section.

_gputop_themes=(default dracula gruvbox monochrome radeon)
_gputop_kinds=(auto igpu dgpu)

_gputop() {
    local curcontext="$curcontext" state ret=1

    _arguments -s -S \
        '--version[print the version and exit]' \
        {-c+,--config=}'[configuration file to load instead of searching the default locations]:configuration file:_files -g "*.toml"' \
        {-i+,--interval=}'[sampling interval, overriding general.interval_ms]:interval (seconds): ' \
        '--drm-root=[sysfs DRM class directory (default /sys/class/drm)]:DRM class directory:_files -/' \
        '--proc-root=[procfs mount point (default /proc)]:procfs mount point:_files -/' \
        '--dump[write one JSON snapshot of the live machine to stdout and exit]' \
        '--devices[list the discovered GPUs as JSON and exit]' \
        '--kind=[force every device to be classified as integrated or discrete]:device kind:_values "kind" auto igpu dgpu' \
        '--theme=[colour theme]:colour theme:_values "theme" default dracula gruvbox monochrome radeon' \
        '--no-color[disable 24-bit colour and fall back to terminal ANSI colours]' \
        '--no-processes[skip the /proc scan, the most expensive part of a sample]' \
        '--pretty[indent --dump and --devices output for human reading]' \
        '--blocks[force the per-block panel on, overriding blocks.enabled]' \
        '--no-blocks[never start radeontop, even when it is installed]' \
        '--log=[record the session; .csv or .json, optionally .zst]:recording:_files -g "*.csv *.json *.jsonl *.ndjson *.zst *.zstd"' \
        '(--json)'--check='[report which metrics this machine can provide, and why any are missing]' \
        '(--check)'--json='[emit the --check report as JSON instead of text]' \
        '1: :_gputop_note' \
        '*:: :->args'

    case "$state" in
        args)
            _arguments -s : \
                '(- *)'{-h,--help}'[show the help message and exit]'
            ;;
    esac

    return ret
}

# Nothing to complete for a bare argument: gputop takes options only, and a file name in
# that position is a mistake worth a pointer rather than a silent match.
_gputop_note() {
    if (( CURRENT == 1 )); then
        _message 'interactive keys are documented in gputop(1)'
    else
        _message 'gputop takes options only; see gputop(1)'
    fi
}

if [ "$funcstack[1]" = "_gputop" ]; then
    _gputop "$@"
else
    compdef _gputop gputop
fi