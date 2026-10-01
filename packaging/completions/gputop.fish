# fish completion for gputop
#
# Installed as /usr/share/fish/vendor_completions.d/gputop.fish, which fish loads at
# startup for every shell.
#
# The keys gputop uses once the interface is up are not listed here.  Completion happens
# before the interface exists, so there is nothing to complete them against, and a second
# copy of the key table is a second thing to forget when a binding changes.  They are
# documented in gputop(1), which ships in the same package: "man gputop", section KEYS.

# gputop takes options only, so the file completion fish does for plain arguments is
# noise.  Individual options that do take a path turn it back on below.
complete -c gputop -f

# Options that take no value.
complete -c gputop -l version  -d 'print the version and exit'
complete -c gputop -l dump     -d 'write one JSON snapshot of the live machine to stdout and exit'
complete -c gputop -l devices  -d 'list the discovered GPUs as JSON and exit'
complete -c gputop -l no-color -d 'disable 24-bit colour and fall back to terminal ANSI colours'
complete -c gputop -l no-processes -d 'skip the /proc scan, the most expensive part of a sample'
complete -c gputop -l pretty   -d 'indent --dump and --devices output for human reading'
complete -c gputop -l blocks   -d 'force the per-block panel on, overriding blocks.enabled'
complete -c gputop -l no-blocks -d 'never start radeontop, even when it is installed'
complete -c gputop -l check    -d 'report which metrics this machine can provide, and why any are missing'
complete -c gputop -l json     -d 'emit the --check report as JSON instead of text'

# Options with a closed set of values.  -x keeps fish from also offering file names,
# which would be offering a value gputop rejects.
complete -c gputop -l theme -x -a 'default dracula gruvbox' -d 'colour theme'
complete -c gputop -l kind  -x -a 'auto igpu dgpu' -d 'force every device to be classified as integrated or discrete'

# Options with a free-form number.  No candidates: the useful values are the step sizes
# the "+" and "-" keys use, and guessing between them is worse than saying nothing.
complete -c gputop -s i -l interval -r -d 'sampling interval in seconds, overriding general.interval_ms'

# Options with a path.  -F turns file completion back on for the argument.
#
# A config file is TOML but its name is not constrained by anything, so .toml files are
# suggested and any other file is still offered.  A recording is the opposite case: the
# suffix picks the format, and the trailing .zst/.zstd picks compression, so the
# suggestions are exclusive and the wrong ones are not worth showing.
complete -c gputop -s c -l config -r -F -a '*.toml' -d 'configuration file to load instead of searching the default locations'
complete -c gputop -l log -r -x -a '*.csv *.json *.jsonl *.ndjson *.zst *.zstd' -d 'record the session; .csv or .json, optionally .zst'

# Root overrides name directories, not files.
complete -c gputop -l drm-root  -r -x -a '(__fish_complete_directories)' -d 'sysfs DRM class directory (default /sys/class/drm)'
complete -c gputop -l proc-root -r -x -a '(__fish_complete_directories)' -d 'procfs mount point (default /proc)'