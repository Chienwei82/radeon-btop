# bash completion for gputop
#
# Installed as /usr/share/bash-completion/completions/gputop and loaded by
# bash-completion, so it is sourced automatically on Bash 4.1 and later.
#
# The keys gputop uses once the interface is up are not completed here: a completion
# function cannot see what is on screen, and a key table in a completion file is a second
# copy of the one in gputop(1) that drifts.  They are documented in the man page, which
# ships in the same package.

_gputop_themes="default dracula gruvbox"
_gputop_kinds="auto igpu dgpu"

_gputop_long_opts="--version --config --interval --drm-root --proc-root --dump --devices
                  --kind --theme --no-color --no-processes --pretty --blocks --no-blocks
                  --log --check --json"

# _gputop_files VALUE [PATTERN ...]
#
# Complete paths for VALUE, keeping only those matching one of PATTERN when patterns are
# given.  An empty result falls back to every file: a filter that returns nothing is worse
# than an imprecise one, because a recording may be called anything.
_gputop_files() {
    local value="$1"
    shift
    local -a matches=()
    local -a keep=()
    local candidate pattern

    COMPREPLY=( $(compgen -f -- "$value") )
    [ $# -eq 0 ] && return 0

    for candidate in "${COMPREPLY[@]}"; do
        for pattern in "$@"; do
            case "$candidate" in
                $pattern) keep+=("$candidate"); break ;;
            esac
        done
    done

    [ ${#keep[@]} -gt 0 ] && COMPREPLY=( "${keep[@]}" )
    return 0
}

_gputop() {
    local cur prev opt

    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # "--theme=dracula" is one argument, but "=" is in COMP_WORDBREAKS, so bash also offers
    # the value as a word of its own.  Split the two spellings back apart here so that both
    # go through the same case below.
    case "$cur" in
        --*=*) opt="${cur%%=*}"; cur="${cur#*=}" ;;
        *)     opt="" ;;
    esac

    case "$opt" in
        --config)
            _gputop_files "$cur" '*.toml'
            return 0
            ;;
        --log)
            # The suffix before .zst/.zstd picks the format and the trailing one picks the
            # compression, so session.csv and session.csv.zst are both legal targets.
            _gputop_files "$cur" '*.csv' '*.json' '*.jsonl' '*.ndjson' '*.zst' '*.zstd'
            return 0
            ;;
        --drm-root|--proc-root)
            COMPREPLY=( $(compgen -d -- "$cur") )
            return 0
            ;;
        --theme)
            COMPREPLY=( $(compgen -W "$_gputop_themes" -- "$cur") )
            return 0
            ;;
        --kind)
            COMPREPLY=( $(compgen -W "$_gputop_kinds" -- "$cur") )
            return 0
            ;;
        --interval)
            # A number of seconds.  There is nothing to complete, and offering numbers
            # would be offering the wrong ones: the useful values are the step sizes the
            # "+" and "-" keys use, which the man page states.
            return 0
            ;;
    esac

    case "$prev" in
        -c|--config)
            _gputop_files "$cur" '*.toml'
            return 0
            ;;
        --log)
            _gputop_files "$cur" '*.csv' '*.json' '*.jsonl' '*.ndjson' '*.zst' '*.zstd'
            return 0
            ;;
        --drm-root|--proc-root)
            COMPREPLY=( $(compgen -d -- "$cur") )
            return 0
            ;;
        --theme)
            COMPREPLY=( $(compgen -W "$_gputop_themes" -- "$cur") )
            return 0
            ;;
        --kind)
            COMPREPLY=( $(compgen -W "$_gputop_kinds" -- "$cur") )
            return 0
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        # Long options only.  "-h" and "--help" are left to the program: argparse prints
        # them, they are not a stable interface, and a completion that offers them cannot
        # promise what they say.
        COMPREPLY=( $(compgen -W "$_gputop_long_opts" -- "$cur") )
        return 0
    fi

    return 0
}

complete -F _gputop gputop