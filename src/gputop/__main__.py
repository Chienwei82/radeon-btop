"""Entry point for ``python -m gputop``."""

from gputop.cli import main

# Guarded rather than raised at module scope: ``gputop.__main__:main`` is also the console
# script, and an unguarded ``raise SystemExit(main())`` would run the CLI as an import side
# effect -- anything that imported this module for the ``main`` symbol would launch the
# monitor instead of getting the function.
if __name__ == "__main__":
    raise SystemExit(main())
