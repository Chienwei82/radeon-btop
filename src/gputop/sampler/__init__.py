"""Background sampling.

The only thing the UI is allowed to do with a :class:`Sampler` is consume immutable
snapshots.  All mutable state lives inside the sampler thread.
"""

from gputop.sampler.engine import DEFAULT_INTERVAL_S, Sampler, SamplerOptions

__all__ = ["DEFAULT_INTERVAL_S", "Sampler", "SamplerOptions"]
