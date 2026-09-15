"""Hermite-only compatibility facade for Hunyuan3D-2.1.

The native pipeline and adaptive-CFG module remain the baseline integration;
the shared corrected-sign Hermite state, schedule, reset, and telemetry come
from ``hicache-pp``. DMD belongs to the sibling ``hunyuan2.1-plus-plus`` arm.
"""

try:
    from hicache_pp.hermite import (
        hicache_decide,
        hicache_forecast,
        hicache_init as _central_hicache_init,
        hicache_reset,
        hicache_telemetry,
        hicache_update_derivatives,
        physicists_hermite,
        scaled_hermite,
    )
except ImportError as exc:  # pragma: no cover - installation failure path
    raise ImportError(
        "hunyuan2.1-plus requires hicache-pp>=1.2.1; install requirements.txt"
    ) from exc


def hicache_init(num_steps, interval=4, max_order=1, first_enhance=2,
                 end_enhance=None, sigma=0.5, backend="hermite", history=5):
    """Create a Hermite cache and reject non-baseline backends explicitly."""
    if backend != "hermite":
        raise ValueError(
            "hunyuan2.1-plus supports only backend='hermite'; use the ++ sibling for DMD"
        )
    return _central_hicache_init(
        num_steps=num_steps, interval=interval, max_order=max_order,
        first_enhance=first_enhance, end_enhance=end_enhance, sigma=sigma,
        backend="hermite", history=history,
    )


__all__ = [
    "hicache_decide", "hicache_forecast", "hicache_init", "hicache_reset",
    "hicache_telemetry", "hicache_update_derivatives", "physicists_hermite",
    "scaled_hermite",
]


if __name__ == "__main__":
    import torch

    state = hicache_init(num_steps=6, interval=3, first_enhance=0)
    sample = torch.ones(2)
    for step in range(3):
        state["step"] = step
        if hicache_decide(state) == "full":
            hicache_update_derivatives(state, sample)
        else:
            hicache_forecast(state)
    assert hicache_telemetry(state)["decisions"]["full"] > 0
    print("hunyuan2.1-plus Hermite smoke passed")
