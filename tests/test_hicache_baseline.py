"""CPU tests for the Hunyuan3D-2.1 Hermite reference arm."""

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("hunyuan21_baseline_hicache", ROOT / "hy3dshape/hy3dshape/hicache.py")
hicache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hicache)


def test_central_core_identity_and_hermite_only_scope():
    import hicache_pp.hermite as central

    assert hicache.hicache_forecast is central.hicache_forecast
    assert hicache.scaled_hermite is central.scaled_hermite
    assert hicache.hicache_init(4)["backend"] == "hermite"
    with pytest.raises(ValueError, match="only backend='hermite'"):
        hicache.hicache_init(4, backend="dmd")


def test_baseline_and_no_cache_traces():
    for interval, expected_full in ((3, 2), (1, 6)):
        state = hicache.hicache_init(6, interval=interval, first_enhance=0)
        calls = 0
        outputs = []
        for step in range(6):
            state["step"] = step
            if hicache.hicache_decide(state) == "full":
                calls += 1
                value = torch.ones(2)
                hicache.hicache_update_derivatives(state, value)
            else:
                value = hicache.hicache_forecast(state)
            outputs.append(value)
            state["step"] += 1
        assert calls == expected_full
        assert all(torch.isfinite(value).all() for value in outputs)
        assert hicache.hicache_telemetry(state)["decisions"]["full"] == expected_full


def test_reset_isolates_two_seed_runs_and_cfg_formula():
    state_a = hicache.hicache_init(6, interval=3, first_enhance=0)
    run_a = state_a["run_id"]
    hicache.hicache_reset(state_a)
    state_b = hicache.hicache_init(6, interval=3, first_enhance=0)
    assert state_a["run_id"] != run_a
    assert state_b["run_id"] != state_a["run_id"]
    assert state_a["derivatives"] == {}
    assert hicache.hicache_telemetry(state_a)["decisions"] == {"full": 0, "forecast": 0}

    cond, uncond, scale = torch.tensor([3.0]), torch.tensor([1.0]), 5.0
    assert torch.equal(uncond + scale * (cond - uncond), torch.tensor([11.0]))


def test_adaptive_cfg_real_step_index_is_independent_of_cache_counter():
    # The pipeline writes i (the diffusion index) into adaptive-CFG state, while
    # HiCache owns its separate cadence counter. This is the identity contract.
    adaptive = {"step": 0, "anchors": [], "last_gamma": None, "n_full": 0, "n_skip": 0}
    cache = hicache.hicache_init(8, interval=3, first_enhance=0)
    for i in (0, 1, 2, 5):
        adaptive["step"] = i
        cache["step"] = i
        hicache.hicache_decide(cache)
        adaptive["anchors"].append((i, torch.tensor([float(i)])))
    assert [step for step, _ in adaptive["anchors"]] == [0, 1, 2, 5]
    assert cache["counter"] != adaptive["step"]
