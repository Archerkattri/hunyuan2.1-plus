<div align="center">

# Hunyuan3D-2.1 + HiCache

**Tencent's Hunyuan3D-2.1 image/text-to-3D, accelerated by the HiCache Hermite velocity cache on its DiT flow-matching loop.**

*A clean, first-class integration of [HiCache](https://arxiv.org/abs/2508.16984) — training-free diffusion acceleration that forecasts the cached velocity with a scaled-Hermite **polynomial** basis instead of running the DiT on skipped denoise steps.*

![training&#8209;free](https://img.shields.io/badge/training--free-%E2%9C%93-2e8f5c)
&nbsp;![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)
&nbsp;![Hunyuan3D&#8209;2.1](https://img.shields.io/badge/base-Hunyuan3D--2.1-d96902)
&nbsp;![arXiv](https://img.shields.io/badge/HiCache-arXiv%3A2508.16984-b5212f?logo=arxiv)

</div>

## When to use this repo

These repos are **complementary accelerators, not competing solutions** — each speeds up a *different*
base generator, and the `+` / `++` suffix is a **method choice**, not a rival product. Pick by
**(1) which base model you run**, then **(2) which forecast basis you want**:

| base generator | `+` = HiCache (Hermite) | `++` = HiCache++ (DMD) |
|---|---|---|
| Hunyuan3D-2.1 | `hunyuan2.1-plus` | `hunyuan2.1-plus-plus` |
| Hunyuan3D-2 mini | `hunyuan2-plus` | `hunyuan2-plus-plus` |
| SAM 3D Objects | `sam3d-plus` | `sam3d-plus-plus` |
| Fast-SAM3D | `fastsam3d-plus` | `fastsam3d-plus-plus` |
| DiT-XL/2 (ImageNet) | `dit-plus` | `dit-plus-plus` |
| TRELLIS (v1) | `faster-trellis` | `faster-trellis-plus-plus` |
| TRELLIS.2-4B (v2) | `hermit-trellis2` | `hermit-trellis2-plus-plus` |

- **`+` (HiCache / scaled-Hermite):** the *published* polynomial velocity-forecast basis — conservative, reproduces the HiCache paper. Use it to deploy the established method.
- **`++` (HiCache++ / DMD exponential):** our Dynamic-Mode-Decomposition basis — *the same near-lossless quality at wider skip intervals*, where the polynomial diverges. Use it when you push the cache interval for more speed.
- **standalone / model-agnostic:** [`hicache-plus-plus`](https://github.com/Archerkattri/hicache-plus-plus) — the forecaster itself, to add DMD caching to *your own* diffusion/flow model.
- **`fast-trellis2`** = the TaylorSeer baseline fork (the upstream "Fast" accel) — the v2 reference point, not a HiCache variant.

> **This repo:** `hunyuan2.1-plus` — **Hunyuan3D-2.1 × HiCache (Hermite)** — the published polynomial cache, deeply integrated into Tencent's 2.1 image-to-3D.

---

## What this is

[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) (© Tencent) generates a textured 3D asset from a single image (or text) — a DiT **flow-matching** sampler denoises a latent shape, then a paint stage textures it. The shape sampler is the cost: it runs the DiT once per sampling step.

This fork adds **HiCache** to that loop. On most sampling steps the expensive `self.model(...)` forward is skipped and the (CFG-combined) flow-matching velocity is *forecast* from cached anchors — so HiCache computes the velocity only every `interval` steps and predicts the rest, skipping `(interval-1)/interval` of the DiT forwards. The forecaster is a **first-class part of the pipeline**: `Hunyuan3DDiTFlowMatchingPipeline.__call__` reads it natively, with **no runtime monkey-patching**. Training-free and geometry-preserving.

## Method — Hermite polynomial forecast

At each compute ("full") step HiCache updates backward finite-difference derivatives `Δ^i F_t` of the cached velocity. On a skipped step `k` steps later it forecasts

```
F̂_{t−k} = F_t + Σ_{i≥1} (Δ^i F_t / i!) · H̃_i(−k)
```

with the **dual-scaled physicist's Hermite** polynomial `H̃_n(x) = σ^n H_n(σ x)`, `σ ∈ (0,1)`. The `σ` contraction keeps the high-order terms bounded, giving a more stable extrapolation than the equivalent Taylor (monomial) series — TaylorSeer is exactly the special case where `H̃_i(−k)` is the monomial `(−k)^i`. With fewer than two anchors the forecast degenerates to plain reuse of the cached velocity (the correct zero-information forecast).

The Hermite basis is a **polynomial**, so it is the right call only at a **modest skip**: it is lossless at low intervals and degrades as the skip grows, because a polynomial diverges under extrapolation. The diffusion feature trajectory actually lives on a sum-of-exponentials (the solution of a near-linear feature-ODE), which a polynomial can only locally truncate. The sibling fork **[`hunyuan2.1-plus-plus`](../hunyuan2.1-plus-plus)** swaps in the *exponential* (DMD/Prony) forecaster that is exact on that class and holds quality at larger skip intervals.

## How to enable it

```python
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")

# Turn on HiCache: compute the DiT velocity every `interval` steps, Hermite-forecast the rest.
pipe.enable_hicache(
    interval=3,        # one DiT forward, then interval-1 forecasts
    max_order=1,       # highest finite-difference / Hermite order
    first_enhance=2,   # always compute the first few steps (warm-up)
    sigma=0.5,         # Hermite contraction in (0, 1)
)

mesh = pipe(image="assets/demo.png")[0]   # same call as upstream — caching is transparent

pipe.disable_hicache()   # back to the dense sampler
```

`enable_hicache` just records the schedule on the pipeline; the denoise loop in `hy3dshape/hy3dshape/pipelines.py` consumes it via `hicache_init / hicache_decide / hicache_update_derivatives / hicache_forecast` from [`hy3dshape/hy3dshape/hicache.py`](hy3dshape/hy3dshape/hicache.py). The Hermite core is CPU-testable with no GPU or model: `python -m hy3dshape.hy3dshape.hicache`.

## Results

On Hunyuan3D-2.1 (Toys4K, F-score@0.05, 3-seed), the Hermite polynomial cache is the **baseline** point of comparison: lossless only at low skip (≈ 0.88 at interval-3, ~1.7× faster) and dropping as the interval grows (≈ 0.74 at interval-5 vs an uncached ≈ 0.89). The *exponential* method that holds quality much further out — ≈ 0.83 at interval-5 — lives in the sibling fork **[`hunyuan2.1-plus-plus`](../hunyuan2.1-plus-plus)**.

For the full cross-model benchmarks (controlled forecast microbenchmark, Hunyuan3D-2.1, Hunyuan3D-2-mini, SAM3D, Fast-SAM3D) and the Hermite-vs-exponential tables, see the standalone library **[`hicache-plus-plus`](../hicache-plus-plus)**.

## Attribution

- **Base model:** [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) © Tencent — see [`PROJECT.md`](PROJECT.md) and the upstream license. All Hunyuan3D-2.1 code, weights, and trademarks belong to Tencent.
- **HiCache:** *HiCache: Training-free Acceleration of Diffusion Models via Hermite Polynomial Feature Forecasting* (arXiv:[2508.16984](https://arxiv.org/abs/2508.16984)). The Hermite forecaster here is a clean reimplementation; only the loop wiring is Hunyuan-specific.
- **TaylorSeer** — the monomial (Taylor) feature cache that HiCache's Hermite basis generalises.

## Weights & data

Model weights and demo/example assets are **not** committed to this repo — only the acceleration
architecture (code + integration). Download the base-model weights from the upstream project,
[Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1), per its instructions, and point the loader at them (see the code / upstream README). This
keeps the repository lightweight and avoids redistributing third-party weights.

## Citation

If you use this repository, please cite the base model and the acceleration method(s):

```bibtex
@misc{hunyuan3d2025hunyuan3d,
    title={Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material},
    author={Tencent Hunyuan3D Team},
    year={2025},
    eprint={2506.15442},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@misc{hunyuan3d22025tencent,
    title={Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation},
    author={Tencent Hunyuan3D Team},
    year={2025},
    eprint={2501.12202},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

```bibtex
@misc{hicache2025,
  title  = {HiCache: Training-free Acceleration of Diffusion Models via Hermite Polynomial Feature Forecasting},
  eprint = {2508.16984}, archivePrefix = {arXiv}, primaryClass = {cs.CV}, year = {2025}
}
```
