# The Physics of Play

**Aerodynamics in Summer Sports & FIFA World Cup 2026**

Interactive CFD simulations built with [ΦFlow](https://github.com/tum-pbs/PhiFlow) (JAX backend), demonstrating core aerodynamic concepts through sports-inspired examples.

## Notebook Structure

Two notebooks are available:

| Notebook | Description | Size |
|----------|-------------|------|
| `sports2.ipynb` | **Working version** — 15 cells, all executed | ~530 KB (with outputs) |
| `sports.ipynb` | Original reference (kept for history) | ~666 KB |

### Cell outline (sports2.ipynb)

| # | Cell | Description |
|---|------|-------------|
| 0–1 | **Introduction** | Project overview, Navier-Stokes equations, Reynolds number |
| 2 | **Setup** | Imports (`phi.jax.flow`), dark theme, output directory |
| 3–6 | **Module 1** | Magnus effect — rotating cylinder in crossflow, pressure + streamlines animation |
| 7–10 | **Module 2** | Knuckleball vs Magnus — spinning and non-spinning cylinder, velocity + pressure comparison |
| 11–14 | **Module 3** | Wake interaction — single runner, inline drafting, echelon offset |

## Modules

1. **The World Cup Free-Kick (Magnus Effect)** — A rotating cylinder (8×4 domain, 256×128 grid) generates an asymmetric pressure field (Magnus effect), visualized as pressure animation + velocity streamlines. Exports to `assets/magnus_effect.mp4` and `assets/magnus_streamlines.mp4`.

2. **The Knuckleball (No-Spin Wake)** — Side-by-side comparison of a rotating (ω=10 rad/s) and non-rotating cylinder (ω=0) in an 8×4 domain, 256×128 grid. Reveals how spin stabilises the wake, delays flow separation, and alters the pressure field. Exports to `assets/knuckleball_comparison.mp4` (velocity + streamlines) and `assets/knuckleball_pressure.mp4` (pressure field).

3. **The Peloton & Running Pack** — Velocity fields around rectangular runner cross-sections (4×8 domain, 128×256 grid) in three configurations: solo baseline, inline drafting, and echelon offset. Includes drag comparison table. Exports to `assets/wake_comparison.mp4`, `assets/wake_streamlines.mp4`, and `assets/drag_comparison.png`.

## Getting Started

```bash
uv sync
uv run jupyter notebook sports2.ipynb
```

Or after venv activation:
```bash
source .venv/bin/activate
jupyter notebook sports2.ipynb
```

## Requirements

- **Python** 3.12
- **uv** package manager ([install guide](https://docs.astral.sh/uv/#installation))
- **ffmpeg** — required for MP4 export (`brew install ffmpeg` on macOS)
- See `pyproject.toml` for full Python dependency list

## Implementation Notes

- **Backend**: `from phi.jax.flow import *` — JAX on CPU (GPU optional)
- **Solver**: ΦFlow `fluid.make_incompressible()` with auto-CG solver
- **Obstacle rotation**: ΦFlow's `Obstacle(angular_velocity=ω)` handles the rotating no-slip BC natively
- **Dark theme**: matplotlib `dark_background` style with magma/inferno colormaps
- **Staggered → Centered**: `v @ CenteredGrid(...)` then `.vector[0|1].values.native("x", "y")`

See [`assets/AGENTS.md`](assets/AGENTS.md) for detailed ΦFlow API reference.
