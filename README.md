# The Physics of Play

**Aerodynamics in Summer Sports & FIFA World Cup 2026**

Interactive CFD simulations built with [ΦFlow](https://github.com/tum-pbs/PhiFlow) (JAX backend), demonstrating core aerodynamic concepts through sports-inspired examples on a 128×256 Cartesian grid.

## Notebook Structure

The single notebook `physics_of_play.ipynb` contains **13 cells**:

| # | Cell | Description |
|---|------|-------------|
| 1–2 | **Introduction** | Project overview, Navier-Stokes equations, Reynolds number |
| 3–6 | **Setup** | Imports (`phi.jax.flow`), dark theme, domain config, viz & sim helpers |
| 7–9 | **Module 1** | Magnus effect — rotating cylinder in crossflow, pressure field + trajectory animation |
| 10–12 | **Module 2** | Wake interaction — single runner, inline drafting, echelon offset |
| 13 | **Conclusion** | Drag comparison, physics summary |

## Modules

1. **The World Cup Free-Kick** — A rotating cylinder in crossflow generates an asymmetric pressure field (Magnus effect), visualized as a 2-panel animation: pressure heatmap + reconstructed ball trajectory. Exports to `assets/magnus_effect.mp4`.

2. **The Peloton & Running Pack** — Velocity fields around athlete silhouettes (SVG path → binary mask) in three configurations: solo baseline, inline drafting, and echelon offset. Exports to `assets/wake_comparison.mp4`.

## Getting Started

```bash
uv sync
uv run jupyter notebook physics_of_play.ipynb
```

Or after venv activation:
```bash
source .venv/bin/activate
jupyter notebook physics_of_play.ipynb
```

## Requirements

- **Python** 3.12
- **uv** package manager ([install guide](https://docs.astral.sh/uv/#installation))
- **ffmpeg** — required for MP4 export (`brew install ffmpeg` on macOS)
- See `pyproject.toml` for full Python dependency list

## Outputs

Generated animations are saved to `assets/`. Run all notebook cells sequentially to produce them.

## Implementation Notes

- **Backend**: `from phi.jax.flow import *` — JAX on CPU (GPU optional)
- **Solver**: ΦFlow `fluid.make_incompressible()` with auto-CG solver
- **Obstacle rotation**: ΦFlow's `Obstacle(angular_velocity=ω)` handles the rotating no-slip BC natively
- **Dark theme**: matplotlib `dark_background` style with magma/inferno colormaps

See [`PLAN.md`](PLAN.md) for detailed technical decisions and API references.
