# The Physics of Play

**Aerodynamics in Summer Sports & FIFA World Cup 2026**

CFD simulations built with [ΦFlow](https://github.com/tum-pbs/PhiFlow) (JAX backend), demonstrating core aerodynamic concepts through sports-inspired examples.

[**View the portfolio site →**](https://ajeetdash.github.io/Sports-Aerodynamics/)

## Simulations

| # | Simulation | Key Result | Output |
|---|-----------|-----------|--------|
| 1 | **Magnus Effect** — rotating cylinder (ω=10), 256×128 grid, 160 steps | Asymmetric pressure field produces lift | `magnus_effect.mp4`, `magnus_streamlines.mp4` |
| 2 | **Knuckleball vs Magnus** — ω=10 vs ω=0 side-by-side | Spin stabilises wake, reduces pressure drag | `knuckleball_comparison.mp4`, `knuckleball_pressure.mp4` |
| 3 | **Tactical Positioning** — 3 formation cases, 248×186 grid, 200 steps | 26.1% drag reduction for overlapping fullback | `soccer_positioning.mp4` |
| 4 | **Tactical Continuum** — 7v7 Gaussian stress field, 60 frames | Attack/defence pressure front visualisation | `tactical_continuum.mp4` |

## Project Structure

```
src/
├── build_all.py       # Orchestrator — runs all simulations
├── shot_aero.py       # Magnus + Knuckleball (cylinder in crossflow)
├── tactical.py        # Tactical positioning + macro stress field
├── domain.py          # Domain constants (grid, bounds, dt)
├── utils.py           # Shared theme, paths, imports
└── __init__.py
assets/                # Generated MP4s
docs/
└── index.html         # GitHub Pages portfolio site
sports draft.ipynb     # Experimentation notebook (unchanged)
```

## Getting Started

```bash
uv sync
python src/build_all.py       # Run all simulations
uv run jupyter notebook       # Explore interactively
```

## Requirements

- **Python** 3.12
- **uv** package manager
- **ffmpeg** — MP4 export (`brew install ffmpeg`)
- **ΦFlow** (≥3.4), JAX, matplotlib, numpy, tqdm

## Implementation Notes

- **Backend**: `phi.jax.flow` — JAX on CPU (GPU optional)
- **Solver**: `fluid.make_incompressible()` with iterative CG
- **Obstacle rotation**: `Obstacle(angular_velocity=ω)` for rotating BC
- **Dark theme**: matplotlib `dark_background` with magma/inferno colormaps
- **Staggered → Centered**: `v @ CenteredGrid(...)` → `.vector[0|1].values.native("x", "y")`

See [`assets/AGENTS.md`](assets/AGENTS.md) for ΦFlow API reference.
