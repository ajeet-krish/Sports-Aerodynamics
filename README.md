# The Physics of Play

**Aerodynamics in Summer Sports & FIFA World Cup 2026**

CFD simulations built with [ΦFlow](https://github.com/tum-pbs/PhiFlow) (JAX backend), demonstrating core aerodynamic concepts through sports-inspired examples.

[**View the portfolio site →**](https://ajeetdash.github.io/Sports-Aerodynamics/)

## Project Overview

This project bridges computational fluid dynamics with sports science through four self-contained simulation modules:

| # | Module | Method | Key Result |
|---|--------|--------|-----------|
| 1 | **Shot Aerodynamics** | Navier-Stokes (phiflow, 256×128) | Magnus lift from rotating cylinder; knuckleball unsteady wake |
| 2 | **Tactical Positioning** | Navier-Stokes (phiflow, 248×186) | 26.1% drag reduction via wake shielding (overlapping fullback) |
| 3 | **Tactical Stress Fields** | Gaussian superposition + continuum mechanics | Two-phase flow interface & Von Mises yield detection in 7v7 |
| 4 | **Team Heatmaps** | Gaussian influence fields + Darcy permeability | Passing lane identification via influence thresholding |

## Simulation Details

### 1. Shot Aerodynamics — Magnus vs Knuckleball

A cylinder in crossflow at Re ≈ 2×10⁵, comparing a rotating case (ω = 10 rad/s) with a non-rotating case (ω = 0).

- **Domain**: 8.0 × 4.0 m, cylinder radius 0.3 m at (2.0, 2.0)
- **Grid**: 256 × 128, staggered (MAC)
- **Solver**: Semi-Lagrangian advection + CG pressure solve
- **Inlet**: Uniform 1.0 m/s from left (x⁻) boundary
- **Outputs**: `magnus_effect.mp4`, `magnus_streamlines.mp4`, `knuckleball_comparison.mp4`, `knuckleball_pressure.mp4`

### 2. Tactical Positioning — Formation Aerodynamics

Players modelled as rectangular bluff bodies in crossflow (10 m/s inlet). Three cases:

| Case | Configuration | Metric |
|------|-------------|--------|
| 1 — Isolated Winger | Single player at (2.0, 3.0) | Baseline drag |
| 2 — Midfield Press Wall | Three players at x = 2.0, y = {1.5, 3.0, 4.5} | Vorticity amplification |
| 3 — Overlapping Fullback | Two players in tandem at x = {2.0, 4.0}, y = 3.0 | 26.1% drag reduction |

- **Domain**: 8.0 × 6.0 m, player size 0.30 × 0.44 m
- **Grid**: 248 × 186, staggered (MAC)
- **Outputs**: `soccer_positioning.mp4`

### 3. Tactical Stress Fields — Continuum Mechanics

Players modelled as anisotropic, velocity-stretched Gaussian influence fields. Attackers advance from left to right while defenders drop deeper over 60 eased frames.

- **Pitch**: 10.5 × 6.8 m (scaled from full size)
- **Grid**: 336 × 218
- **Stress**: tanh((attack − defend) / 3.0), range [−1, +1]
- **Yield detection**: 8-neighbour local maxima in stress > 0.4
- **Outputs**: `stress_field.png`, `stress_field_continuum.mp4`

### 4. Team Heatmaps — Defensive Influence & Passing Lanes

Eleven players per formation (GK + 10 outfield) modelled as isotropic Gaussian influence zones. Passing lanes identified as regions where influence < threshold (0.5).

| Formation | Style | Central Zone Permeability |
|-----------|-------|--------------------------|
| 4‑2‑3‑1 (high block) | Dispersed, high press | 46% open |
| 5‑3‑2 (low block) | Compact, deep defence | 7% open |

- **Pitch**: 10.5 × 6.8 m, full pitch with markings
- **Grid**: 336 × 218
- **Influence**: σ_x = 0.80, σ_y = 0.55, threshold = 0.50
- **Outputs**: `team_heatmaps.png`, `defensive_collapse.mp4`

## Project Structure

```
src/
├── build_all.py       # Orchestrator — runs all simulations
├── shot_aero.py       # Magnus + Knuckleball (cylinder in crossflow)
├── tactical.py        # Tactical positioning + stress fields + team heatmaps
├── domain.py          # Domain constants (grid, bounds, dt)
├── utils.py           # Shared theme, paths, imports
└── __init__.py
assets/                # Generated MP4s and PNGs
docs/
├── index.html         # Landing page hub
├── theory.html        # Background fluid dynamics theory
├── shot.html          # Section 1: Aerodynamics of a Shot
├── tactical.html      # Section 2: Tactical Positioning
└── heatmaps.html      # Section 4: Team Heatmaps
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
- **Dark theme**: matplotlib `dark_background` with magma/inferno/plasma/coolwarm colormaps
- **Staggered → Centered**: `v @ CenteredGrid(...)` → `.vector[0|1].values.native("x", "y")`
