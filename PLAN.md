# Project Plan — The Physics of Play

## Charter

A single, narrative-driven Jupyter notebook demonstrating core CFD concepts using ΦFlow with a JAX backend. High-contrast, dark-themed, publication-quality animations designed for a technical portfolio.

## Notebook Structure

```
sports2.ipynb               # Clean notebook (15 cells, ~21KB)
sports.ipynb                # Original notebook (kept for reference, ~285K lines)
```

### Cell outline

| # | Type | Content |
|---|------|---------|
| 0 | md | Title & Introduction |
| 1 | md | Governing Equations (Navier-Stokes, continuity, Re) |
| 2 | code | Imports (`from phi.jax.flow import *`), dark theme, rcParams |
| 3 | md | Module 1 intro — Magnus effect, 8×4 domain, 256×128 grid |
| 4 | code | Module 1 simulation — rotating cylinder `geom.Sphere`, 120 steps |
| 5 | code | Module 1 pressure animation → `magnus_effect.mp4` |
| 6 | code | Module 1 streamlines animation → `magnus_streamlines.mp4` |
| 7 | md | Module 3 intro — Knuckleball vs Magnus, same 8×4 domain |
| 8 | code | Module 3 simulation — two runs: ω=10 and ω=0, 120 steps each |
| 9 | code | Module 3 2-panel velocity+streamlines → `knuckleball_comparison.mp4` |
| 10 | code | Module 3 2-panel pressure comparison → `knuckleball_pressure.mp4` |
| 11 | md | Module 2 intro — Peloton drafting, 4×8 domain, 128×256 grid |
| 12 | code | Module 2 simulation — 3 cases (solo/inline/echelon), rectangles |
| 13 | code | Module 2 3-panel velocity animation → `wake_comparison.mp4` |
| 14 | code | Module 2 3-panel streamlines animation → `wake_streamlines.mp4` |

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Python version | 3.12 | Highest version both ΦFlow and JAX officially support |
| Solver backend | JAX (CPU) | Performance on grids, optional GPU support |
| Package manager | uv | Fast, deterministic, single-command setup |
| Visualization | matplotlib FuncAnimation | Dark theme, magma/inferno colormaps |
| Module 1 grid | 256 × 128 (8×4 domain) | Wide rectangle, flow left→right, higher streamwise resolution |
| Module 2 grid | 128 × 256 (4×8 domain) | Tall rectangle for drafting configuration |
| Module 3 grid | 256 × 128 (8×4 domain) | Same as Module 1 (soccer ball cross-section) |
| Runner shape | `geom.Box` 0.3×0.5 | Rectangular torso cross-section |

## ΦFlow API Notes

See `assets/AGENTS.md` for comprehensive API reference.

### Quick reference
- **Import**: `from phi.jax.flow import *`
- **Scalar diffusion**: `diffuse.explicit(v, NU, DT)` from `phi.physics`
- **Pressure projection**: `v, p = fluid.make_incompressible(v, obstacle)` — 2 return values
- **Centered extraction**: `v @ CenteredGrid(0, x=256, y=128, bounds=BOX)` then `.vector[0|1].values.native("x", "y")`
- **Transpose required**: `.T` for all arrays passed to matplotlib

## Project Structure

```
Sports Aerodynamics/
├── .python-version            # 3.12
├── pyproject.toml             # uv-managed dependencies
├── README.md                  # Project overview + setup guide
├── PLAN.md                    # This file
├── .gitignore
├── build_notebook.py          # Generator script for sports2.ipynb
├── sports.ipynb               # Original notebook (reference, ~285K lines)
├── sports2.ipynb              # Clean notebook (11 cells)
├── assets/
│   ├── AGENTS.md              # API reference for future sessions
│   ├── magnus_effect.mp4
│   ├── magnus_streamlines.mp4
│   ├── knuckleball_comparison.mp4
│   ├── knuckleball_pressure.mp4
│   ├── wake_comparison.mp4
│   ├── wake_streamlines.mp4
│   └── drag_comparison.png
└── .venv/                     # Virtual environment
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `phiflow` | >=3.4.0 | Differentiable PDE solver (incompressible Navier-Stokes) |
| `jax[cpu]` | — | JAX backend for ΦFlow (GPU optional) |
| `matplotlib` | — | Dark-theme animations, FuncAnimation → MP4 |
| `numpy` | — | Array ops, mask construction, force integration |
| `tqdm` | — | Progress bars in notebook |
| `jupyter` | — | Notebook runtime |
| `ffmpeg` | system | MP4 encoding (via matplotlib writer) |

## Execution

```bash
uv sync
uv run jupyter notebook sports2.ipynb
```

Or after activation:
```bash
source .venv/bin/activate
jupyter notebook sports2.ipynb
```
