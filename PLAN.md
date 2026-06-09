# Project Plan — The Physics of Play

## Charter

A single, narrative-driven Jupyter notebook demonstrating core CFD concepts using ΦFlow with a JAX backend. High-contrast, dark-themed, publication-quality animations designed for a technical portfolio.

## Notebook Structure

```
physics_of_play.ipynb
├── 1. Title & Introduction               (markdown)
├── 2. Governing Equations                 (markdown — Navier-Stokes, continuity, Re)
├── 3. Imports & Backend                   (code — from phi.jax.flow import *)
├── 4. Domain & Style                      (code — grid, dark theme, rcParams)
├── 5. Visualization Helpers               (code — make_figure, save_animation)
├── 6. Simulation Helpers                  (code — run_simulation)
├── 7. Module 1: The World Cup Free-Kick   (markdown — header)
│   ├── 8. Magnus simulation               (code — run sim, collect frames + forces)
│   └── 9. Magnus animation + export       (code — FuncAnimation → magnus_effect.mp4)
├── 10. Module 2: The Peloton              (markdown — header)
│   ├── 11. Wake simulation (A/B/C)        (code — single, inline, echelon)
│   └── 12. Wake animation + export        (code — comparison → wake_comparison.mp4)
└── 13. Conclusion                         (markdown — drag table, summary)
```

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Python version | 3.12 | Highest version both ΦFlow and JAX officially support |
| Solver backend | JAX (CPU) | Performance on 128×256 grids, optional GPU support |
| Package manager | uv | Fast, deterministic, single-command setup |
| Visualization | matplotlib.animation.FuncAnimation | Dark theme, magma/inferno colormaps |
| Athlete shapes | SVG path → binary mask | Clean, swappable, zero binary bloat in repo |

## ΦFlow API Notes

### Backend selection
Use backend-specific import (NOT `phi.set_backend()`):
```python
from phi.jax.flow import *   # JAX mode
# from phi.flow import *     # NumPy fallback
```

### StaggeredGrid constructor
```python
StaggeredGrid((u, v), boundary_dict, x=nx, y=ny, bounds=geom.Box(...))
```
- 1st arg: tuple of velocity components (inflow values)
- 2nd arg: dict of boundary conditions e.g. `{'x-': vec(x=1, y=0), 'x+': ZERO_GRADIENT, 'y': 0}`
- `x=`, `y=`: grid resolution in cells
- `bounds=`: physical domain extent

### Obstacle rotation
ΦFlow's `Obstacle(geometry, velocity=vec(x, y), angular_velocity=ω)` accepts `angular_velocity` directly — no manual tangential-velocity override needed. The solver enforces the rotating no-slip BC during `make_incompressible`.

### Pressure projection
Default `Solve()` (auto-detect CG) works reliably. For closed domains add `rank_deficiency=1`:
```python
v, p = fluid.make_incompressible(v, obstacle)           # open domain
v, p = fluid.make_incompressible(v, obstacle, Solve(rank_deficiency=1))  # closed
```
The `scipy-direct` solver may diverge — prefer default auto-CG.

### Extracting numpy arrays from fields
```python
p_np = p.values.native('x', 'y')
vel_mag = field.vec_length(v)
vel_np = vel_mag.values.native('x', 'y')
```

## Athlete Silhouette (SVG Path → Binary Mask)

1. Hard-code SVG `<path d="..." />` string for runner profile
2. Parse with `svg.path.parse_path(d_string)` → list of Bezier/Line segments
3. Rasterize to binary mask at simulation grid resolution
4. Wrap as `phi.geom.Geometry` for ΦFlow obstacle

Fallback: geometric primitives (ellipse + rectangles) if SVG parsing proves fragile.

## Project Structure

```
Sports Aerodynamics/
├── .python-version          # 3.12
├── pyproject.toml           # uv-managed dependencies
├── README.md                # Project overview + setup guide
├── PLAN.md                  # This file
├── .gitignore
├── assets/                  # Generated .mp4/.gif output
│   ├── magnus_effect.mp4
│   └── wake_comparison.mp4
└── physics_of_play.ipynb    # Single monolithic notebook
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `phiflow` | >=3.4.0 | Differentiable PDE solver (incompressible Navier-Stokes) |
| `jax[cpu]` | — | JAX backend for ΦFlow (GPU optional) |
| `matplotlib` | — | Dark-theme animations, FuncAnimation → MP4 |
| `numpy` | — | Array ops, mask construction, force integration |
| `tqdm` | — | Progress bars in notebook |
| `svg.path` | — | SVG path parsing for athlete silhouettes |
| `jupyter` | — | Notebook runtime |
| `ffmpeg` | system | MP4 encoding (via matplotlib writer) |

## Execution

```bash
uv sync
uv run jupyter notebook physics_of_play.ipynb
```

Or after activation:
```bash
source .venv/bin/activate
jupyter notebook physics_of_play.ipynb
```
