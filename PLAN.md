# Project Plan — The Physics of Play

## Charter

A single, narrative-driven Jupyter notebook demonstrating core CFD concepts using ΦFlow with a JAX backend. High-contrast, dark-themed, publication-quality animations designed for a technical portfolio.

## Notebook Structure

```
physics_of_play.ipynb
├── 1. Title & Introduction
├── 2. Governing Equations (Navier-Stokes, continuity)
├── 3. Setup (imports, domain config, visualization engine)
├── 4. Module 1: The World Cup Free-Kick (Magnus Effect)
│   ├── Rotating cylinder in crossflow
│   ├── Side-by-side: pressure field + trajectory
│   └── Export: assets/magnus_effect.mp4
├── 5. Module 2: The Peloton & Running Pack (Wake Interaction)
│   ├── Case A: Single runner (baseline)
│   ├── Case B: Inline drafting
│   ├── Case C: Echelon offset
│   └── Export: assets/wake_comparison.mp4
├── 6. Conclusion (drag comparison, physics summary)
```

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Python version | 3.12 | Highest version both ΦFlow and JAX officially support |
| Solver backend | JAX (CPU) | Performance on 128×256 grids, optional GPU support |
| Package manager | uv | Fast, deterministic, single-command setup |
| Visualization | matplotlib.animation.FuncAnimation | Dark theme, magma/inferno colormaps |
| Athlete shapes | SVG path → binary mask | Clean, swappable, zero binary bloat in repo |

## Athlete Silhouette (SVG Path to Binary Mask)

1. Hard-code SVG `<path d="..." />` string for runner profile
2. Parse with `svg.path.parse_path(d_string)` → list of Bezier/Line segments
3. Rasterize to binary mask at simulation grid resolution
4. Wrap as `phi.geom.Geometry` for ΦFlow obstacle

Fallback: geometric primitives (ellipse + rectangles) if SVG parsing proves fragile.

## Magnus Effect — Rotating Cylinder BC

ΦFlow's `Obstacle(geometry, velocity=vec(x, y))` sets a uniform velocity on obstacle cells. To model rotation:
1. Each timestep, compute tangential velocity `v_t = ω × (r - r_center)` at each cell inside the cylinder
2. Override velocity field in obstacle region with this rotational field
3. `make_incompressible` enforces it as the no-slip boundary condition

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

- `phiflow>=3.4.0` — Differentiable PDE solver (ΦFlow)
- `jax[cpu]` — JAX backend for ΦFlow
- `matplotlib` — Animation and plotting
- `numpy` — Array operations
- `tqdm` — Progress bars

## Execution

```bash
uv sync
jupyter notebook physics_of_play.ipynb
```
