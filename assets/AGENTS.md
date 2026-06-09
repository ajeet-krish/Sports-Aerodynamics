# AGENTS.md — Project Reference

## Domain & Grid

| Property | Module 1 (Magnus) | Module 2 (Wake) |
|----------|-------------------|------------------|
| Domain | 8 × 4 (wide) | 4 × 8 (tall) |
| Grid | 256 × 128 | 128 × 256 |
| Obstacle | `geom.Sphere(x=2.0, y=2.0, radius=0.3)` | `geom.Box` — 0.3 × 0.5 (HW=0.15, HH=0.25) |
| Inflow | 1.0 m/s | 1.0 m/s |
| Steps | 120, save every 4 → 30 frames | 80, save every 2 → 40 frames |
| DT | 0.3 | 0.3 |

## ΦFlow 3.x API Notes

### Backend
```python
from phi.jax.flow import *   # JAX mode (NOT phi.set_backend)
```

### Scalar diffusion
```python
from phi.physics import diffuse
v = diffuse.explicit(v, NU, DT)    # NOT fluid.diffuse
```

### Obstacle rotation
```python
Obstacle(geometry, velocity=vec(x=0, y=0), angular_velocity=ω)
```
ΦFlow enforces rotating no-slip BC during `make_incompressible`.

### Pressure projection
```python
from phi.field import Solve
v, p = fluid.make_incompressible(v, obstacle)           # 2-value return
v, p = fluid.make_incompressible(v, obstacle, solve=Solve())  # explicit solve
```
In ΦFlow 3.x `make_incompressible` returns only `(velocity, pressure)` — not 5 values.

### Staggered → Centered extraction (for streamplots)
```python
vc = v @ CenteredGrid(0, x=256, y=128, bounds=BOX)
u = np.array(vc.vector[0].values.native("x", "y"))
v = np.array(vc.vector[1].values.native("x", "y"))
```

### Native array extraction
- `p.values.native("x", "y")` returns array of shape `(x_size, y_size)`
- `.native()` takes dimension names as separate positional args
- For matplotlib: `.T` is required (phiflow returns (x,y); imshow/streamplot expects (y,x))

### Face-based drag masks (Module 2)
Upstream/downstream face patches for `geom.Box` obstacles:
```python
up = (XX > cx - HW*1.5) & (XX < cx - HW*0.8) & (YY > cy - HH) & (YY < cy + HH)
dn = (XX > cx + HW*0.8) & (XX < cx + HW*1.5) & (YY > cy - HH) & (YY < cy + HH)
drag = p_np[up].mean() - p_np[dn].mean()
```

## matplotlib conventions
- `plt.style.use("dark_background")`
- `cmap="inferno"` for velocity, `cmap="magma"` for pressure
- `FuncAnimation` with `ax.clear()` for streamplots (simpler than removing collections)
- All animation exports use `writer="ffmpeg"`, `dpi=120-150`

## File structure
```
Sports Aerodynamics/
├── sports.ipynb         # Original notebook (messy, ~285K lines)
├── sports2.ipynb        # Clean notebook (11 cells, ~20KB)
├── assets/
│   ├── AGENTS.md
│   ├── magnus_effect.mp4
│   ├── magnus_streamlines.mp4
│   ├── wake_comparison.mp4
│   ├── wake_streamlines.mp4
│   └── drag_comparison.png
├── PLAN.md
└── README.md
```
