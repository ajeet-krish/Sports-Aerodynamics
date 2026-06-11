"""Tactical Positioning — 3 formation cases + 7v7 Macro Stress Field + 11v11 Team Heatmaps."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phi import geom
from phi.jax.flow import Obstacle, StaggeredGrid, CenteredGrid, fluid, advect, vec, field, math, ZERO_GRADIENT

from src.utils import setup_theme, ASSETS

setup_theme()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse, Circle, Rectangle
from tqdm import trange


# ── Domain & boundary conditions ──
BOX = geom.Box(x=8.0, y=6.0)
BOUNDARY = {
    "x-": vec(x=10.0, y=0.0),
    "x+": ZERO_GRADIENT,
    "y": 0,
}

NX, NY = 248, 186

# ── Player shape ──
HW, HH = 0.15, 0.22

# ── Solver parameters ──
DT = 0.03
N_STEPS = 200
SAVE_EVERY = 4
CENTERED = CenteredGrid(0, x=NX, y=NY, bounds=BOX)


def player_at(cx, cy):
    return geom.Box(x=(cx - HW, cx + HW), y=(cy - HH, cy + HH))


# ── Drag masks ──
x_g = np.linspace(0, 8.0, NX)
y_g = np.linspace(0, 6.0, NY)
XX, YY = np.meshgrid(x_g, y_g, indexing="ij")


def face_drag_mask(cx, cy):
    front = (XX > cx - HW * 1.5) & (XX < cx - HW * 0.8) & (YY > cy - HH) & (YY < cy + HH)
    rear  = (XX > cx + HW * 0.8) & (XX < cx + HW * 1.5) & (YY > cy - HH) & (YY < cy + HH)
    return front, rear


def box_drag(p_np, cx, cy):
    front, rear = face_drag_mask(cx, cy)
    return float(p_np[front].mean() - p_np[rear].mean()) if front.sum() and rear.sum() else 0.0


def inject_scalar_near_inlet(scalar):
    s_np = np.array(scalar.values.native("x,y"))
    s_np[:5, :] = 1.0
    return CenteredGrid(math.tensor(s_np, math.spatial("x,y")), bounds=BOX, extrapolation=ZERO_GRADIENT)


def run_case(obstacles, label, drag_pos=None, with_scalar=False):
    v0 = StaggeredGrid((10.0, 0.0), BOUNDARY, x=NX, y=NY, bounds=BOX)
    v, _ = fluid.make_incompressible(v0, ())

    scalar = None
    if with_scalar:
        s_init = np.zeros((NX, NY))
        s_init[:5, :] = 1.0
        scalar = CenteredGrid(math.tensor(s_init, math.spatial("x,y")), bounds=BOX, extrapolation=ZERO_GRADIENT)

    frames = []
    drag_series = []
    v = v0

    for i in trange(N_STEPS, desc=label):
        v = advect.semi_lagrangian(v, v, dt=DT)
        v, p = fluid.make_incompressible(v, obstacles)

        if with_scalar and scalar is not None:
            scalar = advect.semi_lagrangian(scalar, v, dt=DT)
            scalar = inject_scalar_near_inlet(scalar)

        if i % SAVE_EVERY == 0:
            vc = v @ CENTERED
            u_c = np.array(vc.vector[0].values.native("x,y"))
            v_c = np.array(vc.vector[1].values.native("x,y"))
            vel_mag = field.vec_length(v)
            vel_np = np.array(vel_mag.values.native("x,y"))
            p_np = np.array(p.values.native("x,y"))
            vort = field.curl(v)
            vort_np = np.array(vort.values.native("x,y"))

            frame = {
                "velocity": vel_np, "u": u_c, "v": v_c,
                "pressure": p_np, "vorticity": vort_np, "step": i,
            }
            if with_scalar and scalar is not None:
                frame["scalar"] = np.array(scalar.values.native("x,y"))
            frames.append(frame)

            if drag_pos:
                drag_series.append(box_drag(p_np, *drag_pos))

    avg_drag = np.mean(drag_series[-len(drag_series) // 5:]) if drag_series else 0.0
    return frames, avg_drag, drag_series


def animate_tactical(frames_c1, frames_c2, frames_c3, drag_c1, drag_c3, pct_reduction):
    n_frames4 = min(len(frames_c1), len(frames_c2), len(frames_c3))
    trim = lambda f: f[:n_frames4]
    frames_c1, frames_c2, frames_c3 = trim(frames_c1), trim(frames_c2), trim(frames_c3)

    vmin_v, vmax_v = 0, 12
    vort_all = np.concatenate([f["vorticity"].ravel() for f in frames_c2])
    vmin_w, vmax_w = np.percentile(vort_all, 2), np.percentile(vort_all, 98)
    p_all = np.concatenate([f["pressure"].ravel() for f in frames_c3])
    vmin_p, vmax_p = p_all.min(), p_all.max()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#111111")
    fig.suptitle("Tactical Soccer Positioning — Aerodynamic Comparison", color="white", fontsize=15, y=0.98)

    titles = [
        "Case 1 — Isolated Winger (Baseline)",
        "Case 2 — Midfield Press Wall",
        "Case 3 — Overlapping Fullback",
    ]
    frames_list = [frames_c1, frames_c2, frames_c3]
    cmaps = ["inferno", "RdBu_r", "magma"]
    players_list = [
        [(2.0, 3.0)],
        [(2.0, 1.5), (2.0, 3.0), (2.0, 4.5)],
        [(2.0, 3.0), (4.0, 3.0)],
    ]
    imgs = []

    for ax, title, frames, cmap, players in zip(axes, titles, frames_list, cmaps, players_list):
        ax.set_facecolor("#111111")
        ax.set_title(title, color="white", fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 6)

        if frames is frames_c1:
            data = frames[0]["velocity"]
            im = ax.imshow(data.T, origin="lower", cmap=cmap, aspect="auto", extent=[0, 8, 0, 6], vmin=vmin_v, vmax=vmax_v)
            ax.streamplot(np.linspace(0, 8, NX), np.linspace(0, 6, NY),
                          frames[0]["u"].T, frames[0]["v"].T, color="white", linewidth=0.6, density=1.2, arrowsize=0.5)
        elif frames is frames_c2:
            data = frames[0]["vorticity"]
            im = ax.imshow(data.T, origin="lower", cmap=cmap, aspect="auto", extent=[0, 8, 0, 6], vmin=vmin_w, vmax=vmax_w)
        else:
            data = frames[0]["pressure"]
            im = ax.imshow(data.T, origin="lower", cmap=cmap, aspect="auto", extent=[0, 8, 0, 6], vmin=vmin_p, vmax=vmax_p)
            s = frames[0].get("scalar")
            if s is not None:
                ax.contour(np.linspace(0, 8, NX), np.linspace(0, 6, NY), s.T,
                           levels=[0.1, 0.5, 0.9], colors="cyan", linewidths=0.5, alpha=0.4)

        for cx, cy in players:
            ax.add_patch(Ellipse((cx, cy), width=HW*2, height=HH*2, color="white", fill=False, lw=1.5))
        imgs.append(im)

    axes[2].text(4.5, 5.3, f"Drag Reduction:\n{pct_reduction:.0f}%", color="lime", fontsize=13, weight="bold",
                 bbox=dict(facecolor="#111111", edgecolor="lime", alpha=0.8, pad=4))
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    cbar_ax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
    cbar = fig.colorbar(imgs[0], cax=cbar_ax)
    cbar.set_label("Velocity (m/s)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    def update(idx):
        for ax in axes:
            ax.clear()
            ax.set_facecolor("#111111")
            ax.set_xlim(0, 8)
            ax.set_ylim(0, 6)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")

        ax = axes[0]
        ax.set_title(titles[0], color="white", fontsize=10)
        ax.imshow(frames_c1[idx]["velocity"].T, origin="lower", cmap="inferno", aspect="auto", extent=[0, 8, 0, 6], vmin=vmin_v, vmax=vmax_v)
        ax.streamplot(np.linspace(0, 8, NX), np.linspace(0, 6, NY),
                      frames_c1[idx]["u"].T, frames_c1[idx]["v"].T, color="white", linewidth=0.6, density=1.2, arrowsize=0.5)
        for cx, cy in players_list[0]:
            ax.add_patch(Ellipse((cx, cy), width=HW*2, height=HH*2, color="white", fill=False, lw=1.5))

        ax = axes[1]
        ax.set_title(titles[1], color="white", fontsize=10)
        ax.imshow(frames_c2[idx]["vorticity"].T, origin="lower", cmap="RdBu_r", aspect="auto", extent=[0, 8, 0, 6], vmin=vmin_w, vmax=vmax_w)
        for cx, cy in players_list[1]:
            ax.add_patch(Ellipse((cx, cy), width=HW*2, height=HH*2, color="white", fill=False, lw=1.5))

        ax = axes[2]
        ax.set_title(titles[2], color="white", fontsize=10)
        ax.imshow(frames_c3[idx]["pressure"].T, origin="lower", cmap="magma", aspect="auto", extent=[0, 8, 0, 6], vmin=vmin_p, vmax=vmax_p)
        s = frames_c3[idx].get("scalar")
        if s is not None:
            ax.contour(np.linspace(0, 8, NX), np.linspace(0, 6, NY), s.T,
                       levels=[0.1, 0.5, 0.9], colors="cyan", linewidths=0.5, alpha=0.4)
        for cx, cy in players_list[2]:
            ax.add_patch(Ellipse((cx, cy), width=HW*2, height=HH*2, color="white", fill=False, lw=1.5))
        ax.text(4.5, 5.3, f"Drag Reduction:\n{pct_reduction:.0f}%", color="lime", fontsize=13, weight="bold",
                bbox=dict(facecolor="#111111", edgecolor="lime", alpha=0.8, pad=4))

        fig.suptitle(f"Tactical Soccer Positioning — Step {frames_c1[idx]['step']}  (t = {frames_c1[idx]['step'] * DT:.2f}s)",
                     color="white", fontsize=14, y=0.98)
        return imgs

    anim = FuncAnimation(fig, update, frames=n_frames4, interval=80, blit=False)
    path = ASSETS / "soccer_positioning.mp4"
    anim.save(str(path), writer="ffmpeg", fps=10, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def run_tactical():
    case1 = [Obstacle(player_at(2.0, 3.0), velocity=vec(x=0.0, y=0.0))]
    case2 = [
        Obstacle(player_at(2.0, 1.5), velocity=vec(x=0.0, y=0.0)),
        Obstacle(player_at(2.0, 3.0), velocity=vec(x=0.0, y=0.0)),
        Obstacle(player_at(2.0, 4.5), velocity=vec(x=0.0, y=0.0)),
    ]
    case3 = [
        Obstacle(player_at(2.0, 3.0), velocity=vec(x=0.0, y=0.0)),
        Obstacle(player_at(4.0, 3.0), velocity=vec(x=0.0, y=0.0)),
    ]

    print("Case 1 — Isolated winger (baseline)")
    frames_c1, drag_c1, _ = run_case(case1, "Case 1 (winger)", (2.0, 3.0))

    print("\nCase 2 — Midfield press wall")
    frames_c2, _, _ = run_case(case2, "Case 2 (press wall)")

    print("\nCase 3 — Overlapping fullback (with scalar tracer)")
    frames_c3, drag_c3, _ = run_case(case3, "Case 3 (fullback)", (4.0, 3.0), with_scalar=True)

    n = min(len(frames_c1), len(frames_c2), len(frames_c3))
    frames_c1, frames_c2, frames_c3 = frames_c1[:n], frames_c2[:n], frames_c3[:n]

    drag_ratio = drag_c3 / drag_c1 if drag_c1 != 0 else 0.0
    pct_reduction = (1 - drag_ratio) * 100

    print(f"\n{'=' * 50}")
    print(f"Baseline drag (Case 1 — isolated winger):       {drag_c1:.4f}")
    print(f"Trailing drag (Case 3 — overlapping fullback):   {drag_c3:.4f}")
    print(f"Drag ratio:                                      {drag_ratio:.3f}")
    print(f"Drag reduction:                                  {pct_reduction:.1f}%")
    print(f"{'=' * 50}")

    animate_tactical(frames_c1, frames_c2, frames_c3, drag_c1, drag_c3, pct_reduction)
    return frames_c1, frames_c2, frames_c3


# ── Macro Stress Field (Continuum Mechanics) ──

PITCH_W_S, PITCH_H_S = 10.5, 6.8
NX_S, NY_S = 336, 218
N_FRAMES = 60
SIGMA_X, SIGMA_Y = 0.50, 0.35
VEL_STRETCH = 0.12

_SX = PITCH_W_S / 10.0
_SY = PITCH_H_S / 7.0


def gaussian_stress(xx, yy, cx, cy, vx, vy, sx_base, sy_base, stretch):
    speed = np.sqrt(vx**2 + vy**2) + 1e-8
    theta = np.arctan2(vy, vx)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    sx = sx_base + stretch * speed
    sy = sy_base
    a = cos_t**2 / sx**2 + sin_t**2 / sy**2
    b = sin_t * cos_t * (1 / sx**2 - 1 / sy**2)
    c = sin_t**2 / sx**2 + cos_t**2 / sy**2
    dx = xx - cx
    dy = yy - cy
    return np.exp(-0.5 * (a * dx**2 + 2 * b * dx * dy + c * dy**2))


def detect_yield_points(stress_field, threshold=0.4):
    h, w = stress_field.shape
    mx = np.ones_like(stress_field, dtype=bool)
    mx[1:-1, 1:-1] &= stress_field[1:-1, 1:-1] >= stress_field[:-2, 1:-1]
    mx[1:-1, 1:-1] &= stress_field[1:-1, 1:-1] >= stress_field[2:, 1:-1]
    mx[1:-1, 1:-1] &= stress_field[1:-1, 1:-1] >= stress_field[1:-1, :-2]
    mx[1:-1, 1:-1] &= stress_field[1:-1, 1:-1] >= stress_field[1:-1, 2:]
    mx[1:-1, 1:-1] &= stress_field[1:-1, 1:-1] >= stress_field[:-2, :-2]
    mx[1:-1, 1:-1] &= stress_field[1:-1, 1:-1] >= stress_field[2:, 2:]
    mx[1:-1, 1:-1] &= stress_field[1:-1, 1:-1] >= stress_field[:-2, 2:]
    mx[1:-1, 1:-1] &= stress_field[1:-1, 1:-1] >= stress_field[2:, :-2]
    mx &= stress_field > threshold
    return np.argwhere(mx)


def run_macro_stress():
    x = np.linspace(0, PITCH_W_S, NX_S)
    y = np.linspace(0, PITCH_H_S, NY_S)
    XX, YY = np.meshgrid(x, y, indexing="ij")

    attack_start = np.array([
        [2.0, 3.5], [1.5, 5.2], [1.5, 1.8], [1.2, 4.2], [1.2, 2.8], [0.5, 5.8], [0.5, 1.2],
    ]) * [_SX, _SY]
    attack_end = np.array([
        [6.0, 3.5], [5.8, 5.8], [5.8, 1.2], [4.5, 4.5], [4.5, 2.5], [2.5, 5.8], [2.5, 1.2],
    ]) * [_SX, _SY]
    defend_start = np.array([
        [6.0, 3.5], [6.0, 4.8], [6.0, 2.2], [6.8, 3.5], [6.8, 5.5], [6.8, 1.5], [7.8, 3.5],
    ]) * [_SX, _SY]
    defend_end = np.array([
        [4.2, 3.5], [4.2, 4.5], [4.2, 2.5], [5.0, 3.5], [5.0, 5.2], [5.0, 1.8], [6.2, 3.5],
    ]) * [_SX, _SY]

    t_frac = np.linspace(0, 1, N_FRAMES)
    eased = np.sin(np.pi * t_frac / 2)

    attack_pos = attack_start[np.newaxis] + eased[:, np.newaxis, np.newaxis] * (attack_end - attack_start)[np.newaxis]
    defend_pos = defend_start[np.newaxis] + eased[:, np.newaxis, np.newaxis] * (defend_end - defend_start)[np.newaxis]
    attack_vel = np.gradient(attack_pos, axis=0)
    defend_vel = np.gradient(defend_pos, axis=0)

    print("Computing stress fields...")
    fields = []
    yield_points = []
    for frame in trange(N_FRAMES):
        attack_field = np.zeros((NX_S, NY_S))
        defend_field = np.zeros((NX_S, NY_S))
        for p in range(7):
            ax, ay = attack_pos[frame, p, 0], attack_pos[frame, p, 1]
            avx, avy = attack_vel[frame, p, 0], attack_vel[frame, p, 1]
            attack_field += gaussian_stress(XX, YY, ax, ay, avx, avy, SIGMA_X, SIGMA_Y, VEL_STRETCH)
            dx, dy = defend_pos[frame, p, 0], defend_pos[frame, p, 1]
            dvx, dvy = defend_vel[frame, p, 0], defend_vel[frame, p, 1]
            defend_field += gaussian_stress(XX, YY, dx, dy, dvx, dvy, SIGMA_X, SIGMA_Y, VEL_STRETCH)
        stress = np.tanh((attack_field - defend_field) / 3.0)
        fields.append(stress)
        yield_points.append(detect_yield_points(stress, threshold=0.4))

    vmin, vmax = -1.0, 1.0
    levels = np.linspace(vmin, vmax, 40)

    # ── Static 2-Panel ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#111111")
    fig.suptitle("Tactical Stress Fields — Continuum Mechanics of 7v7",
                 color="white", fontsize=14, y=0.98)

    ax0 = axes[0]
    ax0.set_facecolor("#111111")
    draw_pitch(ax0)
    contour = ax0.contourf(x, y, fields[0].T, levels=levels,
                           cmap="coolwarm", vmin=vmin, vmax=vmax, extend="both")
    ax0.contour(x, y, fields[0].T, levels=[0.0], colors="white", linewidths=2.0)
    ax0.scatter(attack_pos[0, :, 0], attack_pos[0, :, 1], c="white", s=35,
                edgecolors="#e74c3c", linewidths=1.2, zorder=5)
    ax0.scatter(defend_pos[0, :, 0], defend_pos[0, :, 1], c="white", s=35,
                edgecolors="#3498db", linewidths=1.2, zorder=5)
    ax0.set_xlim(0, PITCH_W_S)
    ax0.set_ylim(0, PITCH_H_S)
    ax0.set_aspect("equal")
    ax0.set_title("Stress Field Interface", color="white", fontsize=11)
    ax0.text(0.02, 0.98, "Two-phase flow interface — zero-stress contour is the equilibrium\n"
                         "boundary where attacking and defensive stress fields balance",
             color="white", fontsize=8, alpha=0.85, transform=ax0.transAxes, va="top")
    ax0.text(0.02, 0.04, "Attack ○   Defence ○", color="white", fontsize=8,
             transform=ax0.transAxes, va="bottom")

    ax1 = axes[1]
    ax1.set_facecolor("#111111")
    draw_pitch(ax1)
    ax1.contourf(x, y, fields[0].T, levels=levels,
                 cmap="coolwarm", vmin=vmin, vmax=vmax, extend="both")
    ax1.contour(x, y, fields[0].T, levels=[0.0], colors="white", linewidths=2.0)
    ax1.quiver(attack_pos[0, :, 0], attack_pos[0, :, 1],
               attack_vel[0, :, 0], attack_vel[0, :, 1],
               color="#e74c3c", scale=1.0, width=0.015, alpha=0.9)
    ax1.quiver(defend_pos[0, :, 0], defend_pos[0, :, 1],
               defend_vel[0, :, 0], defend_vel[0, :, 1],
               color="#3498db", scale=1.0, width=0.015, alpha=0.9)

    if len(yield_points[0]) > 0:
        yx = x[yield_points[0][:, 0]]
        yy_p = y[yield_points[0][:, 1]]
        ax1.scatter(yx, yy_p, c="none", s=80, marker="X",
                    edgecolors="#f1c40f", linewidths=1.5, zorder=6)

    ax1.set_xlim(0, PITCH_W_S)
    ax1.set_ylim(0, PITCH_H_S)
    ax1.set_aspect("equal")
    ax1.set_title("Vectors & Von Mises Yield", color="white", fontsize=11)
    ax1.text(0.02, 0.98, "Von Mises yield criterion — stress concentrations (yellow X)\n"
                         "indicate localised structural failure in the defensive block",
             color="white", fontsize=8, alpha=0.85, transform=ax1.transAxes, va="top")

    fig.tight_layout(rect=[0, 0, 0.92, 0.93])
    cbar_ax = fig.add_axes([0.93, 0.12, 0.012, 0.76])
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.set_label("Stress σ = Attack − Defence", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    path = ASSETS / "stress_field.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="#111111")
    print(f"Saved {path}")
    plt.close(fig)

    # ── Animation ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#111111")
    fig.suptitle("Tactical Stress Fields — 7v7 Continuum Mechanics",
                 color="white", fontsize=14, y=0.98)

    def update(idx):
        for ax in axes:
            ax.clear()
            ax.set_facecolor("#111111")

        ax0 = axes[0]
        draw_pitch(ax0)
        ax0.contourf(x, y, fields[idx].T, levels=levels,
                     cmap="coolwarm", vmin=vmin, vmax=vmax, extend="both")
        ax0.contour(x, y, fields[idx].T, levels=[0.0], colors="white", linewidths=2.0)
        ax0.scatter(attack_pos[idx, :, 0], attack_pos[idx, :, 1], c="white", s=35,
                    edgecolors="#e74c3c", linewidths=1.2, zorder=5)
        ax0.scatter(defend_pos[idx, :, 0], defend_pos[idx, :, 1], c="white", s=35,
                    edgecolors="#3498db", linewidths=1.2, zorder=5)
        ax0.set_xlim(0, PITCH_W_S)
        ax0.set_ylim(0, PITCH_H_S)
        ax0.set_aspect("equal")
        pct = idx / (N_FRAMES - 1) * 100
        ax0.set_title(f"Frame {idx} — Interface Evolution ({pct:.0f}%)", color="white", fontsize=11)

        ax1 = axes[1]
        draw_pitch(ax1)
        ax1.contourf(x, y, fields[idx].T, levels=levels,
                     cmap="coolwarm", vmin=vmin, vmax=vmax, extend="both")
        ax1.contour(x, y, fields[idx].T, levels=[0.0], colors="white", linewidths=2.0)
        ax1.quiver(attack_pos[idx, :, 0], attack_pos[idx, :, 1],
                   attack_vel[idx, :, 0], attack_vel[idx, :, 1],
                   color="#e74c3c", scale=1.0, width=0.015, alpha=0.9)
        ax1.quiver(defend_pos[idx, :, 0], defend_pos[idx, :, 1],
                   defend_vel[idx, :, 0], defend_vel[idx, :, 1],
                   color="#3498db", scale=1.0, width=0.015, alpha=0.9)

        if len(yield_points[idx]) > 0:
            yx = x[yield_points[idx][:, 0]]
            yy_p = y[yield_points[idx][:, 1]]
            ax1.scatter(yx, yy_p, c="none", s=80, marker="X",
                        edgecolors="#f1c40f", linewidths=1.5, zorder=6)

        ax1.set_xlim(0, PITCH_W_S)
        ax1.set_ylim(0, PITCH_H_S)
        ax1.set_aspect("equal")
        ax1.set_title(f"Frame {idx} — Yield & Breach Points", color="white", fontsize=11)

        fig.suptitle(f"Tactical Stress Fields — Frame {idx}/{N_FRAMES - 1}  (t = {idx * 0.1:.1f}s)",
                     color="white", fontsize=14, y=0.98)
        return axes

    anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=80, blit=False)
    path = ASSETS / "stress_field_continuum.mp4"
    anim.save(str(path), writer="ffmpeg", fps=12, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# Section 2 — Team Heatmaps (11v11)
# ═══════════════════════════════════════════════════════════

PITCH_W_T, PITCH_H_T = 10.5, 6.8
NX_T, NY_T = 336, 218
SIGMA_T_X, SIGMA_T_Y = 0.80, 0.55
INFLUENCE_THRESHOLD = 0.50
N_COLLAPSE_FRAMES = 60

# ── Formation definitions ──
# All 11 positions: GK first, then outfield

F4231_HIGH = np.array([
    [10.0, 3.4],
    [8.0,  0.4],
    [8.5,  2.0],
    [8.5,  4.8],
    [8.0,  6.4],
    [7.0,  1.2],
    [7.0,  5.6],
    [6.0,  0.2],
    [6.0,  3.4],
    [6.0,  6.6],
    [4.5,  3.4],
])

F4231_LOW = np.array([
    [10.2, 3.4],
    [9.0,  1.6],
    [9.2,  2.6],
    [9.2,  4.2],
    [9.0,  5.2],
    [8.2,  2.2],
    [8.2,  4.6],
    [7.5,  1.6],
    [7.5,  3.4],
    [7.5,  5.2],
    [6.0,  3.4],
])

F532 = np.array([
    [10.2, 3.4],
    [8.0,  1.4],
    [9.0,  2.2],
    [9.0,  3.4],
    [9.0,  4.6],
    [8.0,  5.4],
    [8.0,  2.0],
    [8.0,  3.4],
    [8.0,  4.8],
    [6.5,  2.4],
    [6.5,  4.4],
])


def draw_pitch(ax, color="#555555", lw=0.6):
    PW, PH = PITCH_W_T, PITCH_H_T

    ax.plot([0, PW, PW, 0, 0], [0, 0, PH, PH, 0], color=color, lw=lw)

    ax.axvline(PW / 2, color=color, lw=lw)

    ax.add_patch(Circle((PW / 2, PH / 2), 0.915, color=color, fill=False, lw=lw))

    pa_w, pa_h = 1.65, 4.03
    pa_y = (PH - pa_h) / 2
    ax.add_patch(Rectangle((0, pa_y), pa_w, pa_h, color=color, fill=False, lw=lw))
    ax.add_patch(Rectangle((PW - pa_w, pa_y), pa_w, pa_h, color=color, fill=False, lw=lw))

    ga_w, ga_h = 0.55, 1.83
    ga_y = (PH - ga_h) / 2
    ax.add_patch(Rectangle((0, ga_y), ga_w, ga_h, color=color, fill=False, lw=lw))
    ax.add_patch(Rectangle((PW - ga_w, ga_y), ga_w, ga_h, color=color, fill=False, lw=lw))

    goal_w, goal_h = 0.15, 0.732
    goal_y = (PH - goal_h) / 2
    ax.add_patch(Rectangle((-goal_w, goal_y), goal_w, goal_h, color=color, fill=False, lw=lw * 1.5))
    ax.add_patch(Rectangle((PW, goal_y), goal_w, goal_h, color=color, fill=False, lw=lw * 1.5))


def compute_influence_field(xx, yy, positions, sigma_x=SIGMA_T_X, sigma_y=SIGMA_T_Y):
    field = np.zeros_like(xx)
    for cx, cy in positions:
        field += np.exp(-0.5 * ((xx - cx)**2 / sigma_x**2 + (yy - cy)**2 / sigma_y**2))
    return field


def compute_permeability(field, threshold=INFLUENCE_THRESHOLD):
    return np.sum(field < threshold) / field.size * 100


# ── Bucket C: Static 2-Panel Comparison ──

def plot_team_heatmaps():
    print("── Team Heatmaps — Static Comparison ──")
    x = np.linspace(0, PITCH_W_T, NX_T)
    y = np.linspace(0, PITCH_H_T, NY_T)
    XX, YY = np.meshgrid(x, y, indexing="ij")

    formations = [
        (F4231_HIGH, "4-2-3-1 — Dispersed Block", "Darcy Regime: High permeability, wide passing channels open between players"),
        (F532, "5-3-2 — Compact Block", "Choked Flow: Low permeability, passing lanes constricted by narrow defensive spacing"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#111111")
    fig.suptitle("11v11 Defensive Influence Fields — Heatmap & Passing Lane Analysis",
                 color="white", fontsize=14, y=0.98)

    vmin, vmax = 0.0, 4.0
    levels = np.linspace(vmin, vmax, 50)

    for ax, (pos, label, caption) in zip(axes, formations):
        field = compute_influence_field(XX, YY, pos)
        # Permeability within defensive half only (x > 5.25)
        half_mask = XX > PITCH_W_T / 2
        perm = np.sum(field[half_mask] < INFLUENCE_THRESHOLD) / np.sum(half_mask) * 100

        ax.set_facecolor("#111111")
        draw_pitch(ax)

        cf = ax.contourf(x, y, field.T, levels=levels, cmap="plasma", vmin=vmin, vmax=vmax, extend="max")

        cs = ax.contour(x, y, field.T, levels=[INFLUENCE_THRESHOLD],
                        colors="cyan", linewidths=1.5, alpha=0.8,
                        linestyles="dashed")
        ax.clabel(cs, fmt="%.2f", colors="cyan", fontsize=7, inline=True)

        ax.scatter(pos[1:, 0], pos[1:, 1], c="white", s=35,
                   edgecolors="black", linewidths=0.5, zorder=5)
        ax.scatter([pos[0, 0]], [pos[0, 1]], c="gold", s=30,
                   marker="*", edgecolors="black", linewidths=0.5, zorder=5)

        ax.set_xlim(0, PITCH_W_T)
        ax.set_ylim(0, PITCH_H_T)
        ax.set_aspect("equal")
        ax.set_title(label, color="white", fontsize=11)

        ax.text(0.02, 0.98, caption, color="cyan", fontsize=8, alpha=0.85,
                transform=ax.transAxes, va="top")
        ax.text(0.02, 0.92, f"Defensive half: {perm:.0f}% open space",
                color="white", fontsize=9, transform=ax.transAxes, va="top")

    fig.tight_layout(rect=[0, 0, 0.92, 0.93])
    cbar_ax = fig.add_axes([0.93, 0.12, 0.012, 0.76])
    cbar = fig.colorbar(cf, cax=cbar_ax)
    cbar.set_label("Defensive Influence", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    path = ASSETS / "team_heatmaps.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="#111111")
    print(f"Saved {path}")
    plt.close(fig)


# ── Buckets D & E: Collapse Animation + Metrics ──

def animate_team_collapse():
    print("── Team Heatmaps — Defensive Collapse Animation ──")
    x = np.linspace(0, PITCH_W_T, NX_T)
    y = np.linspace(0, PITCH_H_T, NY_T)
    XX, YY = np.meshgrid(x, y, indexing="ij")

    t_frac = np.linspace(0, 1, N_COLLAPSE_FRAMES)
    positions_seq = F4231_HIGH[np.newaxis] + t_frac[:, np.newaxis, np.newaxis] * (F4231_LOW - F4231_HIGH)[np.newaxis]

    print("Computing collapse fields...")
    half_mask = XX > PITCH_W_T / 2
    fields = []
    perm_series = []
    for frame in trange(N_COLLAPSE_FRAMES):
        field = compute_influence_field(XX, YY, positions_seq[frame])
        fields.append(field)
        perm_series.append(np.sum(field[half_mask] < INFLUENCE_THRESHOLD) / np.sum(half_mask) * 100)

    vmin, vmax = 0.0, 4.0
    levels = np.linspace(vmin, vmax, 50)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#111111",
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle("Defensive Collapse — 4-2-3-1 Low Block Transition",
                 color="white", fontsize=14, y=0.98)

    def update(idx):
        for ax in axes:
            ax.clear()
            ax.set_facecolor("#111111")

        ax0 = axes[0]
        half_mask = XX > PITCH_W_T / 2

        draw_pitch(ax0)
        cf = ax0.contourf(x, y, fields[idx].T, levels=levels,
                          cmap="plasma", vmin=vmin, vmax=vmax, extend="max")
        cs = ax0.contour(x, y, fields[idx].T, levels=[INFLUENCE_THRESHOLD],
                         colors="cyan", linewidths=1.5, alpha=0.8,
                         linestyles="dashed")

        pos = positions_seq[idx]
        ax0.scatter(pos[1:, 0], pos[1:, 1], c="white", s=35,
                    edgecolors="black", linewidths=0.5, zorder=5)
        ax0.scatter([pos[0, 0]], [pos[0, 1]], c="gold", s=30,
                    marker="*", edgecolors="black", linewidths=0.5, zorder=5)

        ax0.set_xlim(0, PITCH_W_T)
        ax0.set_ylim(0, PITCH_H_T)
        ax0.set_aspect("equal")
        pct = idx / (N_COLLAPSE_FRAMES - 1) * 100
        ax0.set_title(f"Collapse: {pct:.0f}% — 4-2-3-1", color="white", fontsize=11)

        ax1 = axes[1]
        ax1.plot(t_frac[:idx + 1], perm_series[:idx + 1], color="cyan", lw=2)
        ax1.set_xlim(0, 1)
        y_max = max(perm_series) * 1.15
        ax1.set_ylim(0, max(y_max, 5))
        ax1.set_xlabel("Collapse Progress", color="white")
        ax1.set_ylabel("Open Space (%) in Defensive Half", color="white")
        ax1.set_title("Permeability (Darcy Flux Analog)", color="white", fontsize=10)
        ax1.grid(alpha=0.2)
        ax1.scatter([t_frac[idx]], [perm_series[idx]], color="cyan", s=40, zorder=5)
        ax1.text(0.5, 0.9, f"{perm_series[idx]:.0f}% open", color="cyan", fontsize=11,
                 ha="center", transform=ax1.transAxes)

        return axes

    anim = FuncAnimation(fig, update, frames=N_COLLAPSE_FRAMES, interval=80, blit=False)
    path = ASSETS / "defensive_collapse.mp4"
    anim.save(str(path), writer="ffmpeg", fps=12, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


# ── Orchestrator ──

def run_team_heatmaps():
    plot_team_heatmaps()
    animate_team_collapse()


if __name__ == "__main__":
    run_tactical()
    run_macro_stress()
    run_team_heatmaps()
