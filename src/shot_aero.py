"""Shot Aerodynamics — Magnus vs Knuckleball cylinder simulation."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phi.jax.flow import Obstacle, StaggeredGrid, CenteredGrid, fluid, advect, vec, field, ZERO_GRADIENT

from src.utils import setup_theme, ASSETS
from src.domain import BOX, BOUNDARY, CENTER, RADIUS, CYLINDER, CENTERED, DT, N_STEPS, SAVE_EVERY

setup_theme()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from tqdm import trange


def run_cylinder(obstacle, label):
    v0 = StaggeredGrid((1.0, 0.0), BOUNDARY, x=256, y=128, bounds=BOX)
    v, _ = fluid.make_incompressible(v0, ())
    frames = []
    v = v0
    for i in trange(N_STEPS, desc=label):
        v = advect.semi_lagrangian(v, v, dt=DT)
        v, p = fluid.make_incompressible(v, obstacle)
        if i % SAVE_EVERY == 0:
            vc = v @ CENTERED
            u_c = np.array(vc.vector[0].values.native("x", "y"))
            v_c = np.array(vc.vector[1].values.native("x", "y"))
            vel_mag = field.vec_length(v)
            p_np = np.array(p.values.native("x", "y"))
            frames.append({
                "pressure": p_np,
                "u": u_c,
                "v": v_c,
                "velocity": np.array(vel_mag.values.native("x", "y")),
                "step": i,
            })
    return frames


def animate_velocity_comparison(frames_magnus, frames_knuckle):
    n_frames = min(len(frames_magnus), len(frames_knuckle))
    fm = frames_magnus[:n_frames]
    fk = frames_knuckle[:n_frames]

    x_s = np.linspace(0, 8.0, 256)
    y_s = np.linspace(0, 4.0, 128)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#111111")
    fig.suptitle("Knuckleball vs Regular Shot — Wake Comparison", color="white", fontsize=14, y=0.98)

    titles = [
        "Magnus Effect (\u03c9 = 10 rad/s)",
        "Knuckleball (\u03c9 = 0 rad/s)",
    ]
    frames_list = [fm, fk]
    imgs = []

    for ax, title, frames in zip([ax1, ax2], titles, frames_list):
        ax.set_facecolor("#111111")
        ax.set_title(title, color="white", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0.5, 7.5)
        ax.set_ylim(0.5, 3.5)

        im = ax.imshow(
            frames[0]["velocity"].T,
            origin="lower", cmap="inferno", aspect="auto",
            extent=[0, 8, 0, 4], vmin=0, vmax=1.5,
        )
        ax.add_patch(Circle(CENTER, RADIUS, color="white", fill=False, lw=2.0))
        ax.streamplot(
            x_s, y_s, frames[0]["u"].T, frames[0]["v"].T,
            color=frames[0]["velocity"].T, cmap="inferno",
            linewidth=0.8, density=1.5, arrowstyle="->", arrowsize=0.6,
        )
        imgs.append(im)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    cbar = fig.colorbar(imgs[0], ax=[ax1, ax2], fraction=0.04, pad=0.02)
    cbar.set_label("Velocity magnitude", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    def update(idx):
        for ax, frames in zip([ax1, ax2], frames_list):
            ax.clear()
            ax.set_facecolor("#111111")
            ax.set_xlim(0.5, 7.5)
            ax.set_ylim(0.5, 3.5)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.imshow(
                frames[idx]["velocity"].T,
                origin="lower", cmap="inferno", aspect="auto",
                extent=[0, 8, 0, 4], vmin=0, vmax=1.5,
            )
            ax.add_patch(Circle(CENTER, RADIUS, color="white", fill=False, lw=2.0))
            ax.streamplot(
                x_s, y_s, frames[idx]["u"].T, frames[idx]["v"].T,
                color=frames[idx]["velocity"].T, cmap="inferno",
                linewidth=0.8, density=1.5, arrowstyle="->", arrowsize=0.6,
            )
        return (ax1, ax2)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=80, blit=False)
    path = ASSETS / "knuckleball_comparison.mp4"
    anim.save(str(path), writer="ffmpeg", fps=10, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


def animate_pressure_comparison(frames_magnus, frames_knuckle):
    n_frames = min(len(frames_magnus), len(frames_knuckle))
    fm = frames_magnus[:n_frames]
    fk = frames_knuckle[:n_frames]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#111111")
    fig.suptitle("Knuckleball vs Magnus — Pressure Field", color="white", fontsize=14, y=0.98)

    titles = [
        "Magnus Effect (\u03c9 = 10 rad/s)",
        "Knuckleball (\u03c9 = 0 rad/s)",
    ]
    frames_list = [fm, fk]
    imgs = []

    for ax, title, frames in zip([ax1, ax2], titles, frames_list):
        ax.set_facecolor("#111111")
        ax.set_title(title, color="white", fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0.5, 7.5)
        ax.set_ylim(0.5, 3.5)

        p0 = frames[0]["pressure"]
        im = ax.imshow(p0.T, origin="lower", cmap="magma", aspect="auto", extent=[0, 8, 0, 4])
        ax.add_patch(Circle(CENTER, RADIUS, color="white", fill=False, lw=2.0))
        imgs.append(im)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    cbar = fig.colorbar(imgs[0], ax=[ax1, ax2], fraction=0.04, pad=0.02)
    cbar.set_label("Pressure", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    def update(idx):
        for ax, frames, im in zip([ax1, ax2], frames_list, imgs):
            data = frames[idx]["pressure"]
            im.set_data(data.T)
            im.set_clim(data.min(), data.max())
        fig.suptitle(
            f"Knuckleball vs Magnus — Step {frames_list[0][idx]['step']}"
            f"  (t = {frames_list[0][idx]['step'] * 0.3:.1f}s)",
            color="white", fontsize=14, y=0.98,
        )
        return imgs

    anim = FuncAnimation(fig, update, frames=n_frames, interval=80, blit=True)
    path = ASSETS / "knuckleball_pressure.mp4"
    anim.save(str(path), writer="ffmpeg", fps=10, dpi=150)
    print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    obstacle_spin = Obstacle(CYLINDER, velocity=vec(x=0.0, y=0.0), angular_velocity=10.0)
    obstacle_nospin = Obstacle(CYLINDER, velocity=vec(x=0.0, y=0.0), angular_velocity=0.0)

    print("Magnus (ω = 10 rad/s)")
    frames_magnus = run_cylinder(obstacle_spin, "Magnus")

    print("\nKnuckleball (ω = 0 rad/s)")
    frames_knuckle = run_cylinder(obstacle_nospin, "Knuckleball")

    print(f"\nCaptured {len(frames_magnus)} frames each")
    print(f"  u shape: {frames_magnus[0]['u'].shape}")

    animate_velocity_comparison(frames_magnus, frames_knuckle)
    animate_pressure_comparison(frames_magnus, frames_knuckle)
