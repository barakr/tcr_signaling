"""Generate an animation of the kinetic segregation Monte Carlo simulation.

Produces an MP4 movie showing TCR (red) and CD45 (blue) molecules on the
membrane patch as the simulation evolves. The membrane height field is shown
as a background colormap. A depletion zone ring is overlaid.

Usage::

    python -m projects.tcr_signaling.models.kinetic_segregation.animate \\
        --output ks_segregation.mp4 --n_steps 5000 --fps 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .model import PATCH_SIZE_NM, _compute_depletion_width, simulate_ks


def render_animation(
    snapshots: list[dict],
    patch_size: float = PATCH_SIZE_NM,
    output: str = "ks_segregation.mp4",
    fps: int = 15,
    dpi: int = 120,
) -> Path:
    """Render simulation snapshots to an MP4 animation.

    Parameters
    ----------
    snapshots : List of dicts with keys tcr_pos, cd45_pos, h, step.
    patch_size : Patch size in nm.
    output : Output file path.
    fps : Frames per second.
    dpi : Resolution.

    Returns
    -------
    Path to the saved animation file.
    """
    fig, (ax_main, ax_h) = plt.subplots(1, 2, figsize=(14, 6))
    fig.set_facecolor("white")

    center = patch_size / 2.0
    extent_um = [0, patch_size / 1000, 0, patch_size / 1000]  # convert nm to um

    # --- Left panel: molecule positions ---
    ax_main.set_xlim(0, patch_size / 1000)
    ax_main.set_ylim(0, patch_size / 1000)
    ax_main.set_aspect("equal")
    ax_main.set_xlabel("x (um)")
    ax_main.set_ylabel("y (um)")

    tcr_scatter = ax_main.scatter([], [], c="red", s=20, alpha=0.7, label="TCR", zorder=3)
    cd45_scatter = ax_main.scatter([], [], c="royalblue", s=12, alpha=0.5, label="CD45", zorder=2)
    depletion_circle_inner = plt.Circle(
        (center / 1000, center / 1000), 0, fill=False, color="gold", lw=2, ls="--", zorder=4
    )
    depletion_circle_outer = plt.Circle(
        (center / 1000, center / 1000), 0, fill=False, color="gold", lw=2, ls="--", zorder=4
    )
    ax_main.add_patch(depletion_circle_inner)
    ax_main.add_patch(depletion_circle_outer)
    ax_main.legend(loc="upper right", fontsize=9)
    title_text = ax_main.set_title("Step 0")

    # --- Right panel: membrane height field ---
    h0 = snapshots[0]["h"]
    im = ax_h.imshow(
        h0.T if h0.shape[0] == h0.shape[1] else h0,
        cmap="viridis",
        origin="lower",
        extent=extent_um,
        vmin=0,
        vmax=50,
        aspect="equal",
    )
    ax_h.set_xlabel("x (um)")
    ax_h.set_ylabel("y (um)")
    ax_h.set_title("Membrane height (nm)")
    fig.colorbar(im, ax=ax_h, label="h (nm)", shrink=0.8)

    fig.tight_layout()

    def update(frame_idx):
        snap = snapshots[frame_idx]
        tcr = snap["tcr_pos"] / 1000.0  # nm -> um
        cd45 = snap["cd45_pos"] / 1000.0

        tcr_scatter.set_offsets(tcr)
        cd45_scatter.set_offsets(cd45)

        # Compute radial statistics for depletion zone visualization
        c_um = center / 1000.0
        tcr_r = np.sqrt(np.sum((tcr - c_um) ** 2, axis=1))
        cd45_r = np.sqrt(np.sum((cd45 - c_um) ** 2, axis=1))
        tcr_75 = np.percentile(tcr_r, 75)
        cd45_25 = np.percentile(cd45_r, 25)

        depletion_circle_inner.set_radius(tcr_75)
        depletion_circle_outer.set_radius(cd45_25)

        # Update membrane height
        im.set_data(snap["h"])

        depletion_w = _compute_depletion_width(snap["tcr_pos"], snap["cd45_pos"], patch_size)
        title_text.set_text(f"Step {snap['step']}  |  depletion width = {depletion_w:.0f} nm")

        return tcr_scatter, cd45_scatter, im, title_text

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=1000 // fps, blit=False)

    out_path = Path(output)
    suffix = out_path.suffix.lower()
    if suffix == ".gif":
        anim.save(str(out_path), writer="pillow", fps=fps, dpi=dpi)
    else:
        anim.save(str(out_path), writer="ffmpeg", fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"Animation saved to {out_path} ({len(snapshots)} frames, {fps} fps)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Animate kinetic segregation simulation")
    parser.add_argument("--output", type=str, default="ks_segregation.gif", help="Output file")
    parser.add_argument("--time_sec", type=float, default=50.0)
    parser.add_argument("--rigidity_kT_nm2", type=float, default=20.0)
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--n_tcr", type=int, default=50)
    parser.add_argument("--n_cd45", type=int, default=100)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--snapshot_every", type=int, default=50, help="Record every N steps")
    args = parser.parse_args()

    print(f"Running KS simulation: {args.n_steps} steps, snapshots every {args.snapshot_every}...")
    result = simulate_ks(
        time_sec=args.time_sec,
        rigidity_kT_nm2=args.rigidity_kT_nm2,
        n_tcr=args.n_tcr,
        n_cd45=args.n_cd45,
        grid_size=args.grid_size,
        n_steps=args.n_steps,
        seed=args.seed,
        snapshot_interval=args.snapshot_every,
    )
    print(f"Simulation done. Depletion width = {result['depletion_width_nm']:.1f} nm")
    print(f"Rendering {len(result['snapshots'])} frames...")

    render_animation(result["snapshots"], output=args.output, fps=args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
