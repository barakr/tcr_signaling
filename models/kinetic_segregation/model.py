"""Kinetic segregation model — 2D Monte Carlo membrane simulation.

Simulates TCR-CD45 spatial segregation on a 2um x 2um membrane patch using
Metropolis Monte Carlo. TCR molecules are attracted to tight-contact regions
(low membrane height) while CD45 molecules are repelled due to their large
ectodomain. The primary output is the depletion zone width — the spatial gap
between TCR-enriched and CD45-enriched domains.

Reference: Neve-Oz, Sherman & Raveh, Frontiers in Immunology, 2024.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .potentials import bending_energy_delta, cd45_repulsion, tcr_pmhc_potential

# Default physical parameters from Supplementary Table S1
PATCH_SIZE_NM = 2000.0  # 2 um patch
N_TCR_DEFAULT = 50
N_CD45_DEFAULT = 100
CD45_HEIGHT_NM = 35.0  # ectodomain height
SIGMA_BIND_NM = 3.0  # TCR-pMHC binding well width
TIME_REF_SEC = 20.0  # reference time for n_steps scaling


def _initial_positions(
    n: int, patch_size: float, rng: np.random.Generator, center_bias: bool = False
) -> NDArray[np.float64]:
    """Generate initial (x, y) positions for molecules on the patch.

    If center_bias is True, molecules are placed near the patch center
    (mimicking the initial TCR cluster).
    """
    if center_bias:
        center = patch_size / 2.0
        spread = patch_size / 6.0
        pos = rng.normal(loc=center, scale=spread, size=(n, 2))
        return np.clip(pos, 0.0, patch_size)
    return rng.uniform(0.0, patch_size, size=(n, 2))


def _height_at(pos: NDArray[np.float64], h: NDArray[np.float64], dx: float) -> NDArray[np.float64]:
    """Interpolate membrane height at molecule positions using nearest grid point."""
    nx, ny = h.shape
    ix = np.clip((pos[:, 0] / dx).astype(int), 0, nx - 1)
    iy = np.clip((pos[:, 1] / dx).astype(int), 0, ny - 1)
    return h[ix, iy]


def _compute_depletion_width(
    tcr_pos: NDArray[np.float64],
    cd45_pos: NDArray[np.float64],
    patch_size: float,
) -> float:
    """Compute depletion zone width from radial distributions.

    Uses the median radial distance from the patch center for each species.
    The depletion width is the gap between the CD45 median radius and the
    TCR median radius. The median is more robust than percentile-based
    metrics when molecule counts are small (10-50).
    """
    center = patch_size / 2.0
    tcr_r = np.sqrt(np.sum((tcr_pos - center) ** 2, axis=1))
    cd45_r = np.sqrt(np.sum((cd45_pos - center) ** 2, axis=1))
    return max(0.0, float(np.median(cd45_r) - np.median(tcr_r)))


def simulate_ks(
    time_sec: float,
    rigidity_kT_nm2: float,
    u_assoc: float = 20.0,
    n_tcr: int = N_TCR_DEFAULT,
    n_cd45: int = N_CD45_DEFAULT,
    grid_size: int = 64,
    n_steps: int | None = None,
    seed: int = 42,
    snapshot_interval: int = 0,
) -> dict:
    """Run kinetic segregation Monte Carlo simulation.

    Parameters
    ----------
    time_sec : Simulation time (sec). Mapped to MC steps.
    rigidity_kT_nm2 : Membrane bending rigidity (kT).
    u_assoc : TCR-pMHC binding potential depth (kT).
    n_tcr : Number of TCR molecules.
    n_cd45 : Number of CD45 molecules.
    grid_size : Number of grid points per dimension for membrane height field.
    n_steps : Number of full MC sweeps. Each sweep updates every molecule and
        every grid cell once. If None, derived from time_sec.
    seed : Random seed.
    snapshot_interval : If > 0, record (tcr_pos, cd45_pos, h) every N steps.

    Returns
    -------
    dict with keys: depletion_width_nm, final_tcr_mean_r, final_cd45_mean_r,
                    accept_rate, n_steps_actual, and optionally snapshots
    """
    rng = np.random.default_rng(seed)
    patch = PATCH_SIZE_NM
    dx = patch / grid_size

    # Initialize membrane height field — flat at CD45 height (equilibrium gap)
    h = np.full((grid_size, grid_size), CD45_HEIGHT_NM, dtype=np.float64)
    # Depress center to create initial tight-contact seed
    center_idx = grid_size // 2
    radius_idx = max(1, grid_size // 8)
    y_idx, x_idx = np.ogrid[:grid_size, :grid_size]
    dist_sq = (x_idx - center_idx) ** 2 + (y_idx - center_idx) ** 2
    h[dist_sq <= radius_idx**2] = 5.0  # tight contact ~5 nm

    # Initialize molecule positions
    tcr_pos = _initial_positions(n_tcr, patch, rng, center_bias=True)
    cd45_pos = _initial_positions(n_cd45, patch, rng, center_bias=False)

    # MC sweeps: scale with time (more time = more equilibration).
    # When n_steps is given, treat it as the base count at TIME_REF_SEC
    # and scale linearly so longer simulations always get more sweeps.
    if n_steps is None:
        n_steps = max(50, round(time_sec * 5))
    else:
        n_steps = max(n_steps, round(n_steps * time_sec / TIME_REF_SEC))

    # Molecular step size: must resolve the TCR binding well (sigma_bind)
    # while still allowing reasonable exploration of the patch.
    step_size_mol = max(SIGMA_BIND_NM * 2, dx * 0.15)
    # Height step size: adapt to bending rigidity so acceptance stays reasonable.
    # Stiffer membranes need smaller perturbations.
    kappa = rigidity_kT_nm2
    step_size_h = min(5.0, dx / (4.0 * max(1.0, np.sqrt(kappa))))

    accepted = 0
    total_proposals = 0
    snapshots: list[dict] = []

    # Record initial snapshot
    if snapshot_interval > 0:
        snapshots.append(
            {"step": 0, "tcr_pos": tcr_pos.copy(), "cd45_pos": cd45_pos.copy(), "h": h.copy()}
        )

    for step_i in range(n_steps):
        # --- Phase 1: Sweep ALL molecules ---
        for mol_set, is_tcr in [(tcr_pos, True), (cd45_pos, False)]:
            for idx in range(len(mol_set)):
                old_pos = mol_set[idx].copy()

                # Compute old energy for this molecule
                old_h = _height_at(mol_set[idx : idx + 1], h, dx)
                if is_tcr:
                    old_e = tcr_pmhc_potential(float(old_h[0]), u_assoc, SIGMA_BIND_NM)
                else:
                    old_e = cd45_repulsion(float(old_h[0]), CD45_HEIGHT_NM)

                # Propose displacement
                mol_set[idx] += rng.normal(0, step_size_mol, size=2)
                mol_set[idx] = np.clip(mol_set[idx], 0.0, patch)

                # Compute new energy
                new_h = _height_at(mol_set[idx : idx + 1], h, dx)
                if is_tcr:
                    new_e = tcr_pmhc_potential(float(new_h[0]), u_assoc, SIGMA_BIND_NM)
                else:
                    new_e = cd45_repulsion(float(new_h[0]), CD45_HEIGHT_NM)

                dE = new_e - old_e
                total_proposals += 1
                if dE < 0 or rng.random() < np.exp(-dE):
                    accepted += 1
                else:
                    mol_set[idx] = old_pos

        # --- Phase 2: Sweep ALL grid cells ---
        for gi in range(grid_size):
            for gj in range(grid_size):
                old_h_val = h[gi, gj]

                # Energy from molecules at this grid cell
                old_mol_e = 0.0
                cell_tcr = np.where((tcr_pos[:, 0] // dx).astype(int).clip(0, grid_size - 1) == gi)[
                    0
                ]
                cell_tcr = cell_tcr[
                    (tcr_pos[cell_tcr, 1] // dx).astype(int).clip(0, grid_size - 1) == gj
                ]
                old_mol_e += len(cell_tcr) * tcr_pmhc_potential(old_h_val, u_assoc, SIGMA_BIND_NM)

                cell_cd45 = np.where(
                    (cd45_pos[:, 0] // dx).astype(int).clip(0, grid_size - 1) == gi
                )[0]
                cell_cd45 = cell_cd45[
                    (cd45_pos[cell_cd45, 1] // dx).astype(int).clip(0, grid_size - 1) == gj
                ]
                old_mol_e += len(cell_cd45) * cd45_repulsion(old_h_val, CD45_HEIGHT_NM)

                # Propose new height
                h[gi, gj] = old_h_val + rng.normal(0, step_size_h)
                h[gi, gj] = max(0.0, h[gi, gj])  # membrane can't go below 0

                dE_bend = bending_energy_delta(h, kappa, dx, gi, gj, old_h_val, h[gi, gj])
                new_mol_e = len(cell_tcr) * tcr_pmhc_potential(
                    h[gi, gj], u_assoc, SIGMA_BIND_NM
                ) + len(cell_cd45) * cd45_repulsion(h[gi, gj], CD45_HEIGHT_NM)

                dE = dE_bend + (new_mol_e - old_mol_e)
                total_proposals += 1
                if dE < 0 or rng.random() < np.exp(-min(dE, 500)):
                    accepted += 1
                else:
                    h[gi, gj] = old_h_val

        # Record snapshot
        if snapshot_interval > 0 and (step_i + 1) % snapshot_interval == 0:
            snapshots.append(
                {
                    "step": step_i + 1,
                    "tcr_pos": tcr_pos.copy(),
                    "cd45_pos": cd45_pos.copy(),
                    "h": h.copy(),
                }
            )

    # Measure depletion from final configuration (not averaged — this is dynamics)
    depletion_width = _compute_depletion_width(tcr_pos, cd45_pos, patch)
    center = patch / 2.0
    tcr_r = np.sqrt(np.sum((tcr_pos - center) ** 2, axis=1))
    cd45_r = np.sqrt(np.sum((cd45_pos - center) ** 2, axis=1))

    accept_rate = accepted / total_proposals if total_proposals > 0 else 0.0

    result = {
        "depletion_width_nm": depletion_width,
        "final_tcr_mean_r_nm": float(np.mean(tcr_r)),
        "final_cd45_mean_r_nm": float(np.mean(cd45_r)),
        "accept_rate": accept_rate,
        "n_steps_actual": n_steps,
    }
    if snapshot_interval > 0:
        result["snapshots"] = snapshots
    return result
