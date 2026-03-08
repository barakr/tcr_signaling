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

from .potentials import bending_energy_delta, cd45_repulsion, mol_repulsion, tcr_pmhc_potential

# Default physical parameters from Supplementary Table S1
PATCH_SIZE_NM = 2000.0  # 2 um patch
N_TCR_DEFAULT = 50
N_CD45_DEFAULT = 100
CD45_HEIGHT_NM = 35.0  # ectodomain height
SIGMA_BIND_NM = 3.0  # TCR-pMHC binding well width

# Brownian dynamics diffusion coefficients (nm²/s)
D_MOL_DEFAULT = 1e5  # membrane protein diffusion
D_H_DEFAULT = 5e4    # membrane height relaxation
DT_SAFETY = 0.5      # stability safety factor


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
        return pos % patch_size
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
    D_mol: float = D_MOL_DEFAULT,
    D_h: float = D_H_DEFAULT,
    dt_override: float | None = None,
    cd45_height: float = CD45_HEIGHT_NM,
    cd45_k_rep: float = 1.0,
    mol_repulsion_eps: float = 0.0,
    mol_repulsion_rcut: float = 10.0,
    n_pmhc: int = 0,
    pmhc_pos: NDArray[np.float64] | None = None,
    pmhc_seed: int | None = None,
    pmhc_mode: str = "inner_circle",
    pmhc_radius: float | None = None,
) -> dict:
    """Run kinetic segregation Monte Carlo simulation.

    Parameters
    ----------
    time_sec : Simulation time (sec). Mapped to MC steps via Brownian dynamics dt.
    rigidity_kT_nm2 : Membrane bending rigidity (kT).
    u_assoc : TCR-pMHC binding potential depth (kT).
    n_tcr : Number of TCR molecules.
    n_cd45 : Number of CD45 molecules.
    grid_size : Number of grid points per dimension for membrane height field.
    n_steps : Number of full MC sweeps. If given, overrides auto-computation
        from time_sec / dt. Each sweep updates every molecule and every grid
        cell once.
    seed : Random seed.
    snapshot_interval : If > 0, record (tcr_pos, cd45_pos, h) every N steps.
    D_mol : Molecular diffusion coefficient (nm²/s).
    D_h : Membrane height diffusion coefficient (nm²/s).
    dt_override : If given, use this time step instead of auto-computing from
        stability constraint. Step sizes are still derived from D and dt.

    Returns
    -------
    dict with keys: depletion_width_nm, final_tcr_mean_r, final_cd45_mean_r,
                    accept_rate, n_steps_actual, dt_seconds, step_size_h_nm,
                    step_size_mol_nm, and optionally snapshots
    """
    if pmhc_mode not in ("uniform", "inner_circle"):
        raise ValueError(f"Unknown pmhc_mode: {pmhc_mode!r} (expected 'uniform' or 'inner_circle')")

    rng = np.random.default_rng(seed)
    patch = PATCH_SIZE_NM
    dx = patch / grid_size
    kappa = rigidity_kT_nm2

    # Initialize membrane height field — flat at CD45 height (equilibrium gap)
    h = np.full((grid_size, grid_size), cd45_height, dtype=np.float64)
    # Depress center to create initial tight-contact seed
    center_idx = grid_size // 2
    radius_idx = max(1, grid_size // 8)
    y_idx, x_idx = np.ogrid[:grid_size, :grid_size]
    dist_sq = (x_idx - center_idx) ** 2 + (y_idx - center_idx) ** 2
    h[dist_sq <= radius_idx**2] = 5.0  # tight contact ~5 nm

    # --- pMHC initialization (before TCR so TCR can co-locate) ---
    # When n_pmhc=0 and pmhc_pos=None, all cells have pMHC (backward compat)
    pmhc_grid: NDArray[np.int32] | None = None
    if n_pmhc > 0 or pmhc_pos is not None:
        if pmhc_pos is None:
            pmhc_rng = np.random.default_rng(pmhc_seed if pmhc_seed is not None else seed + 1)
            eff_radius = pmhc_radius if pmhc_radius is not None else patch / 3.0
            center_xy = patch / 2.0
            if pmhc_mode == "inner_circle":
                # Rejection sampling within centered disc
                pmhc_pos = np.empty((n_pmhc, 2), dtype=np.float64)
                placed = 0
                while placed < n_pmhc:
                    cand = pmhc_rng.uniform(0.0, patch, size=(n_pmhc * 4, 2))
                    dist = np.sqrt(np.sum((cand - center_xy) ** 2, axis=1))
                    valid = cand[dist <= eff_radius]
                    take = min(len(valid), n_pmhc - placed)
                    pmhc_pos[placed : placed + take] = valid[:take]
                    placed += take
            else:
                # uniform mode: full patch
                pmhc_pos = pmhc_rng.uniform(0.0, patch, size=(n_pmhc, 2))
        pmhc_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        pi = (pmhc_pos[:, 0] // dx).astype(int).clip(0, grid_size - 1)
        pj = (pmhc_pos[:, 1] // dx).astype(int).clip(0, grid_size - 1)
        for a, b in zip(pi, pj):
            pmhc_grid[a, b] += 1

    # --- TCR initialization ---
    if n_pmhc > 0 and pmhc_pos is not None:
        # Co-locate TCR on top of pMHC positions with small jitter
        indices = rng.integers(0, n_pmhc, size=n_tcr)
        tcr_pos = pmhc_pos[indices].copy() + rng.normal(0, SIGMA_BIND_NM, size=(n_tcr, 2))
        tcr_pos = tcr_pos % patch
    else:
        # Backward compat: center-biased Gaussian
        tcr_pos = _initial_positions(n_tcr, patch, rng, center_bias=True)

    # CD45: always uniform
    cd45_pos = _initial_positions(n_cd45, patch, rng, center_bias=False)

    # Brownian dynamics time step: dt = σ² / (2D), with stability constraint
    # dt_stable = dx² / (2 * D_h * κ)
    if dt_override is not None:
        dt = dt_override
    else:
        dt_stable = dx**2 / (2.0 * D_h * kappa)
        dt = dt_stable * DT_SAFETY

    # Derive step sizes from physics: σ = sqrt(2 * D * dt)
    step_size_mol = np.sqrt(2.0 * D_mol * dt)
    step_size_h = np.sqrt(2.0 * D_h * dt)

    # Auto-compute n_steps from physical time if not explicitly given
    if n_steps is None:
        n_steps = max(50, round(time_sec / dt))

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
                    old_gi = min(int(mol_set[idx, 0] // dx), grid_size - 1)
                    old_gj = min(int(mol_set[idx, 1] // dx), grid_size - 1)
                    has_pmhc_old = pmhc_grid is None or pmhc_grid[old_gi, old_gj] > 0
                    old_e = (
                        tcr_pmhc_potential(float(old_h[0]), u_assoc, SIGMA_BIND_NM)
                        if has_pmhc_old
                        else 0.0
                    )
                else:
                    old_e = cd45_repulsion(float(old_h[0]), cd45_height, cd45_k_rep)
                if mol_repulsion_eps > 0.0:
                    old_e += mol_repulsion(
                        mol_set[idx], idx, mol_set, mol_repulsion_eps,
                        mol_repulsion_rcut, patch,
                    )

                # Propose displacement (periodic wrap)
                mol_set[idx] += rng.normal(0, step_size_mol, size=2)
                mol_set[idx] = mol_set[idx] % patch

                # Compute new energy
                new_h = _height_at(mol_set[idx : idx + 1], h, dx)
                if is_tcr:
                    new_gi = min(int(mol_set[idx, 0] // dx), grid_size - 1)
                    new_gj = min(int(mol_set[idx, 1] // dx), grid_size - 1)
                    has_pmhc_new = pmhc_grid is None or pmhc_grid[new_gi, new_gj] > 0
                    new_e = (
                        tcr_pmhc_potential(float(new_h[0]), u_assoc, SIGMA_BIND_NM)
                        if has_pmhc_new
                        else 0.0
                    )
                else:
                    new_e = cd45_repulsion(float(new_h[0]), cd45_height, cd45_k_rep)
                if mol_repulsion_eps > 0.0:
                    new_e += mol_repulsion(
                        mol_set[idx], idx, mol_set, mol_repulsion_eps,
                        mol_repulsion_rcut, patch,
                    )

                dE = new_e - old_e
                total_proposals += 1
                u = rng.random()
                if dE <= 0 or (u > 0.0 and np.log(u) < -dE):
                    accepted += 1
                else:
                    mol_set[idx] = old_pos

        # --- Phase 2: Checkerboard + snapshot grid update (matching C/GPU) ---
        # Pre-bin molecules to grid counts (O(N) instead of O(N) per cell)
        tcr_gi = (tcr_pos[:, 0] // dx).astype(int).clip(0, grid_size - 1)
        tcr_gj = (tcr_pos[:, 1] // dx).astype(int).clip(0, grid_size - 1)
        cd45_gi = (cd45_pos[:, 0] // dx).astype(int).clip(0, grid_size - 1)
        cd45_gj = (cd45_pos[:, 1] // dx).astype(int).clip(0, grid_size - 1)
        tcr_count = np.zeros((grid_size, grid_size), dtype=int)
        cd45_count = np.zeros((grid_size, grid_size), dtype=int)
        for ti, tj in zip(tcr_gi, tcr_gj):
            tcr_count[ti, tj] += 1
        for ci, cj in zip(cd45_gi, cd45_gj):
            cd45_count[ci, cj] += 1

        for color in range(2):
            # Collect cells of this color
            cells = [
                (gi, gj)
                for gi in range(grid_size)
                for gj in range(grid_size)
                if (gi + gj) % 2 == color
            ]

            # Pass 1 (propose): generate proposals, write ALL to h[]
            old_vals = np.empty(len(cells))
            u_accepts = np.empty(len(cells))
            for k, (gi, gj) in enumerate(cells):
                old_vals[k] = h[gi, gj]
                new_h_val = h[gi, gj] + rng.normal(0, step_size_h)
                new_h_val = abs(new_h_val)  # reflecting boundary
                h[gi, gj] = new_h_val
                u_accepts[k] = rng.random()

            # Snapshot: freeze h[] for consistent reads
            h_snap = h.copy()

            # Pass 2 (evaluate): bending deltas from frozen snapshot
            accepted_flags = np.zeros(len(cells), dtype=bool)
            for k, (gi, gj) in enumerate(cells):
                old_h_val = old_vals[k]
                new_h_val = h_snap[gi, gj]

                n_tcr_cell = tcr_count[gi, gj]
                n_cd45_cell = cd45_count[gi, gj]

                # TCR contribution only where pMHC is present
                has_pmhc = pmhc_grid is None or pmhc_grid[gi, gj] > 0
                tcr_old = (
                    n_tcr_cell * tcr_pmhc_potential(old_h_val, u_assoc, SIGMA_BIND_NM)
                    if has_pmhc
                    else 0.0
                )
                old_mol_e = tcr_old + n_cd45_cell * cd45_repulsion(
                    old_h_val, cd45_height, cd45_k_rep
                )

                dE_bend = bending_energy_delta(
                    h_snap, kappa, dx, gi, gj, old_h_val, new_h_val
                )
                tcr_new = (
                    n_tcr_cell * tcr_pmhc_potential(new_h_val, u_assoc, SIGMA_BIND_NM)
                    if has_pmhc
                    else 0.0
                )
                new_mol_e = tcr_new + n_cd45_cell * cd45_repulsion(
                    new_h_val, cd45_height, cd45_k_rep
                )

                dE = dE_bend + (new_mol_e - old_mol_e)
                total_proposals += 1
                u = u_accepts[k]
                if dE <= 0 or (u > 0.0 and np.log(u) < -dE):
                    accepted += 1
                    accepted_flags[k] = True

            # Pass 3 (apply): restore rejected cells
            for k, (gi, gj) in enumerate(cells):
                if not accepted_flags[k]:
                    h[gi, gj] = old_vals[k]

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
        "dt_seconds": dt,
        "step_size_h_nm": step_size_h,
        "step_size_mol_nm": step_size_mol,
        "D_mol_nm2_per_s": D_mol,
        "D_h_nm2_per_s": D_h,
    }
    result["pmhc_mode"] = pmhc_mode
    if pmhc_radius is not None:
        result["pmhc_radius_nm"] = pmhc_radius
    if snapshot_interval > 0:
        result["snapshots"] = snapshots
    if pmhc_pos is not None:
        result["pmhc_pos"] = pmhc_pos
    return result
