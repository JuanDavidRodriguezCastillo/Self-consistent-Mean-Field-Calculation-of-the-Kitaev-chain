#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu sep 25 10:45:20 2025

@author:jrodri
"""


import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt


def calcular_mu_efectivo(n_old, mu, U, N, out):
   
    out[0] = mu - 2 * U * (2 * n_old[1] - 1)
    out[-1] = mu - 2 * U * (2 * n_old[-2] - 1)
    if N > 2:
        out[1:-1] = mu - 4 * U * (n_old[:-2] + n_old[2:] - 1)
    return out


def construir_bloque_estatico(N, w, Delta):
   
    H = np.zeros((2 * N, 2 * N), dtype=float)
    off_idx = np.arange(N - 1)
    H[off_idx, off_idx + 1] = -w
    H[off_idx + 1, off_idx] = -w
    H[off_idx + N, off_idx + N + 1] = w
    H[off_idx + N + 1, off_idx + N] = w
    H[off_idx, off_idx + N + 1] = -Delta
    H[off_idx + N + 1, off_idx] = -Delta
    H[off_idx + 1, off_idx + N] = Delta
    H[off_idx + N, off_idx + 1] = Delta
    return H


def simulacion_campo_medio(N, w, Delta, mu, U, H_static, n_init=None,
                            tol=1e-10, max_iter=600,
                            mixing_fast=1.0, mixing_slow=0.35,
                            patience=150, tail=40, verbose=False):
   
    H = H_static.copy()
    idx = np.arange(N)
    mu_eff = np.empty(N)
    n_old = np.full(N, 0.5) if n_init is None else n_init.copy()

    mixing = mixing_fast
    best_diff = np.inf
    stall = 0
    buf = deque(maxlen=tail)
    converged = False
    evals = None

    for iteracion in range(max_iter):
        calcular_mu_efectivo(n_old, mu, U, N, mu_eff)
        H[idx, idx] = -mu_eff
        H[idx + N, idx + N] = mu_eff

        evals, evecs = np.linalg.eigh(H)
        n_new = np.sum(evecs[N:, N:] ** 2, axis=1)
        diff = np.max(np.abs(n_new - n_old))
        buf.append(n_old)

        if diff < tol:
            n_old = n_new
            converged = True
            break

        if diff < best_diff * 0.999:
            best_diff = diff
            stall = 0
        else:
            stall += 1
            if stall > patience and mixing != mixing_slow:
                mixing = mixing_slow
                stall = 0

        n_old = n_old + mixing * (n_new - n_old)

    if not converged:
        n_old = np.mean(buf, axis=0)
        calcular_mu_efectivo(n_old, mu, U, N, mu_eff)
        H[idx, idx] = -mu_eff
        H[idx + N, idx + N] = mu_eff
        evals, evecs = np.linalg.eigh(H)

    energia_minima = np.min(np.abs(evals))

    if verbose:
        tag = "convergio" if converged else "NO convergio (fallback promediado)"
        print(f"{tag} en {iteracion + 1} iteraciones, E_gs={energia_minima:.6f}")
    print(f'N={sum(n_old)}')
    return {'E_gs': energia_minima, 'n': n_old, 'iters': iteracion + 1, 'converged': converged}


def _procesar_fila(args):
    """Worker: sweeps mu for one fixed U. Runs in its own process."""
    i_in, current_u, mus, N_sites, w, Delta, warm_start, solver_kwargs = args
    H_static = construir_bloque_estatico(N_sites, w, Delta)
    fila = np.empty(len(mus))
    n_no_conv = 0
    n_guess = None
    for j_in, current_mu in enumerate(mus):
        resultado = simulacion_campo_medio(N_sites, w, Delta, current_mu, current_u,
                                            H_static, n_init=n_guess, **solver_kwargs)
        fila[j_in] = resultado['E_gs']
        if not resultado['converged']:
            n_no_conv += 1
        if warm_start:
            n_guess = resultado['n']
    return i_in, fila, n_no_conv


def run_grid(N_sites, w, Delta, mus, us, use_multiprocessing=True, warm_start=False,
             max_workers=None, **solver_kwargs):
    E = np.zeros((len(us), len(mus)))
    n_unconverged_total = 0
    tareas = [(i_in, current_u, mus, N_sites, w, Delta, warm_start, solver_kwargs)
              for i_in, current_u in enumerate(us)]

    if use_multiprocessing:
        max_workers = max_workers or (os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_procesar_fila, t): t[0] for t in tareas}
            n_done = 0
            for future in as_completed(futures):
                i_in, fila, n_no_conv = future.result()
                E[i_in, :] = fila
                n_unconverged_total += n_no_conv
                n_done += 1
                print(f'Advance: {round(100 * n_done / len(us), 2):.2f}%')
    else:
        for n_done, t in enumerate(tareas, start=1):
            i_in, fila, n_no_conv = _procesar_fila(t)
            E[i_in, :] = fila
            n_unconverged_total += n_no_conv
            print(f'Advance: {round(100 * n_done / len(us), 2):.2f}%')

    return E, n_unconverged_total


if __name__ == "__main__":

    # --- user-facing switches -------------------------------------------
    USE_MULTIPROCESSING = True   # parallelize across U-rows (recommended)
    WARM_START = False           # see module docstring before turning on
    # ----------------------------------------------------------------------

    mus = np.linspace(0, 3, 30)
    us = np.linspace(0, 1, 30)
    N_sites = 100
    w = 1
    Delta = 1

    t0 = time.time()
    E, n_unconverged = run_grid(
        N_sites, w, Delta, mus, us,
        use_multiprocessing=USE_MULTIPROCESSING,
        warm_start=WARM_START,
        tol=1e-10, max_iter=600, mixing_fast=1.0, mixing_slow=0.35,
        patience=150, tail=40,
    )
    elapsed = time.time() - t0

    print(f'Advance: 100.00%  (tiempo total: {elapsed:.1f} s)')
    n_total = len(mus) * len(us)
    print(f'{n_unconverged}/{n_total} puntos no convergieron a tol=1e-10 '
          f'(se uso el promedio de las ultimas iteraciones; ver docstring del script).')

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(us, mus, E.T, shading='auto', cmap='viridis')
    plt.colorbar(label='Menor energia de excitacion ($E_{gs}$)')
    plt.xlabel('U')
    plt.ylabel(r'$\mu$')
    plt.title('Diagrama de Fases Topologicas')
    plt.savefig('phase_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()