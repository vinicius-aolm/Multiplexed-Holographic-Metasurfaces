#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementação de grades de Dammann via FFT para geração de spot clouds.

Código para cálculo de metassuperfícies periódicas usando algoritmo GS.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import json
from typing import Dict, List, Tuple, Optional


# =============================================================================
# Configurações principais - baseadas no nosso setup
# =============================================================================

# Parâmetros padrão do nosso sistema
P = 520e-9              # tamanho do pixel [m]
wavelength = 1064e-9    # comprimento de onda [m]
supercell_pixels = 45   # pixels por supercélula
n_supercells = 10       # número de supercélulas por lado
iters_gs = 400          # iterações do GS
random_seed = 0         # semente para reprodutibilidade


# =============================================================================
# Funções principais do algoritmo (baseadas no notebook)
# =============================================================================

def generate_dammann_phase_map(
    P: float = 520e-9,
    wavelength: float = 1064e-9,
    supercell_pixels: int = 45,
    n_supercells: int = 10,
    iters_gs: int = 400,
    random_seed: int = 0,
    verbose: bool = True
) -> tuple[np.ndarray, dict, list]:
    """
    Gera o mapa de fase para uma grade de Dammann usando o algoritmo GS.
    
    Calcula uma supercélula fase-única que produz um padrão de spots uniforme
    no far-field quando replicada em mosaico.
    """
    np.random.seed(random_seed)

    N_super = supercell_pixels
    dx = P
    d = dx * N_super

    # --- Grade k e alvo ---
    kx = np.fft.fftfreq(N_super, d=dx)
    ky = np.fft.fftfreq(N_super, d=dx)
    kx_shift = np.fft.fftshift(kx)
    ky_shift = np.fft.fftshift(ky)
    KX, KY = np.meshgrid(kx_shift, ky_shift)
    K_rad = np.sqrt(KX**2 + KY**2)
    
    # Raio de corte limitado por λ e Nyquist
    target_radius = min(1.0 / wavelength, 1.0 / (2.0 * dx))
    target_amp = (K_rad <= target_radius).astype(float)

    # --- Algoritmo GS ---
    plane_field = np.exp(1j * 2.0 * np.pi * np.random.rand(N_super, N_super))
    errors = []
    
    for it in range(iters_gs):
        # Propagação para o far-field
        far = np.fft.fft2(plane_field)
        far_shift = np.fft.fftshift(far)
        
        # Calcula erro
        amp_current = np.abs(far_shift)
        amp_norm = amp_current / (amp_current.max() if amp_current.max() > 0 else 1.0)
        err = np.sqrt(np.mean((amp_norm - target_amp)**2))
        errors.append(err)
        
        # Aplica restrição de amplitude
        far_shift = target_amp * np.exp(1j * np.angle(far_shift))
        far = np.fft.ifftshift(far_shift)
        plane_field = np.fft.ifft2(far)
        
        # Fase única no plano da supercélula
        plane_field = np.exp(1j * np.angle(plane_field))

    supercell_phase = np.angle(plane_field)

    # --- Construção da metassuperfície completa ---
    full_phase = np.tile(supercell_phase, (n_supercells, n_supercells))
    full_field = np.exp(1j * full_phase)
    
    # --- Cálculo de métricas ---
    N_total = N_super * n_supercells
    FF_full = np.fft.fft2(full_field)
    FF_full_shift = np.fft.fftshift(FF_full)
    I_far = np.abs(FF_full_shift)**2
    
    # Encontra ordens propagantes
    p_max = int(np.floor(d / wavelength))
    orders, intensities = [], []
    kx_full_shift = np.fft.fftshift(np.fft.fftfreq(N_total, d=dx))
    ky_full_shift = np.fft.fftshift(np.fft.fftfreq(N_total, d=dx))

    for p in range(-p_max, p_max + 1):
        for q in range(-p_max, p_max + 1):
            sx, sy = p * wavelength / d, q * wavelength / d
            if (sx**2 + sy**2) <= 1.0:
                fx_target, fy_target = p / d, q / d
                ix = np.argmin(np.abs(kx_full_shift - fx_target))
                iy = np.argmin(np.abs(ky_full_shift - fy_target))
                val = I_far[iy, ix]
                intensities.append(val)

    intensities = np.array(intensities)
    M = len(intensities)
    
    # Calcula eficiência e uniformidade
    total_energy = I_far.sum()
    de_sum = intensities.sum() / total_energy if total_energy > 0 else 0.0
    mean_I = intensities.mean() if M > 0 else 1.0
    rmse = np.sqrt(np.mean((intensities / mean_I - 1.0)**2)) if M > 0 else 0.0
    
    metrics = {'DE': de_sum, 'RMSE': rmse, 'M_orders': M}
    
    if verbose:
        print("---- Resumo Dammann ----")
        print(f"Supercell: {N_super}x{N_super}, metasurface: {N_total}x{N_total}")
        print(f"Iterações GS: {iters_gs}")
        print(f"Ordens propagantes M = {metrics['M_orders']}")
        print(f"DE ≈ {metrics['DE']:.6f}")
        print(f"RMSE uniformidade = {metrics['RMSE']:.6f}")

    return full_phase, metrics, errors


def calculate_diffraction_orders(full_phase, P, wavelength, supercell_pixels, n_supercells):
    """
    Calcula as ordens de difração e suas intensidades a partir do mapa de fase.
    """
    N_super = supercell_pixels
    dx = P
    d = dx * N_super
    N_total = N_super * n_supercells
    
    full_field = np.exp(1j * full_phase)
    FF_full = np.fft.fft2(full_field)
    FF_full_shift = np.fft.fftshift(FF_full)
    I_far = np.abs(FF_full_shift)**2
    
    p_max = int(np.floor(d / wavelength))
    orders, intensities = [], []
    kx_full_shift = np.fft.fftshift(np.fft.fftfreq(N_total, d=dx))
    ky_full_shift = np.fft.fftshift(np.fft.fftfreq(N_total, d=dx))

    for p in range(-p_max, p_max + 1):
        for q in range(-p_max, p_max + 1):
            sx, sy = p * wavelength / d, q * wavelength / d
            if (sx**2 + sy**2) <= 1.0:
                fx_target, fy_target = p / d, q / d
                ix = np.argmin(np.abs(kx_full_shift - fx_target))
                iy = np.argmin(np.abs(ky_full_shift - fy_target))
                val = I_far[iy, ix]
                orders.append((p, q))
                intensities.append(val)
    
    return orders, intensities, I_far


# =============================================================================
# Utilitários para organização de arquivos e pastas
# =============================================================================

def find_repo_root(start: Path = Path.cwd()) -> Path:
    """Encontra a raiz do repositório git."""
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    return start


def _nm(x_m: float) -> int:
    """Converte metros para nanômetros."""
    return int(round(float(x_m) * 1e9))


def rich_name(base: str, experiment: str, pol_label: str) -> str:
    """Cria nome de arquivo descritivo com parâmetros físicos."""
    return (f"{base}__{experiment}"
            f"__{pol_label}"
            f"__λ_{_nm(wavelength)}nm"
            f"__P_{_nm(P)}nm"
            f"__scpix_{supercell_pixels}px"
            f"__nsc_{n_supercells}"
            f"__iter_{iters_gs}"
            f"__seed_{random_seed}")


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    """Salva figura e fecha para liberar memória."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Pipeline principal
# =============================================================================

def run_dammann_pipeline(out_dir: Path, experiment: str, pol_label: str = "Y") -> Dict:
    """
    Pipeline completo para geração de grade de Dammann.
    
    Gera as figuras de análise e salva na pasta de resultados.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Dammann] Gerando fase da supercélula...")
    full_phase, metrics, errors = generate_dammann_phase_map(
        P=P,
        wavelength=wavelength,
        supercell_pixels=supercell_pixels,
        n_supercells=n_supercells,
        iters_gs=iters_gs,
        random_seed=random_seed,
        verbose=True
    )

    print(f"[Dammann] Calculando ordens de difração...")
    orders, intensities, I_far = calculate_diffraction_orders(
        full_phase, P, wavelength, supercell_pixels, n_supercells
    )

    # Prepara dados para tabela
    df = pd.DataFrame(orders, columns=["p", "q"])
    df["intensity"] = intensities

    # Gera nomes para os arquivos
    phase_stem = rich_name("phase_map", experiment, pol_label)
    errors_stem = rich_name("convergence", experiment, pol_label)
    orders_stem = rich_name("diffraction_orders", experiment, pol_label)
    summary_stem = rich_name("summary", experiment, pol_label)

    # Salva arrays numéricos
    np.savetxt(out_dir / f"{phase_stem}.txt", full_phase)
    np.savetxt(out_dir / f"{errors_stem}.txt", errors)
    np.savetxt(out_dir / f"{orders_stem}_intensity.txt", I_far)
    df.to_csv(out_dir / f"{orders_stem}_table.csv", index=False)

    # 1) Mapas de fase e difração (3 painéis)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Supercélula
    supercell_view = full_phase[:supercell_pixels, :supercell_pixels]
    im0 = axes[0].imshow(supercell_view, cmap='twilight', origin='lower')
    axes[0].set_title("Fase da supercélula (rad)")
    plt.colorbar(im0, ax=axes[0])
    
    # Metassuperfície
    im1 = axes[1].imshow(full_phase, cmap='twilight', origin='lower')
    axes[1].set_title("Metassuperfície completa")
    plt.colorbar(im1, ax=axes[1])
    
    # Far-field
    I_far_plot = np.log10(I_far + 1e-12)
    im2 = axes[2].imshow(I_far_plot, origin='lower')
    axes[2].set_title("Far-field (log10)")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    _save_fig(fig, out_dir / f"{phase_stem}.png")

    # 2) Convergência do algoritmo
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(errors)
    ax.set_xlabel("Iteração GS")
    ax.set_ylabel("Erro RMSE")
    ax.set_title("Convergência do algoritmo GS")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out_dir / f"{errors_stem}.png")

    # 3) Ordens de difração
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df["p"], df["q"], c=df["intensity"], 
                        cmap="viridis", s=50, edgecolor="k")
    ax.set_xlabel("Ordem p")
    ax.set_ylabel("Ordem q")
    ax.set_title("Ordens de difração propagantes")
    plt.colorbar(scatter, ax=ax, label="Intensidade")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out_dir / f"{orders_stem}.png")

    # 4) Resumo completo (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Supercélula
    im0 = axes[0, 0].imshow(supercell_view, cmap='twilight', origin='lower')
    axes[0, 0].set_title("Supercélula")
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Far-field
    im1 = axes[0, 1].imshow(I_far_plot, origin='lower')
    axes[0, 1].set_title("Far-field")
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Convergência
    axes[1, 0].plot(errors)
    axes[1, 0].set_xlabel("Iteração")
    axes[1, 0].set_ylabel("Erro RMSE")
    axes[1, 0].set_title("Convergência GS")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Ordens
    scatter = axes[1, 1].scatter(df["p"], df["q"], c=df["intensity"], 
                                cmap="viridis", s=30, edgecolor="k")
    axes[1, 1].set_xlabel("Ordem p")
    axes[1, 1].set_ylabel("Ordem q")
    axes[1, 1].set_title("Ordens de difração")
    plt.colorbar(scatter, ax=axes[1, 1])
    
    plt.tight_layout()
    _save_fig(fig, out_dir / f"{summary_stem}.png")

    return {
        "phase_map": full_phase,
        "metrics": metrics,
        "errors": errors,
        "orders_df": df,
        "I_far": I_far,
        "dir": out_dir,
    }


def run_dammann_batch(out_root: Path, experiment: str, pol_label: str = "Y") -> Dict:
    """
    Executa o pipeline Dammann e organiza os resultados.
    """
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = out_root / experiment / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Iniciando experimento Dammann: {experiment}")
    print(f"Parâmetros: λ={wavelength:.2e}m, P={P:.2e}m, supercélula={supercell_pixels}px")

    result = run_dammann_pipeline(run_dir, experiment, pol_label)

    # Salva metadados
    meta = {
        "experiment": experiment,
        "run_id": run_id,
        "pol": pol_label,
        "wavelength_nm": _nm(wavelength),
        "P_nm": _nm(P),
        "supercell_pixels": supercell_pixels,
        "n_supercells": n_supercells,
        "iterations": iters_gs,
        "random_seed": random_seed,
        "metrics": result["metrics"],
    }
    
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n=== Resumo da execução ===")
    print(f"DE (eficiência): {result['metrics']['DE']:.6f}")
    print(f"RMSE (uniformidade): {result['metrics']['RMSE']:.6f}")
    print(f"Ordens propagantes: {result['metrics']['M_orders']}")
    print(f"Resultados em: {result['dir']}")

    return {"run_dir": run_dir, "result": result}


# =============================================================================
# Interface de linha de comando
# =============================================================================

def parse_args():
    """Configura os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Geração de grades de Dammann via FFT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Parâmetros físicos
    parser.add_argument("--wavelength", type=float, default=wavelength,
                       help="Comprimento de onda [m]")
    parser.add_argument("--P", type=float, default=P,
                       help="Tamanho do pixel [m]")
    parser.add_argument("--supercell_pixels", type=int, default=supercell_pixels,
                       help="Pixels por supercélula")
    parser.add_argument("--n_supercells", type=int, default=n_supercells,
                       help="Número de supercélulas por lado")
    parser.add_argument("--iters", type=int, default=iters_gs,
                       help="Número de iterações do GS")
    parser.add_argument("--seed", type=int, default=random_seed,
                       help="Semente para números aleatórios")

    # Arquivos e pastas
    parser.add_argument("--experiment", type=str, default="demo_dammann",
                       help="Nome do experimento")
    parser.add_argument("--out_root", type=Path, 
                       default=find_repo_root() / "results" / "holography-dammann" / "gs_y",
                       help="Pasta base para resultados")
    parser.add_argument("--pol", type=str, default="Y", choices=["X", "Y"],
                       help="Polarização para naming")

    return parser.parse_args()


def main():
    """Função principal."""
    args = parse_args()
    
    # Atualiza parâmetros globais
    global wavelength, P, supercell_pixels, n_supercells, iters_gs, random_seed
    wavelength = args.wavelength
    P = args.P
    supercell_pixels = args.supercell_pixels
    n_supercells = args.n_supercells
    iters_gs = args.iters
    random_seed = args.seed

    # Executa pipeline
    result = run_dammann_batch(
        out_root=args.out_root,
        experiment=args.experiment,
        pol_label=args.pol
    )

    print(f"\n✅ Experimento Dammann finalizado!")
    print(f"📁 Resultados em: {result['run_dir']}")


if __name__ == "__main__":
    main()