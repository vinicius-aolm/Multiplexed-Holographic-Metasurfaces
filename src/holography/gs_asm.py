#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Implementação do Gerchberg-Saxton com Método do Espectro Angular para holografia.

Código para cálculo de hologramas de fase usando GS+ASM.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from PIL import Image
from pathlib import Path
from datetime import datetime
import argparse
import json
from typing import Dict, List, Tuple, Optional


# =============================================================================
# Configurações principais - baseadas no nosso setup
# =============================================================================

# Parâmetros padrão do nosso sistema
wavelength = 1064e-9    # comprimento de onda [m]
z = 380e-6              # distância de propagação [m] 
dx = 520e-9             # tamanho do pixel no SLM [m]
NA = 0.65               # abertura numérica
num_iter = 200          # iterações do GS


# =============================================================================
# Funções principais do algoritmo (mantidas do notebook)
# =============================================================================

def load_and_preprocess_image(image_path, target_size=(450, 450)):
    """
    Carrega e pré-processa a imagem alvo.
    
    Converte para cinza, redimensiona e normaliza. Se a imagem não for encontrada,
    cria um padrão de teste simples em formato de 'H'.
    """
    try:
        image = Image.open(image_path).convert('L')
        image = image.resize(target_size, Image.LANCZOS)
        image_array = np.array(image, dtype=np.float64)
        
        max_val = np.max(image_array) if np.max(image_array) > 0 else 1.0
        return image_array / max_val
        
    except FileNotFoundError:
        print(f"Target image not found. Creating a simple synthetic logo-like pattern for testing...")
        w, h = target_size
        target_image = np.zeros((h, w), dtype=np.float64)
        
        # Simple block shapes: a vertical bar + two horizontal bars
        vbar_w = max(1, w // 9)
        target_image[h//3:2*h//3, w//6:w//6 + vbar_w] = 1.0
        target_image[h//3:h//3 + vbar_w, w//6:5*w//6] = 1.0
        target_image[2*h//3 - vbar_w:2*h//3, w//6:5*w//6] = 1.0
        
        return target_image


def apply_zero_padding(image, padding_factor=2):
    """Coloca a imagem no centro de uma matriz maior com zeros ao redor."""
    original_size = image.shape
    padded_size = (image.shape[0] * padding_factor, image.shape[1] * padding_factor)
    
    padded_image = np.zeros(padded_size, dtype=complex)
    start_row = (padded_size[0] - original_size[0]) // 2
    start_col = (padded_size[1] - original_size[1]) // 2
    
    padded_image[start_row:start_row+original_size[0], 
                 start_col:start_col+original_size[1]] = image
    
    return padded_image, original_size


def create_low_pass_filter(shape, wavelength, dx, NA):
    """Cria máscara circular no domínio da frequência baseada na NA."""
    nx, ny = shape
    fx = np.fft.fftfreq(nx, dx)
    fy = np.fft.fftfreq(ny, dx)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    
    f_cutoff = NA / wavelength
    freq_radius = np.sqrt(FX**2 + FY**2)
    
    return (freq_radius <= f_cutoff).astype(np.float64)


def angular_spectrum_propagation(U, wavelength, z, dx, filter_mask=None):
    """Propaga campo usando método do espectro angular."""
    k = 2 * np.pi / wavelength
    nx, ny = U.shape
    
    fx = np.fft.fftfreq(nx, dx)
    fy = np.fft.fftfreq(ny, dx)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')

    root_term = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
    root_term[root_term < 0] = 0
    
    H = np.exp(1j * k * z * np.sqrt(root_term))
    
    if filter_mask is not None:
        H *= filter_mask

    U_freq = fft2(U)
    U_prop = ifft2(U_freq * H)
    
    return U_prop


def calculate_correlation(target, reconstructed):
    """Calcula correlação de Pearson entre duas imagens."""
    target_real = np.real(target).flatten()
    reconstructed_real = np.real(reconstructed).flatten()
    
    correlation = np.corrcoef(target_real, reconstructed_real)[0, 1]
    return 0.0 if np.isnan(correlation) else float(correlation)


def extract_center(image, original_size):
    """Extrai região central da imagem (remove o padding)."""
    nx, ny = original_size
    start_row = (image.shape[0] - nx) // 2
    start_col = (image.shape[1] - ny) // 2
    return image[start_row:start_row+nx, start_col:start_col+ny]


def gerchberg_saxton_angular_spectrum(target, wavelength, z, dx, NA, num_iter=50):
    """Algoritmo de Gerchberg-Saxton para hologramas de fase única."""
    target_padded, original_size = apply_zero_padding(target)
    nx_pad, ny_pad = target_padded.shape

    filter_mask = create_low_pass_filter((nx_pad, ny_pad), wavelength, dx, NA)

    phase = np.random.rand(nx_pad, ny_pad) * 2 * np.pi
    U = target_padded * np.exp(1j * phase)

    correlations = []

    for i in range(num_iter):
        U_image = angular_spectrum_propagation(U, wavelength, z, dx, filter_mask)

        amplitude_image = np.abs(U_image)
        phase_image = np.angle(U_image)

        target_region = extract_center(target_padded, original_size)
        recon_region = extract_center(amplitude_image, original_size)
        corr = calculate_correlation(target_region, recon_region)
        correlations.append(corr)

        U_image_updated = target_padded * np.exp(1j * phase_image)

        U = angular_spectrum_propagation(U_image_updated, wavelength, -z, dx, filter_mask)

        phase_hologram = np.angle(U)
        U = np.exp(1j * phase_hologram)

        if (i + 1) % 10 == 0:
            print(f"Iteração {i+1}/{num_iter} — Correlação: {corr:.4f}")

    phase_final = extract_center(np.angle(U), original_size)
    
    return phase_final, correlations


def reconstruct_image(phase_map, wavelength, z, dx, NA):
    """Simula a reconstrução a partir do mapa de fase."""
    phase_padded, original_size = apply_zero_padding(np.exp(1j * phase_map))
    filter_mask = create_low_pass_filter(phase_padded.shape, wavelength, dx, NA)
    
    reconstructed = angular_spectrum_propagation(phase_padded, wavelength, z, dx, filter_mask)
    reconstructed = extract_center(np.abs(reconstructed), original_size)
    return np.real(reconstructed)


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


def _um(x_m: float) -> int:
    """Converte metros para micrômetros."""
    return int(round(float(x_m) * 1e6))


def rich_name(base: str, target_name: str, pol_label: str) -> str:
    """Cria nome de arquivo descritivo com parâmetros físicos."""
    return (f"{base}__{target_name}"
            f"__{pol_label}"
            f"__λ_{_nm(wavelength)}nm"
            f"__z_{_um(z)}um"
            f"__dx_{_nm(dx)}nm"
            f"__iter_{int(num_iter)}")


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    """Salva figura e fecha para liberar memória."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Pipeline principal
# =============================================================================

def run_pipeline_for_target(name: str, path: Path, out_dir: Path, pol_label: str = "X", 
                           target_size: Tuple[int, int] = (450, 450)) -> Dict:
    """
    Pipeline completo GS+ASM para um alvo.
    
    Gera as 5 figuras por alvo e salva na pasta do alvo.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{name}] Carregando alvo: {path}")
    target_img = load_and_preprocess_image(str(path), target_size=target_size)

    print(f"[{name}] Executando GS+ASM…")
    phase_map, correlations = gerchberg_saxton_angular_spectrum(
        target_img, wavelength, z, dx, NA, num_iter=num_iter
    )

    print(f"[{name}] Reconstruindo…")
    recon = reconstruct_image(phase_map, wavelength, z, dx, NA)

    # Gera nomes para os arquivos
    phase_stem = rich_name("phase_map", name, pol_label)
    corr_stem = rich_name("correlations", name, pol_label)
    targ_stem = rich_name("imagem_alvo", name, pol_label)
    recon_stem = rich_name("reconstruida", name, pol_label)
    conv_stem = rich_name("convergencia", name, pol_label)
    sumario_stem = rich_name("sumario_alvo", name, pol_label)

    # Salva arrays numéricos
    np.savetxt(out_dir / f"{phase_stem}.txt", phase_map)
    np.savetxt(out_dir / f"{corr_stem}.txt", correlations)

    # 1) Imagem alvo (individual)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(target_img, cmap="gray")
    ax.set_title("Imagem alvo")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_fig(fig, out_dir / f"{targ_stem}.png")

    # 2) Mapa de fase (individual)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(phase_map, cmap="hsv")
    ax.set_title(f"Mapa de fase (pol {pol_label})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_fig(fig, out_dir / f"{phase_stem}.png")

    # 3) Reconstrução (individual)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(recon, cmap="gray")
    ax.set_title("Imagem reconstruída")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_fig(fig, out_dir / f"{recon_stem}.png")

    # 4) Convergência (individual)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(correlations)
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Correlação de Pearson")
    ax.set_title("Convergência do algoritmo")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, out_dir / f"{conv_stem}.png")

    # 5) Resumo (2x2) com os quatro acima
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    
    # Imagem alvo
    im = axes[0, 0].imshow(target_img, cmap="gray")
    axes[0, 0].set_title("Imagem alvo")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # Mapa de fase
    im = axes[0, 1].imshow(phase_map, cmap="hsv")
    axes[0, 1].set_title(f"Mapa de fase (pol {pol_label})")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Reconstruída
    im = axes[1, 0].imshow(recon, cmap="gray")
    axes[1, 0].set_title("Imagem reconstruída")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Convergência
    axes[1, 1].plot(correlations)
    axes[1, 1].set_xlabel("Iteração")
    axes[1, 1].set_ylabel("Correlação de Pearson")
    axes[1, 1].set_title("Convergência")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_fig(fig, out_dir / f"{sumario_stem}.png")

    return {
        "name": name,
        "target": target_img,
        "phase": phase_map,
        "recon": recon,
        "corrs": correlations,
        "dir": out_dir,
    }


def run_batch(targets: List[Tuple[str, Path]], out_root: Path, experiment: str,
             pol_label: str, print_summary: bool = True) -> Dict:
    """
    Executa GS+ASM para vários alvos.
    
    Cria a estrutura de pastas conforme o repositório e gera sumário geral
    apenas se houver até 2 alvos.
    """
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = out_root / experiment / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, path in targets:
        target_dir = run_dir / name
        res = run_pipeline_for_target(name, path, target_dir, pol_label)
        results.append(res)

    # Gera sumário geral apenas se houver até 2 alvos
    if len(targets) <= 2:
        n_cols = len(results)
        fig, axes = plt.subplots(4, n_cols, figsize=(6*n_cols, 16))
        
        if n_cols == 1:
            axes = axes.reshape(4, 1)

        for col, res in enumerate(results):
            name, target_img, phase_map, recon, corrs = res["name"], res["target"], res["phase"], res["recon"], res["corrs"]

            # Linha 1: Alvo
            im0 = axes[0, col].imshow(target_img, cmap="gray")
            axes[0, col].set_title(f"Alvo ({name})")
            plt.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)
            
            # Linha 2: Mapa de fase
            im1 = axes[1, col].imshow(phase_map, cmap="hsv")
            axes[1, col].set_title(f"Mapa de fase ({name})")
            plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)
            
            # Linha 3: Reconstruída
            im2 = axes[2, col].imshow(recon, cmap="gray")
            axes[2, col].set_title(f"Reconstruída ({name})")
            plt.colorbar(im2, ax=axes[2, col], fraction=0.046, pad=0.04)
            
            # Linha 4: Convergência
            axes[3, col].plot(corrs)
            axes[3, col].set_xlabel("Iteração")
            axes[3, col].set_ylabel("Correlação")
            axes[3, col].set_title(f"Convergência ({name})")
            axes[3, col].grid(True, alpha=0.3)

        plt.tight_layout()
        
        # Nome do sumário geral
        target_names = "__".join([name for name, _ in targets])
        summary_stem = rich_name(f"summary_{experiment}", target_names, pol_label)
        summary_path = run_dir / f"{summary_stem}.png"
        _save_fig(fig, summary_path)
    else:
        summary_path = None
        print(f"Sumário geral não gerado para {len(targets)} alvos (muitos para visualização)")

    # Salva metadados
    meta = {
        "experiment": experiment,
        "run_id": run_id,
        "pol": pol_label,
        "wavelength_nm": _nm(wavelength),
        "z_um": _um(z),
        "dx_nm": _nm(dx),
        "iterations": int(num_iter),
        "targets": [str(p) for _, p in targets],
        "summary_png": str(summary_path) if summary_path else None,
    }
    
    with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if print_summary:
        print("\n=== Resumo da execução ===")
        print(json.dumps(meta, indent=2, ensure_ascii=False))

    return {"run_dir": run_dir, "results": results, "summary": summary_path}


# =============================================================================
# Interface de linha de comando
# =============================================================================

def parse_args():
    """Configura os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Gerchberg-Saxton com Espectro Angular",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Parâmetros físicos (agora são opcionais, com defaults)
    parser.add_argument("--wavelength", type=float, default=wavelength, 
                       help="Comprimento de onda [m]")
    parser.add_argument("--z", type=float, default=z,
                       help="Distância de propagação [m]")
    parser.add_argument("--dx", type=float, default=dx,
                       help="Tamanho do pixel no SLM [m]")
    parser.add_argument("--NA", type=float, default=NA,
                       help="Abertura numérica")
    parser.add_argument("--iters", type=int, default=num_iter,
                       help="Número de iterações do GS")

    # Arquivos e pastas
    parser.add_argument("--targets_dir", type=Path, 
                       default=find_repo_root() / "data" / "targets" / "common",
                       help="Pasta com imagens alvo")
    parser.add_argument("--targets", type=str, nargs="+", default=["ilum.png", "ufabc.png"],
                       help="Nomes dos arquivos alvo")
    parser.add_argument("--experiment", type=str, default="demo_holografia",
                       help="Nome do experimento")
    parser.add_argument("--out_root", type=Path, 
                       default=find_repo_root() / "results" / "holography-dammann" / "gs_x",
                       help="Pasta base para resultados")
    parser.add_argument("--pol", type=str, default="X", choices=["X", "Y"],
                       help="Polarização para naming")

    return parser.parse_args()


def main():
    """Função principal."""
    args = parse_args()
    
    # Atualiza parâmetros globais com os valores dos argumentos
    global wavelength, z, dx, NA, num_iter
    wavelength = args.wavelength
    z = args.z
    dx = args.dx
    NA = args.NA
    num_iter = args.iters

    # Prepara lista de alvos
    targets = [(Path(target).stem, args.targets_dir / target) for target in args.targets]

    print(f"Iniciando experimento: {args.experiment}")
    print(f"Alvos: {[name for name, _ in targets]}")
    print(f"Parâmetros: λ={wavelength:.2e}m, z={z:.2e}m, NA={NA}, iterações={num_iter}")

    # Executa pipeline
    result = run_batch(
        targets=targets,
        out_root=args.out_root,
        experiment=args.experiment,
        pol_label=args.pol,
        print_summary=True
    )

    print(f"\n✅ Experimento finalizado!")
    print(f"📁 Resultados em: {result['run_dir']}")
    
    if result['summary']:
        print(f"📊 Sumário geral: {result['summary']}")


if __name__ == "__main__":
    main()