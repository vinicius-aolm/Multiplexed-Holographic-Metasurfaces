# dammann_tools.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
    Gera o mapa de fase para uma grade de Dammann (spot-cloud) usando o algoritmo GS.

    Args:
        P (float): Tamanho do pixel (pixel pitch) em metros.
        wavelength (float): Comprimento de onda em metros.
        supercell_pixels (int): Número de pixels por lado da supercélula.
        n_supercells (int): Número de supercélulas por lado da metassuperfície final.
        iters_gs (int): Número de iterações do algoritmo GS.
        random_seed (int): Semente para a geração de números aleatórios, para reprodutibilidade.
        verbose (bool): Se True, imprime o resumo dos resultados.

    Returns:
        tuple[np.ndarray, dict, list]: Uma tupla contendo:
            - full_phase (np.ndarray): O mapa de fase 2D completo da metassuperfície.
            - metrics (dict): Um dicionário com as métricas 'DE' e 'RMSE'.
            - errors (list): Uma lista com a evolução do erro RMSE a cada iteração.
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
    target_radius = min(1.0 / wavelength, 1.0 / (2.0 * dx))
    target_amp = (K_rad <= target_radius).astype(float)

    # --- Algoritmo GS ---
    plane_field = np.exp(1j * 2.0 * np.pi * np.random.rand(N_super, N_super))
    errors = []
    for it in range(iters_gs):
        far = np.fft.fft2(plane_field)
        far_shift = np.fft.fftshift(far)
        
        amp_current = np.abs(far_shift)
        err = np.sqrt(np.mean((amp_current / (amp_current.max() + 1e-9) - target_amp)**2))
        errors.append(err)
        
        far_shift = target_amp * np.exp(1j * np.angle(far_shift))
        far = np.fft.ifftshift(far_shift)
        
        plane_field = np.fft.ifft2(far)
        plane_field = np.exp(1j * np.angle(plane_field))

    supercell_phase = np.angle(plane_field)

    # --- Construção da Metassuperfície Completa ---
    full_phase = np.tile(supercell_phase, (n_supercells, n_supercells))
    full_field = np.exp(1j * full_phase)
    
    # --- Cálculo de Métricas ---
    N_total = N_super * n_supercells
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
                intensities.append(val)

    intensities = np.array(intensities)
    M = len(intensities)
    
    total_energy = I_far.sum()
    de_sum = intensities.sum() / total_energy
    mean_I = intensities.mean()
    rmse = np.sqrt(np.mean((intensities / mean_I - 1.0)**2))
    
    metrics = {'DE': de_sum, 'RMSE': rmse, 'M_orders': M}
    
    if verbose:
        print("---- Resumo Dammann ----")
        print(f"Supercell: {N_super}x{N_super}, metasurface: {N_total}x{N_total}")
        print(f"Iterações GS: {iters_gs}")
        print(f"Ordens propagantes M = {metrics['M_orders']}")
        print(f"DE ≈ {metrics['DE']:.6f}")
        print(f"RMSE uniformidade = {metrics['RMSE']:.6f}")

    return full_phase, metrics, errors

# --- BLOCO DE EXECUÇÃO PARA TESTE INDEPENDENTE ---
if __name__ == "__main__":
    
    # Chama a função principal para gerar os dados
    full_phase_map, final_metrics, gs_errors = generate_dammann_phase_map(iters_gs=400)
    
    # A partir daqui, o código é apenas para visualização
    
    supercell_view = full_phase_map[:45, :45] # Pega apenas a primeira supercélula para plot
    
    full_field_view = np.exp(1j * full_phase_map)
    I_far_plot = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(full_field_view)))**2 + 1e-12)
    
    # Mapas de fase e difração
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.title("Supercell phase (rad)")
    plt.imshow(supercell_view, cmap='hsv', origin='lower')
    plt.colorbar(label='fase (rad)')

    plt.subplot(1, 3, 2)
    plt.title("Metasurface phase (tile)")
    plt.imshow(full_phase_map, cmap='hsv', origin='lower')
    plt.colorbar(label='fase (rad)')

    plt.subplot(1, 3, 3)
    plt.title("Far-field intensity (log10)")
    plt.imshow(I_far_plot, origin='lower')
    plt.colorbar(label='log10 I')
    plt.tight_layout()
    plt.show()

    # Evolução do erro
    plt.figure(figsize=(6, 4))
    plt.plot(gs_errors)
    plt.xlabel("Iteração GS")
    plt.ylabel("Erro RMSE")
    plt.title("Evolução do erro GS")
    plt.grid(True)
    plt.show()