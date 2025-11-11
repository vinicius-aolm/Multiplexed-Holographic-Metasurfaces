#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase Matching and Heatmap Generation for Metasurface Libraries.

This module provides functions to compute heatmaps of amplitude and phase
distributions and perform phase matching optimization to find the best
metasurface layout for target phase profiles.

Este módulo fornece funções para calcular mapas de calor de distribuições
de amplitude e fase e realizar otimização de casamento de fase para encontrar
o melhor layout de metassuperfície para perfis de fase alvo.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import KDTree

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_heatmaps(
    df: pd.DataFrame,
    x: str = "L_x",
    y: str = "L_y",
    fields: Tuple[str, ...] = ("phase_TE", "amp_TE", "phase_TM", "amp_TM"),
    bins_x: int = 100,
    bins_y: int = 100
) -> Dict[str, np.ndarray]:
    """
    Compute 2D heatmaps for specified fields over L_x and L_y parameter space.
    
    Creates interpolated grid representations of amplitude and phase distributions
    for TE and TM polarizations across the metasurface parameter space.
    
    Calcula mapas de calor 2D para campos especificados sobre o espaço de parâmetros L_x e L_y.
    
    Parameters / Parâmetros:
        df: DataFrame with library data / DataFrame com dados da biblioteca
        x: Column name for x-axis (typically L_x) / Nome da coluna para eixo x (tipicamente L_x)
        y: Column name for y-axis (typically L_y) / Nome da coluna para eixo y (tipicamente L_y)
        fields: Field names to compute heatmaps for / Nomes dos campos para calcular mapas
        bins_x: Number of bins in x direction / Número de bins na direção x
        bins_y: Number of bins in y direction / Número de bins na direção y
        
    Returns / Retorna:
        Dictionary with field names as keys and 2D arrays as values
        Dicionário com nomes de campos como chaves e arrays 2D como valores
        Also includes 'x_grid', 'y_grid' for coordinates
        Também inclui 'x_grid', 'y_grid' para coordenadas
        
    Examples / Exemplos:
        >>> heatmaps = compute_heatmaps(df, fields=("phase_TE", "amp_TE"))
        >>> plt.imshow(heatmaps['phase_TE'], extent=[...])
    """
    # Check required columns
    missing_cols = []
    for col in [x, y] + list(fields):
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Available columns: {df.columns.tolist()}"
        )
    
    logger.info(f"Computing heatmaps for {len(fields)} fields over {x} x {y}")
    
    # Remove NaN values
    df_clean = df[[x, y] + list(fields)].dropna()
    
    if len(df_clean) == 0:
        raise ValueError("No valid data points after removing NaN values")
    
    # Create grid
    x_min, x_max = df_clean[x].min(), df_clean[x].max()
    y_min, y_max = df_clean[y].min(), df_clean[y].max()
    
    x_grid = np.linspace(x_min, x_max, bins_x)
    y_grid = np.linspace(y_min, y_max, bins_y)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # Extract points
    points = df_clean[[x, y]].values
    
    # Interpolate each field
    result = {
        'x_grid': x_grid,
        'y_grid': y_grid,
        'X_grid': X_grid,
        'Y_grid': Y_grid,
    }
    
    for field in fields:
        logger.info(f"  Interpolating {field}...")
        values = df_clean[field].values
        
        # Use linear interpolation
        field_grid = griddata(
            points, values, (X_grid, Y_grid),
            method='linear', fill_value=np.nan
        )
        
        result[field] = field_grid
    
    logger.info(f"Heatmaps computed successfully ({bins_x}x{bins_y} grid)")
    
    return result


def perform_phase_matching(
    df: pd.DataFrame,
    target_phase_tm: np.ndarray,
    target_phase_te: np.ndarray,
    use_height: bool = False,
    height_col: str = "H",
    target_height: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find optimal metasurface layout by matching target TE/TM phase profiles.
    
    For each pixel in the target phase maps, selects the library entry that
    minimizes the quadratic phase error. Optionally filters by height.
    
    Encontra layout ótimo de metassuperfície casando perfis de fase TE/TM alvo.
    
    Parameters / Parâmetros:
        df: Library DataFrame with phase_TE, phase_TM columns
            DataFrame de biblioteca com colunas phase_TE, phase_TM
        target_phase_tm: 2D array of target TM phases
                        Array 2D de fases TM alvo
        target_phase_te: 2D array of target TE phases
                        Array 2D de fases TE alvo
        use_height: Whether to prioritize matching height
                   Se deve priorizar casamento de altura
        height_col: Column name for height parameter
                   Nome da coluna para parâmetro de altura
        target_height: Target height value when use_height=True
                      Valor de altura alvo quando use_height=True
                      
    Returns / Retorna:
        Tuple of (layout_lx, layout_ly, error_map_rms) where:
        - layout_lx: 2D array of selected L_x values
        - layout_ly: 2D array of selected L_y values
        - error_map_rms: 2D array of RMS phase errors
        
        Tupla de (layout_lx, layout_ly, error_map_rms) onde:
        - layout_lx: Array 2D de valores L_x selecionados
        - layout_ly: Array 2D de valores L_y selecionados
        - error_map_rms: Array 2D de erros RMS de fase
        
    Examples / Exemplos:
        >>> lx, ly, err = perform_phase_matching(df, phase_tm, phase_te)
        >>> print(f"Mean error: {err.mean():.4f} rad")
    """
    # Check required columns
    required = ["phase_TE", "phase_TM", "L_x", "L_y"]
    if use_height:
        required.append(height_col)
    
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Available columns: {df.columns.tolist()}"
        )
    
    # Validate target shapes
    if target_phase_te.shape != target_phase_tm.shape:
        raise ValueError(
            f"Target phase shapes must match: "
            f"TE={target_phase_te.shape}, TM={target_phase_tm.shape}"
        )
    
    logger.info(f"Performing phase matching for {target_phase_te.shape} pixels")
    logger.info(f"Library size: {len(df)} entries")
    
    # Filter by height if requested
    if use_height:
        if target_height is None:
            # Use median height from library
            target_height = df[height_col].median()
            logger.info(f"Using median height: {target_height}")
        
        # Filter entries within 10% of target height
        tolerance = 0.1 * target_height
        df_filtered = df[
            (df[height_col] >= target_height - tolerance) &
            (df[height_col] <= target_height + tolerance)
        ].copy()
        
        logger.info(f"Filtered to {len(df_filtered)} entries with H ≈ {target_height}")
        
        if len(df_filtered) == 0:
            logger.warning("No entries match height criterion. Using full library.")
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()
    
    # Remove NaN values
    df_filtered = df_filtered.dropna(subset=required)
    
    if len(df_filtered) == 0:
        raise ValueError("No valid library entries after filtering")
    
    # Build KDTree for fast nearest-neighbor lookup
    phase_points = df_filtered[["phase_TE", "phase_TM"]].values
    tree = KDTree(phase_points)
    
    # Initialize output arrays
    rows, cols = target_phase_te.shape
    layout_lx = np.zeros((rows, cols))
    layout_ly = np.zeros((rows, cols))
    error_map_rms = np.zeros((rows, cols))
    
    # Match each pixel
    total_pixels = rows * cols
    for i in range(rows):
        if i % 10 == 0:
            logger.info(f"  Processing row {i+1}/{rows}...")
        
        for j in range(cols):
            target_point = np.array([target_phase_te[i, j], target_phase_tm[i, j]])
            
            # Find nearest neighbor
            distance, idx = tree.query(target_point)
            
            # Get corresponding L_x, L_y
            best_match = df_filtered.iloc[idx]
            layout_lx[i, j] = best_match["L_x"]
            layout_ly[i, j] = best_match["L_y"]
            
            # Compute RMS error
            error_te = target_phase_te[i, j] - best_match["phase_TE"]
            error_tm = target_phase_tm[i, j] - best_match["phase_TM"]
            error_map_rms[i, j] = np.sqrt(error_te**2 + error_tm**2)
    
    mean_error = error_map_rms.mean()
    logger.info(f"Phase matching complete. Mean RMS error: {mean_error:.6f} rad")
    
    return layout_lx, layout_ly, error_map_rms


def save_heatmap_figures(
    heatmaps: Dict[str, np.ndarray],
    out_dir: Path,
    prefix: str = "heatmap",
    colormap: str = "viridis",
    dpi: int = 300
) -> List[Path]:
    """
    Save heatmap visualizations as PNG files.
    
    Salva visualizações de mapas de calor como arquivos PNG.
    
    Parameters / Parâmetros:
        heatmaps: Dictionary from compute_heatmaps()
                 Dicionário de compute_heatmaps()
        out_dir: Output directory / Diretório de saída
        prefix: Filename prefix / Prefixo do nome do arquivo
        colormap: Matplotlib colormap name / Nome do mapa de cores
        dpi: Resolution in dots per inch / Resolução em pontos por polegada
        
    Returns / Retorna:
        List of saved file paths / Lista de caminhos de arquivos salvos
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    x_grid = heatmaps.get('x_grid')
    y_grid = heatmaps.get('y_grid')
    
    if x_grid is None or y_grid is None:
        raise ValueError("Heatmaps must contain 'x_grid' and 'y_grid'")
    
    extent = [x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()]
    
    for field_name, field_data in heatmaps.items():
        if field_name in ['x_grid', 'y_grid', 'X_grid', 'Y_grid']:
            continue
        
        logger.info(f"Saving heatmap for {field_name}...")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            field_data, origin='lower', extent=extent,
            aspect='auto', cmap=colormap
        )
        ax.set_xlabel('L_x')
        ax.set_ylabel('L_y')
        ax.set_title(f'{field_name} Heatmap')
        plt.colorbar(im, ax=ax)
        
        out_path = out_dir / f"{prefix}_{field_name}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        saved_files.append(out_path)
        
        # Also save array
        np.save(out_dir / f"{prefix}_{field_name}.npy", field_data)
    
    logger.info(f"Saved {len(saved_files)} heatmap figures")
    
    return saved_files


def save_layout_outputs(
    layout_lx: np.ndarray,
    layout_ly: np.ndarray,
    error_map: np.ndarray,
    out_dir: Path,
    prefix: str = "layout"
) -> Dict[str, Path]:
    """
    Save phase matching layout results to files.
    
    Salva resultados de layout de casamento de fase em arquivos.
    
    Parameters / Parâmetros:
        layout_lx: 2D array of L_x values / Array 2D de valores L_x
        layout_ly: 2D array of L_y values / Array 2D de valores L_y
        error_map: 2D array of RMS errors / Array 2D de erros RMS
        out_dir: Output directory / Diretório de saída
        prefix: Filename prefix / Prefixo do nome do arquivo
        
    Returns / Retorna:
        Dictionary with paths to saved files / Dicionário com caminhos dos arquivos salvos
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    # Save CSV files
    logger.info("Saving layout arrays...")
    
    lx_path = out_dir / f"{prefix}_lx.csv"
    np.savetxt(lx_path, layout_lx, delimiter=',', fmt='%.6f')
    saved_paths['layout_lx'] = lx_path
    
    ly_path = out_dir / f"{prefix}_ly.csv"
    np.savetxt(ly_path, layout_ly, delimiter=',', fmt='%.6f')
    saved_paths['layout_ly'] = ly_path
    
    error_path = out_dir / f"{prefix}_error_map.csv"
    np.savetxt(error_path, error_map, delimiter=',', fmt='%.6f')
    saved_paths['error_map'] = error_path
    
    # Save summary figure
    logger.info("Creating summary visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im0 = axes[0].imshow(layout_lx, cmap='viridis')
    axes[0].set_title('Layout L_x')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(layout_ly, cmap='viridis')
    axes[1].set_title('Layout L_y')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(error_map, cmap='hot')
    axes[2].set_title('RMS Error Map')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    
    summary_path = out_dir / f"{prefix}_summary.png"
    fig.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    saved_paths['summary'] = summary_path
    
    logger.info(f"Layout outputs saved to {out_dir}")
    
    return saved_paths


if __name__ == "__main__":
    # Example usage / Exemplo de uso
    import sys
    
    print("Phase matching module loaded successfully")
    print("Use from CLI tools: run_heatmaps.py or run_phase_matching.py")
    print("\nMódulo de casamento de fase carregado com sucesso")
    print("Use das ferramentas CLI: run_heatmaps.py ou run_phase_matching.py")
