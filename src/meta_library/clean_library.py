#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Library Cleaning and Derived Column Generation.

This module provides functions to clean metasurface library DataFrames
and compute derived columns such as transmission amplitudes and phases
for TE and TM polarizations.

Este módulo fornece funções para limpar DataFrames de biblioteca de
metassuperfície e calcular colunas derivadas como amplitudes e fases
de transmissão para polarizações TE e TM.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def append_derived_columns(
    df: pd.DataFrame,
    te_cols: Optional[Tuple[str, str]] = None,
    tm_cols: Optional[Tuple[str, str]] = None,
    unwrap_phase: bool = False,
    phase_unit: str = "rad"
) -> pd.DataFrame:
    """
    Append derived transmission amplitude and phase columns for TE/TM polarizations.
    
    Computes complex S-parameters from real/imaginary components and extracts
    amplitude (|T|) and phase (∠T) for TE and TM modes.
    
    Adiciona colunas derivadas de amplitude e fase de transmissão para polarizações TE/TM.
    
    Parameters / Parâmetros:
        df: Input DataFrame with S-parameter columns
            DataFrame de entrada com colunas de parâmetros S
        te_cols: Tuple of (real_col, imag_col) for TE transmission
                 Tupla de (col_real, col_imag) para transmissão TE
                 Default: ("S21_real", "S21_imag")
        tm_cols: Tuple of (real_col, imag_col) for TM transmission
                 Tupla de (col_real, col_imag) para transmissão TM
                 Default: ("S12_real", "S12_imag")
        unwrap_phase: Whether to unwrap phase discontinuities
                      Se deve desembrulhar descontinuidades de fase
        phase_unit: "rad" for radians or "deg" for degrees
                    "rad" para radianos ou "deg" para graus
                    
    Returns / Retorna:
        DataFrame with added columns: amp_TE, phase_TE, amp_TM, phase_TM,
        S_complex_TE, S_complex_TM
        DataFrame com colunas adicionadas: amp_TE, phase_TE, amp_TM, phase_TM,
        S_complex_TE, S_complex_TM
        
    Examples / Exemplos:
        >>> df_clean = append_derived_columns(df, unwrap_phase=True, phase_unit="deg")
        >>> print(df_clean[['amp_TE', 'phase_TE', 'amp_TM', 'phase_TM']].head())
    """
    df = df.copy()
    
    # Default column names
    if te_cols is None:
        te_cols = ("S21_real", "S21_imag")
    if tm_cols is None:
        tm_cols = ("S12_real", "S12_imag")
    
    # Check if required columns exist
    te_real, te_imag = te_cols
    tm_real, tm_imag = tm_cols
    
    missing_cols = []
    for col in [te_real, te_imag, tm_real, tm_imag]:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        raise ValueError(
            f"Missing required columns for TE/TM computation: {missing_cols}\n"
            f"Available columns: {df.columns.tolist()}"
        )
    
    logger.info(f"Computing derived columns from TE={te_cols}, TM={tm_cols}")
    
    # Build complex S-parameters
    df["S_complex_TE"] = df[te_real] + 1j * df[te_imag]
    df["S_complex_TM"] = df[tm_real] + 1j * df[tm_imag]
    
    # Compute amplitudes
    df["amp_TE"] = np.abs(df["S_complex_TE"])
    df["amp_TM"] = np.abs(df["S_complex_TM"])
    
    # Compute phases
    phase_te = np.angle(df["S_complex_TE"])
    phase_tm = np.angle(df["S_complex_TM"])
    
    # Unwrap if requested (per group if id_nanopilar exists)
    if unwrap_phase:
        logger.info("Unwrapping phase discontinuities...")
        if "id_nanopilar" in df.columns:
            # Unwrap per nanopilar
            phase_te_unwrapped = []
            phase_tm_unwrapped = []
            for _, group in df.groupby("id_nanopilar"):
                phase_te_unwrapped.extend(np.unwrap(phase_te[group.index]))
                phase_tm_unwrapped.extend(np.unwrap(phase_tm[group.index]))
            phase_te = np.array(phase_te_unwrapped)
            phase_tm = np.array(phase_tm_unwrapped)
        else:
            phase_te = np.unwrap(phase_te)
            phase_tm = np.unwrap(phase_tm)
    
    # Convert to degrees if requested
    if phase_unit.lower() == "deg":
        logger.info("Converting phase to degrees...")
        phase_te = np.degrees(phase_te)
        phase_tm = np.degrees(phase_tm)
    
    df["phase_TE"] = phase_te
    df["phase_TM"] = phase_tm
    
    logger.info(f"Added columns: amp_TE, phase_TE, amp_TM, phase_TM, S_complex_TE, S_complex_TM")
    
    return df


def save_library(
    df: pd.DataFrame,
    out_csv: Optional[str] = None,
    out_parquet: Optional[str] = None
) -> None:
    """
    Save DataFrame to CSV and/or Parquet format.
    
    Salva DataFrame em formato CSV e/ou Parquet.
    
    Parameters / Parâmetros:
        df: DataFrame to save / DataFrame para salvar
        out_csv: Output CSV path (optional) / Caminho CSV de saída (opcional)
        out_parquet: Output Parquet path (optional) / Caminho Parquet de saída (opcional)
        
    Raises / Exceções:
        ValueError: If neither output path is specified
                   Se nenhum caminho de saída for especificado
    """
    if not out_csv and not out_parquet:
        raise ValueError("At least one output path (CSV or Parquet) must be specified")
    
    if out_csv:
        csv_path = Path(out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving CSV to {out_csv}...")
        df.to_csv(out_csv, index=False)
        logger.info(f"CSV saved successfully ({len(df)} rows)")
    
    if out_parquet:
        parquet_path = Path(out_parquet)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving Parquet to {out_parquet}...")
        df.to_parquet(out_parquet, index=False)
        logger.info(f"Parquet saved successfully ({len(df)} rows)")


if __name__ == "__main__":
    # Example usage / Exemplo de uso
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python clean_library.py <input_csv_or_parquet>")
        print("Uso: python clean_library.py <csv_ou_parquet_entrada>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        # Read input
        if input_file.endswith('.parquet'):
            df = pd.read_parquet(input_file)
        else:
            df = pd.read_csv(input_file)
        
        logger.info(f"Loaded {len(df)} rows from {input_file}")
        
        # Add derived columns
        df_clean = append_derived_columns(df, unwrap_phase=True, phase_unit="rad")
        
        # Save output
        output_csv = input_file.rsplit('.', 1)[0] + "_cleaned.csv"
        save_library(df_clean, out_csv=output_csv)
        
        print(f"\n✅ Cleaned library saved to {output_csv}")
        print(f"New columns: {[c for c in df_clean.columns if c not in df.columns]}")
        
    except Exception as e:
        logger.error(f"Failed to process file: {e}")
        sys.exit(1)
