#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Limpeza de Biblioteca e Geração de Colunas Derivadas.

Este módulo fornece funções para limpar DataFrames de biblioteca de
metassuperfície e calcular colunas derivadas como amplitudes e fases
de transmissão para polarizações TE e TM.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

# Configurar logging
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
    Adiciona colunas derivadas de amplitude e fase de transmissão para polarizações TE/TM.
    
    Calcula parâmetros S complexos a partir de componentes reais/imaginários e extrai
    amplitude (|T|) e fase (∠T) para modos TE e TM.
    
    Args:
        df: DataFrame de entrada com colunas de parâmetros S
        te_cols: Tupla de (col_real, col_imag) para transmissão TE
                 Padrão: ("S21_real", "S21_imag")
        tm_cols: Tupla de (col_real, col_imag) para transmissão TM
                 Padrão: ("S12_real", "S12_imag")
        unwrap_phase: Se deve desembrulhar descontinuidades de fase
        phase_unit: "rad" para radianos ou "deg" para graus
                    
    Returns:
        DataFrame com colunas adicionadas: amp_TE, phase_TE, amp_TM, phase_TM,
        S_complex_TE, S_complex_TM
        
    Examples:
        >>> df_clean = append_derived_columns(df, unwrap_phase=True, phase_unit="deg")
        >>> print(df_clean[['amp_TE', 'phase_TE', 'amp_TM', 'phase_TM']].head())
    """
    df = df.copy()
    
    # Nomes de colunas padrão
    if te_cols is None:
        te_cols = ("S21_real", "S21_imag")
    if tm_cols is None:
        tm_cols = ("S12_real", "S12_imag")
    
    # Verificar se colunas necessárias existem
    te_real, te_imag = te_cols
    tm_real, tm_imag = tm_cols
    
    missing_cols = []
    for col in [te_real, te_imag, tm_real, tm_imag]:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        raise ValueError(
            f"Colunas necessárias faltando para cálculo TE/TM: {missing_cols}\n"
            f"Colunas disponíveis: {df.columns.tolist()}"
        )
    
    logger.info(f"Calculando colunas derivadas de TE={te_cols}, TM={tm_cols}")
    
    # Construir parâmetros S complexos
    df["S_complex_TE"] = df[te_real] + 1j * df[te_imag]
    df["S_complex_TM"] = df[tm_real] + 1j * df[tm_imag]
    
    # Calcular amplitudes
    df["amp_TE"] = np.abs(df["S_complex_TE"])
    df["amp_TM"] = np.abs(df["S_complex_TM"])
    
    # Calcular fases
    phase_te = np.angle(df["S_complex_TE"])
    phase_tm = np.angle(df["S_complex_TM"])
    
    # Desembrulhar se solicitado (por grupo se id_nanopilar existir)
    if unwrap_phase:
        logger.info("Desembrulhando descontinuidades de fase...")
        if "id_nanopilar" in df.columns:
            # Desembrulhar por nanopilar
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
    
    # Converter para graus se solicitado
    if phase_unit.lower() == "deg":
        logger.info("Convertendo fase para graus...")
        phase_te = np.degrees(phase_te)
        phase_tm = np.degrees(phase_tm)
    
    df["phase_TE"] = phase_te
    df["phase_TM"] = phase_tm
    
    logger.info(f"Colunas adicionadas: amp_TE, phase_TE, amp_TM, phase_TM, S_complex_TE, S_complex_TM")
    
    return df


def save_library(
    df: pd.DataFrame,
    out_csv: Optional[str] = None,
    out_parquet: Optional[str] = None
) -> None:
    """
    Salva DataFrame em formato CSV e/ou Parquet.
    
    Args:
        df: DataFrame para salvar
        out_csv: Caminho CSV de saída (opcional)
        out_parquet: Caminho Parquet de saída (opcional)
        
    Raises:
        ValueError: Se nenhum caminho de saída for especificado
    """
    if not out_csv and not out_parquet:
        raise ValueError("Pelo menos um caminho de saída (CSV ou Parquet) deve ser especificado")
    
    if out_csv:
        csv_path = Path(out_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Salvando CSV em {out_csv}...")
        df.to_csv(out_csv, index=False)
        logger.info(f"CSV salvo com sucesso ({len(df)} linhas)")
    
    if out_parquet:
        parquet_path = Path(out_parquet)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Salvando Parquet em {out_parquet}...")
        df.to_parquet(out_parquet, index=False)
        logger.info(f"Parquet salvo com sucesso ({len(df)} linhas)")


if __name__ == "__main__":
    # Exemplo de uso
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python clean_library.py <csv_ou_parquet_entrada>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        # Ler entrada
        if input_file.endswith('.parquet'):
            df = pd.read_parquet(input_file)
        else:
            df = pd.read_csv(input_file)
        
        logger.info(f"Carregadas {len(df)} linhas de {input_file}")
        
        # Adicionar colunas derivadas
        df_clean = append_derived_columns(df, unwrap_phase=True, phase_unit="rad")
        
        # Salvar saída
        output_csv = input_file.rsplit('.', 1)[0] + "_cleaned.csv"
        save_library(df_clean, out_csv=output_csv)
        
        print(f"\n✅ Biblioteca limpa salva em {output_csv}")
        print(f"Novas colunas: {[c for c in df_clean.columns if c not in df.columns]}")
        
    except Exception as e:
        logger.error(f"Falha ao processar arquivo: {e}")
        sys.exit(1)
