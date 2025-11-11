#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para processar bibliotecas brutas de Touchstone para formato usado nos notebooks.

Este script lê arquivos CSV de bibliotecas brutas (com parâmetros S) e gera
versões processadas com colunas phase_TE, amp_TE, phase_TM, amp_TM.

Uso:
    python scripts/process_raw_library.py <caminho_entrada.csv> [caminho_saida.csv]

Exemplo:
    python scripts/process_raw_library.py Bibliotecas/Altura_Varia/biblioteca_Bib1-27x27-perdas.csv data/meta_library/library_cleaned.csv
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd


def process_library(
    input_path: str,
    output_path: str = None,
    freq_filter: float = None,
    te_params: tuple = ('S13_real', 'S13_imag'),
    tm_params: tuple = ('S24_real', 'S24_imag')
) -> pd.DataFrame:
    """
    Processa biblioteca bruta para formato com phase_TE, amp_TE, phase_TM, amp_TM.
    
    Args:
        input_path: Caminho para CSV de entrada
        output_path: Caminho para CSV de saída (opcional)
        freq_filter: Filtrar por frequência específica (GHz). Se None, usa a primeira.
        te_params: Tupla (col_real, col_imag) para TE
        tm_params: Tupla (col_real, col_imag) para TM
        
    Returns:
        DataFrame processado
    """
    print(f"📂 Carregando biblioteca de: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   Registros originais: {len(df)}")
    print(f"   Colunas: {len(df.columns)}")
    
    # Filtrar por frequência se houver múltiplas
    if 'frequencia_ghz' in df.columns:
        unique_freqs = df['frequencia_ghz'].unique()
        if len(unique_freqs) > 1:
            if freq_filter is None:
                freq_filter = unique_freqs[0]
                print(f"   ⚠️  Múltiplas frequências encontradas. Usando: {freq_filter} GHz")
            df = df[df['frequencia_ghz'] == freq_filter].copy()
            print(f"   Filtrado para: {len(df)} registros")
    
    # Verificar se colunas necessárias existem
    required_cols = ['L_x', 'L_y'] + list(te_params) + list(tm_params)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas necessárias faltando: {missing}")
    
    print(f"\n🔬 Calculando propriedades ópticas...")
    
    # Calcular TE (normalmente S13)
    te_real, te_imag = te_params
    df['phase_TE'] = np.arctan2(df[te_imag], df[te_real])
    df['amp_TE'] = np.sqrt(df[te_real]**2 + df[te_imag]**2)
    print(f"   ✓ TE: phase [{df['phase_TE'].min():.3f}, {df['phase_TE'].max():.3f}] rad")
    print(f"        amp [{df['amp_TE'].min():.3f}, {df['amp_TE'].max():.3f}]")
    
    # Calcular TM (normalmente S24)
    tm_real, tm_imag = tm_params
    df['phase_TM'] = np.arctan2(df[tm_imag], df[tm_real])
    df['amp_TM'] = np.sqrt(df[tm_real]**2 + df[tm_imag]**2)
    print(f"   ✓ TM: phase [{df['phase_TM'].min():.3f}, {df['phase_TM'].max():.3f}] rad")
    print(f"        amp [{df['amp_TM'].min():.3f}, {df['amp_TM'].max():.3f}]")
    
    # Selecionar colunas relevantes
    output_cols = ['L_x', 'L_y', 'phase_TE', 'amp_TE', 'phase_TM', 'amp_TM']
    if 'H' in df.columns:
        output_cols.insert(2, 'H')
    
    df_clean = df[output_cols].copy()
    
    print(f"\n📊 Estatísticas da biblioteca processada:")
    print(f"   L_x: [{df_clean['L_x'].min():.1f}, {df_clean['L_x'].max():.1f}]")
    print(f"   L_y: [{df_clean['L_y'].min():.1f}, {df_clean['L_y'].max():.1f}]")
    print(f"   Combinações únicas (L_x, L_y): {len(df_clean[['L_x', 'L_y']].drop_duplicates())}")
    
    # Salvar se caminho de saída fornecido
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        print(f"\n✅ Biblioteca processada salva em: {output_path}")
        print(f"   Forma final: {df_clean.shape}")
    
    return df_clean


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n❌ Erro: Caminho de entrada não fornecido")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if not os.path.exists(input_path):
        print(f"❌ Erro: Arquivo não encontrado: {input_path}")
        sys.exit(1)
    
    # Determinar caminho de saída
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        # Usar caminho padrão
        output_path = "data/meta_library/library_cleaned.csv"
    
    try:
        process_library(input_path, output_path)
        print("\n🎉 Processamento concluído com sucesso!")
        print(f"\n💡 Dica: Use este arquivo no notebook 01_Library_Heatmaps_Explanation.ipynb")
        print(f"   Atualize library_path = \"{output_path}\"")
    except Exception as e:
        print(f"\n❌ Erro durante processamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
