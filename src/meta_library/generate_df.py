#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Touchstone File Parser for Metasurface Library Generation.

This module provides functions to parse Touchstone (.ts) files containing
S-parameter data for metasurface nanopillars and convert them into
structured pandas DataFrames.

Este módulo fornece funções para analisar arquivos Touchstone (.ts) contendo
dados de parâmetros S para nanopilares de metassuperfície e convertê-los em
DataFrames pandas estruturados.
"""

import re
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

try:
    import skrf as rf
except ImportError:
    rf = None
    logging.warning("scikit-rf not available. S-parameter parsing will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Regular expressions for parsing
_PARAM_ANY_RE = re.compile(r"Parameters\s*=\s*\{(?P<body>.+?)\}", re.IGNORECASE)
_NUM_PORTS_RE = re.compile(r"^\[Number of Ports\]\s*(\d+)\s*$", re.IGNORECASE)


def parse_touchstone_params(path: str, max_header_lines: int = 200) -> dict:
    """
    Parse parameter definitions from Touchstone file header.
    
    Scans the beginning of the file (up to max_header_lines or until '[Network Data]')
    searching for a line containing 'Parameters = { ... }'. Preserves original key names.
    Accepts ';' or ',' as separators between K=V pairs.
    
    Analisa definições de parâmetros do cabeçalho do arquivo Touchstone.
    
    Parameters / Parâmetros:
        path: Path to Touchstone file / Caminho para arquivo Touchstone
        max_header_lines: Maximum lines to scan / Máximo de linhas para escanear
        
    Returns / Retorna:
        Dictionary with parsed parameters / Dicionário com parâmetros analisados
        
    Examples / Exemplos:
        >>> params = parse_touchstone_params("nanopilar_001.ts")
        >>> print(params.get('L_x'), params.get('L_y'))
        400.0 500.0
    """
    params = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, raw in enumerate(f):
                line = raw.strip()
                if i > max_header_lines:
                    break
                # Stop when entering data section
                if line.startswith("[Network Data]"):
                    break
                # Try to capture block within { ... }
                m = _PARAM_ANY_RE.search(line)
                if not m:
                    continue
                body = m.group("body")

                # Try semicolon separator first, then comma
                parts = [p.strip() for p in body.split(";") if p.strip()]
                if len(parts) <= 1:
                    parts = [p.strip() for p in body.split(",") if p.strip()]

                for pair in parts:
                    if "=" not in pair:
                        continue
                    k, v = pair.split("=", 1)
                    k = k.strip()  # preserve names like 'L_x', 'L_y'
                    v = v.strip()
                    # Remove possible residual comments after value
                    v = re.split(r"\s*!|\s+#", v)[0].strip()
                    try:
                        v = float(v)
                    except Exception:
                        # If conversion to float fails, keep value as string (some parameters may be non-numeric)
                        pass
                    params[k] = v
                # Found parameter line; can exit
                break
    except Exception as e:
        logger.warning(f"Error parsing parameters from {path}: {e}")
    return params


def parse_number_of_ports_from_header(path: str) -> Optional[int]:
    """
    Extract number of ports from Touchstone file header.
    
    Searches for '[Number of Ports]' declaration in the header section.
    
    Extrai número de portas do cabeçalho do arquivo Touchstone.
    
    Parameters / Parâmetros:
        path: Path to Touchstone file / Caminho para arquivo Touchstone
        
    Returns / Retorna:
        Number of ports or None if not found / Número de portas ou None se não encontrado
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = _NUM_PORTS_RE.match(line.strip())
                if m:
                    return int(m.group(1))
                if line.startswith("[Network Data]"):
                    break
    except Exception as e:
        logger.warning(f"Error parsing port count from {path}: {e}")
    return None


def touchstone_to_dataframe(
    folder: str,
    recursive: bool = False,
    pattern: str = "*.ts"
) -> pd.DataFrame:
    """
    Convert Touchstone files in a folder to a structured DataFrame.
    
    Reads .ts files and generates a DataFrame with:
      - Metadata: filename, path, id_nanopilar, frequencia_hz/ghz, nports
      - Header parameters: original names (e.g., 'L_x', 'L_y', 'Lambda', 'H')
      - S-parameters: Sij_real/Sij_imag according to nports (1, 2, or 4)
    
    Converts arquivos Touchstone em uma pasta para um DataFrame estruturado.
    
    Parameters / Parâmetros:
        folder: Directory containing Touchstone files / Diretório com arquivos Touchstone
        recursive: Search subdirectories / Buscar subdiretórios
        pattern: File pattern to match / Padrão de arquivo para corresponder
        
    Returns / Retorna:
        DataFrame with parsed data, sorted by id_nanopilar and frequency
        DataFrame com dados analisados, ordenados por id_nanopilar e frequência
        
    Raises / Exceções:
        ValueError: If folder doesn't exist or no valid files found
                   Se pasta não existe ou nenhum arquivo válido encontrado
    """
    if not rf:
        raise ImportError(
            "scikit-rf is required for Touchstone parsing. "
            "Install with: pip install scikit-rf"
        )
    
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"Folder not found: {folder}")

    logger.info(f"Searching for Touchstone files in {folder} (recursive={recursive})...")
    
    data = []
    
    # Collect files
    if recursive:
        files = list(folder_path.rglob(pattern))
    else:
        files = list(folder_path.glob(pattern))
    
    if not files:
        raise ValueError(f"No files matching pattern '{pattern}' found in {folder}")
    
    logger.info(f"Found {len(files)} Touchstone files")
    
    for file_path in files:
        name_arch = file_path.name
        path_ = str(file_path)
        root = file_path.stem
        
        # Extract ID from filename (first number found)
        m = re.search(r"(\d+)", root)
        id_nanopilar = int(m.group(1)) if m else -1

        logger.info(f"Reading file: {name_arch} (ID:{id_nanopilar})...")

        # 1) Parse header parameters
        params = parse_touchstone_params(path_)

        # 2) Read network with scikit-rf
        try:
            network = rf.Network(path_)
            nports = int(network.nports)
        except Exception as e:
            logger.warning(f"  [WARN] scikit-rf failed on '{name_arch}': {e}")
            nports = parse_number_of_ports_from_header(path_) or 0
            if nports == 0:
                logger.error(f"  [ERROR] Could not infer nports. Skipping file.")
                continue
            logger.error(f"  [ERROR] No S-parameter data available. Skipping.")
            continue

        # 3) Create row for each frequency
        for i, f_hz in enumerate(network.f):
            row = {
                "arquivo": name_arch,
                "caminho": path_,
                "id_nanopilar": id_nanopilar,
                "frequencia_hz": float(f_hz),
                "frequencia_ghz": float(f_hz / 1e9),
                "nports": nports,
            }

            # Inject ALL header parameters
            for k, v in params.items():
                row[k] = v

            # ENSURE L_x, L_y, H columns exist (even if NaN)
            for key in ("L_x", "L_y", "H"):
                if key not in row:
                    row[key] = np.nan

            # S-parameters based on port count
            if nports == 1:
                s11 = network.s[i, 0, 0]
                row["S11_real"] = float(np.real(s11))
                row["S11_imag"] = float(np.imag(s11))

            elif nports == 2:
                s11 = network.s[i, 0, 0]
                s21 = network.s[i, 1, 0]
                s12 = network.s[i, 0, 1]
                s22 = network.s[i, 1, 1]
                row["S11_real"] = float(np.real(s11))
                row["S11_imag"] = float(np.imag(s11))
                row["S21_real"] = float(np.real(s21))
                row["S21_imag"] = float(np.imag(s21))
                row["S12_real"] = float(np.real(s12))
                row["S12_imag"] = float(np.imag(s12))
                row["S22_real"] = float(np.real(s22))
                row["S22_imag"] = float(np.imag(s22))

            elif nports == 4:
                for op in range(4):
                    for ip in range(4):
                        s = network.s[i, op, ip]
                        pname = f"S{op+1}{ip+1}"
                        row[f"{pname}_real"] = float(np.real(s))
                        row[f"{pname}_imag"] = float(np.imag(s))

            data.append(row)

    if not data:
        raise ValueError("No valid Touchstone files could be parsed")

    df = pd.DataFrame(data)

    # Sort by id_nanopilar and frequency
    sort_cols = [c for c in ["id_nanopilar", "frequencia_hz"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ignore_index=True)

    logger.info(f"Successfully created DataFrame with {len(df)} rows from {len(files)} files")
    
    return df


if __name__ == "__main__":
    # Example usage / Exemplo de uso
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_df.py <folder_path>")
        print("Uso: python generate_df.py <caminho_pasta>")
        sys.exit(1)
    
    folder = sys.argv[1]
    try:
        df = touchstone_to_dataframe(folder)
        print("\n--- DataFrame created successfully! ---")
        print(f"Shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nFirst rows:\n{df.head()}")
    except Exception as e:
        logger.error(f"Failed to process folder: {e}")
        sys.exit(1)
