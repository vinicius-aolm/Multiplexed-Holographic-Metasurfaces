#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analisador de Arquivos Touchstone para Geração de Biblioteca de Metassuperfície.

Este módulo fornece funções para analisar arquivos Touchstone (.ts) contendo
dados de parâmetros S de nanopilares de metassuperfície e convertê-los em
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
    logging.warning("scikit-rf não disponível. Análise de parâmetros S será limitada.")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Expressões regulares para análise
_PARAM_ANY_RE = re.compile(r"Parameters\s*=\s*\{(?P<body>.+?)\}", re.IGNORECASE)
_NUM_PORTS_RE = re.compile(r"^\[Number of Ports\]\s*(\d+)\s*$", re.IGNORECASE)


def parse_touchstone_params(path: str, max_header_lines: int = 200) -> dict:
    """
    Analisa definições de parâmetros do cabeçalho do arquivo Touchstone.
    
    Varre o início do arquivo (até max_header_lines ou até '[Network Data]')
    procurando por uma linha que contenha 'Parameters = { ... }'. Preserva nomes
    originais das chaves. Aceita ';' ou ',' como separadores entre pares K=V.
    
    Args:
        path: Caminho para arquivo Touchstone
        max_header_lines: Número máximo de linhas para escanear
        
    Returns:
        Dicionário com parâmetros analisados
        
    Examples:
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
                # Parar ao entrar na seção de dados
                if line.startswith("[Network Data]"):
                    break
                # Tentar capturar bloco dentro de { ... }
                m = _PARAM_ANY_RE.search(line)
                if not m:
                    continue
                body = m.group("body")

                # Separadores: primeiro tenta ';', depois ','
                parts = [p.strip() for p in body.split(";") if p.strip()]
                if len(parts) <= 1:
                    parts = [p.strip() for p in body.split(",") if p.strip()]

                for pair in parts:
                    if "=" not in pair:
                        continue
                    k, v = pair.split("=", 1)
                    k = k.strip()  # preserva nomes como 'L_x', 'L_y'
                    v = v.strip()
                    # Remove possíveis comentários residuais após o valor
                    v = re.split(r"\s*!|\s+#", v)[0].strip()
                    try:
                        v = float(v)
                    except Exception:
                        # Se conversão para float falhar, manter como string (alguns parâmetros podem ser não-numéricos)
                        pass
                    params[k] = v
                # Encontrou linha de parâmetros; pode sair
                break
    except Exception as e:
        logger.warning(f"Erro ao analisar parâmetros de {path}: {e}")
    return params


def parse_number_of_ports_from_header(path: str) -> Optional[int]:
    """
    Extrai número de portas do cabeçalho do arquivo Touchstone.
    
    Procura pela declaração '[Number of Ports]' na seção do cabeçalho.
    
    Args:
        path: Caminho para arquivo Touchstone
        
    Returns:
        Número de portas ou None se não encontrado
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
        logger.warning(f"Erro ao analisar contagem de portas de {path}: {e}")
    return None


def touchstone_to_dataframe(
    folder: str,
    recursive: bool = False,
    pattern: str = "*.ts"
) -> pd.DataFrame:
    """
    Converte arquivos Touchstone de uma pasta em um DataFrame estruturado.
    
    Lê arquivos .ts e gera um DataFrame com:
      - Metadados: nome_arquivo, caminho, id_nanopilar, frequencia_hz/ghz, nports
      - Parâmetros do cabeçalho: nomes originais (ex.: 'L_x', 'L_y', 'Lambda', 'H')
      - Parâmetros S: Sij_real/Sij_imag conforme nports (1, 2 ou 4)
    
    Args:
        folder: Diretório contendo arquivos Touchstone
        recursive: Buscar em subdiretórios
        pattern: Padrão de arquivo para corresponder
        
    Returns:
        DataFrame com dados analisados, ordenados por id_nanopilar e frequência
        
    Raises:
        ValueError: Se pasta não existe ou nenhum arquivo válido encontrado
    """
    if not rf:
        raise ImportError(
            "scikit-rf é necessário para análise Touchstone. "
            "Instale com: pip install scikit-rf"
        )
    
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"Pasta não encontrada: {folder}")

    logger.info(f"Procurando arquivos Touchstone em {folder} (recursive={recursive})...")
    
    data = []
    
    # Coletar arquivos
    if recursive:
        files = list(folder_path.rglob(pattern))
    else:
        files = list(folder_path.glob(pattern))
    
    if not files:
        raise ValueError(f"Nenhum arquivo correspondendo ao padrão '{pattern}' encontrado em {folder}")
    
    logger.info(f"Encontrados {len(files)} arquivos Touchstone")
    
    for file_path in files:
        name_arch = file_path.name
        path_ = str(file_path)
        root = file_path.stem
        
        # Extrair ID do nome do arquivo (primeiro número encontrado)
        m = re.search(r"(\d+)", root)
        id_nanopilar = int(m.group(1)) if m else -1

        logger.info(f"Lendo arquivo: {name_arch} (ID:{id_nanopilar})...")

        # 1) Analisar parâmetros do cabeçalho
        params = parse_touchstone_params(path_)

        # 2) Ler rede com scikit-rf
        try:
            network = rf.Network(path_)
            nports = int(network.nports)
        except Exception as e:
            logger.warning(f"  [AVISO] scikit-rf falhou em '{name_arch}': {e}")
            nports = parse_number_of_ports_from_header(path_) or 0
            if nports == 0:
                logger.error(f"  [ERRO] Não foi possível inferir nports. Pulando arquivo.")
                continue
            logger.error(f"  [ERRO] Sem dados de parâmetros S disponíveis. Pulando.")
            continue

        # 3) Criar linha para cada frequência
        for i, f_hz in enumerate(network.f):
            row = {
                "arquivo": name_arch,
                "caminho": path_,
                "id_nanopilar": id_nanopilar,
                "frequencia_hz": float(f_hz),
                "frequencia_ghz": float(f_hz / 1e9),
                "nports": nports,
            }

            # Injetar TODOS os parâmetros do cabeçalho
            for k, v in params.items():
                row[k] = v

            # GARANTIR que colunas L_x, L_y, H existam (mesmo que NaN)
            for key in ("L_x", "L_y", "H"):
                if key not in row:
                    row[key] = np.nan

            # Parâmetros S baseados na contagem de portas
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
        raise ValueError("Nenhum arquivo Touchstone válido pôde ser analisado")

    df = pd.DataFrame(data)

    # Ordenar por id_nanopilar e frequência
    sort_cols = [c for c in ["id_nanopilar", "frequencia_hz"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ignore_index=True)

    logger.info(f"DataFrame criado com sucesso: {len(df)} linhas de {len(files)} arquivos")
    
    return df


if __name__ == "__main__":
    # Exemplo de uso
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python generate_df.py <caminho_pasta>")
        sys.exit(1)
    
    folder = sys.argv[1]
    try:
        df = touchstone_to_dataframe(folder)
        print("\n--- DataFrame criado com sucesso! ---")
        print(f"Forma: {df.shape}")
        print(f"\nColunas: {df.columns.tolist()}")
        print(f"\nPrimeiras linhas:\n{df.head()}")
    except Exception as e:
        logger.error(f"Falha ao processar pasta: {e}")
        sys.exit(1)
