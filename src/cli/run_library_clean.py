#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI tool for cleaning metasurface library data.

Command-line interface to add derived columns (amplitude, phase) to
library DataFrames and perform data cleaning operations.

Interface de linha de comando para adicionar colunas derivadas (amplitude, fase)
a DataFrames de biblioteca e realizar operações de limpeza de dados.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_library import clean_library

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_repo_root(start: Path = Path.cwd()) -> Path:
    """Find repository root by locating .git directory."""
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    return start


def parse_args():
    """
    Configure command-line arguments.
    
    Configura argumentos da linha de comando.
    """
    parser = argparse.ArgumentParser(
        description="Clean metasurface library and add derived columns / "
                   "Limpar biblioteca de metassuperfície e adicionar colunas derivadas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    parser.add_argument(
        "--in",
        dest="input_file",
        type=Path,
        required=True,
        help="Input CSV or Parquet file / Arquivo CSV ou Parquet de entrada"
    )
    
    # Output options
    parser.add_argument(
        "--out-csv",
        type=Path,
        help="Output CSV file path / Caminho do arquivo CSV de saída"
    )
    parser.add_argument(
        "--out-parquet",
        type=Path,
        help="Output Parquet file path / Caminho do arquivo Parquet de saída"
    )
    
    # Processing options
    parser.add_argument(
        "--unwrap-phase",
        action="store_true",
        help="Unwrap phase discontinuities / Desembrulhar descontinuidades de fase"
    )
    parser.add_argument(
        "--phase-unit",
        type=str,
        choices=["rad", "deg"],
        default="rad",
        help="Phase unit: radians or degrees / Unidade de fase: radianos ou graus"
    )
    
    # Column mapping options
    parser.add_argument(
        "--te-real",
        type=str,
        default="S21_real",
        help="TE transmission real part column / Coluna da parte real da transmissão TE"
    )
    parser.add_argument(
        "--te-imag",
        type=str,
        default="S21_imag",
        help="TE transmission imaginary part column / Coluna da parte imaginária da transmissão TE"
    )
    parser.add_argument(
        "--tm-real",
        type=str,
        default="S12_real",
        help="TM transmission real part column / Coluna da parte real da transmissão TM"
    )
    parser.add_argument(
        "--tm-imag",
        type=str,
        default="S12_imag",
        help="TM transmission imaginary part column / Coluna da parte imaginária da transmissão TM"
    )
    
    # Metadata options
    parser.add_argument(
        "--experiment",
        type=str,
        default="library_clean",
        help="Experiment name for organization / Nome do experimento para organização"
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Root output directory (default: results/meta_library/library_clean) / "
             "Diretório raiz de saída (padrão: results/meta_library/library_clean)"
    )
    
    return parser.parse_args()


def create_readme(run_dir: Path, metadata: dict) -> Path:
    """
    Create README.md with run information.
    
    Cria README.md com informações da execução.
    """
    readme_path = run_dir / "README.md"
    
    content = f"""# Library Clean Run

## Summary / Resumo

Library cleaned and enriched with derived columns for amplitude and phase analysis.

Biblioteca limpa e enriquecida com colunas derivadas para análise de amplitude e fase.

## Run Information / Informações da Execução

- **Run ID**: {metadata['run_id']}
- **Experiment**: {metadata['experiment']}
- **Timestamp**: {metadata['timestamp']}
- **Input File**: {metadata['input_file']}

## Processing Options / Opções de Processamento

- **Unwrap Phase**: {metadata['unwrap_phase']}
- **Phase Unit**: {metadata['phase_unit']}
- **TE Columns**: {metadata['te_cols']}
- **TM Columns**: {metadata['tm_cols']}

## Results / Resultados

- **Input Rows**: {metadata['input_rows']}
- **Output Rows**: {metadata['output_rows']}
- **Added Columns**: {', '.join(metadata['added_columns'])}

### Output Files / Arquivos de Saída

"""
    
    if metadata.get('output_csv'):
        content += f"- CSV: `{metadata['output_csv']}`\n"
    if metadata.get('output_parquet'):
        content += f"- Parquet: `{metadata['output_parquet']}`\n"
    
    content += f"\n## Reproducibility / Reprodutibilidade\n\n"
    content += f"```bash\n"
    content += f"python src/cli/run_library_clean.py \\\n"
    content += f"  --in \"{metadata['input_file']}\" \\\n"
    if metadata.get('output_csv'):
        content += f"  --out-csv \"{metadata['output_csv']}\" \\\n"
    if metadata.get('output_parquet'):
        content += f"  --out-parquet \"{metadata['output_parquet']}\" \\\n"
    if metadata['unwrap_phase']:
        content += f"  --unwrap-phase \\\n"
    content += f"  --phase-unit {metadata['phase_unit']} \\\n"
    content += f"  --te-real {metadata['te_cols'][0]} --te-imag {metadata['te_cols'][1]} \\\n"
    content += f"  --tm-real {metadata['tm_cols'][0]} --tm-imag {metadata['tm_cols'][1]} \\\n"
    content += f"  --experiment \"{metadata['experiment']}\"\n"
    content += f"```\n"
    
    readme_path.write_text(content, encoding='utf-8')
    return readme_path


def main():
    """Main execution function / Função principal de execução."""
    args = parse_args()
    
    # Setup output directory
    if args.out_root is None:
        repo_root = find_repo_root()
        out_root = repo_root / "results" / "meta_library" / "library_clean"
    else:
        out_root = args.out_root
    
    # Create run directory
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = out_root / args.experiment / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting library cleaning: {args.experiment}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output directory: {run_dir}")
    
    try:
        # Read input
        logger.info(f"Reading input file...")
        if str(args.input_file).endswith('.parquet'):
            df = pd.read_parquet(args.input_file)
        else:
            df = pd.read_csv(args.input_file)
        
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Store original columns
        original_cols = set(df.columns)
        
        # Add derived columns
        te_cols = (args.te_real, args.te_imag)
        tm_cols = (args.tm_real, args.tm_imag)
        
        logger.info(f"Adding derived columns with TE={te_cols}, TM={tm_cols}")
        df_clean = clean_library.append_derived_columns(
            df,
            te_cols=te_cols,
            tm_cols=tm_cols,
            unwrap_phase=args.unwrap_phase,
            phase_unit=args.phase_unit
        )
        
        # Identify added columns
        added_cols = [col for col in df_clean.columns if col not in original_cols]
        logger.info(f"Added {len(added_cols)} columns: {added_cols}")
        
        # Determine output paths
        if args.out_csv:
            out_csv = args.out_csv
        else:
            out_csv = run_dir / f"library_cleaned_{run_id}.csv"
        
        if args.out_parquet:
            out_parquet = args.out_parquet
        else:
            out_parquet = run_dir / f"library_cleaned_{run_id}.parquet"
        
        # Save outputs
        clean_library.save_library(df_clean, out_csv=out_csv, out_parquet=out_parquet)
        
        # Create metadata
        metadata = {
            "run_id": run_id,
            "experiment": args.experiment,
            "timestamp": datetime.now().isoformat(),
            "input_file": str(args.input_file),
            "input_rows": len(df),
            "output_rows": len(df_clean),
            "unwrap_phase": args.unwrap_phase,
            "phase_unit": args.phase_unit,
            "te_cols": list(te_cols),
            "tm_cols": list(tm_cols),
            "added_columns": added_cols,
            "output_csv": str(out_csv),
            "output_parquet": str(out_parquet),
        }
        
        # Save metadata as JSON
        meta_path = run_dir / "run_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadata saved to {meta_path}")
        
        # Create README
        readme_path = create_readme(run_dir, metadata)
        logger.info(f"README saved to {readme_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("✅ Library Cleaning Complete / Limpeza de Biblioteca Completa")
        print("="*60)
        print(f"\nRun ID: {run_id}")
        print(f"Input rows: {metadata['input_rows']}")
        print(f"Output rows: {metadata['output_rows']}")
        print(f"Added columns: {', '.join(added_cols)}")
        print(f"\nOutputs:")
        print(f"  - CSV: {out_csv}")
        print(f"  - Parquet: {out_parquet}")
        print(f"  - Metadata: {meta_path}")
        print(f"  - README: {readme_path}")
        print(f"\nRun directory: {run_dir}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to clean library: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
