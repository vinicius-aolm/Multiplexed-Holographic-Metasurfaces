#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI tool for building metasurface library from Touchstone files.

Command-line interface to parse Touchstone files and generate structured
library DataFrames with S-parameter data.

Interface de linha de comando para analisar arquivos Touchstone e gerar
DataFrames estruturados de biblioteca com dados de parâmetros S.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_library import generate_df

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
        description="Build metasurface library from Touchstone files / "
                   "Construir biblioteca de metassuperfície de arquivos Touchstone",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    parser.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="Input directory containing Touchstone files / "
             "Diretório de entrada contendo arquivos Touchstone"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively / "
             "Buscar subdiretórios recursivamente"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.ts",
        help="File pattern to match / Padrão de arquivo para corresponder"
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
    
    # Metadata options
    parser.add_argument(
        "--experiment",
        type=str,
        default="library_build",
        help="Experiment name for organization / "
             "Nome do experimento para organização"
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Root output directory (default: results/meta_library/library_build) / "
             "Diretório raiz de saída (padrão: results/meta_library/library_build)"
    )
    
    return parser.parse_args()


def create_readme(run_dir: Path, metadata: dict) -> Path:
    """
    Create README.md with run information.
    
    Cria README.md com informações da execução.
    """
    readme_path = run_dir / "README.md"
    
    content = f"""# Library Build Run

## Summary / Resumo

Library generated from Touchstone files.

Biblioteca gerada a partir de arquivos Touchstone.

## Run Information / Informações da Execução

- **Run ID**: {metadata['run_id']}
- **Experiment**: {metadata['experiment']}
- **Timestamp**: {metadata['timestamp']}
- **Input Directory**: {metadata['input_dir']}
- **Recursive**: {metadata['recursive']}
- **Pattern**: {metadata['pattern']}

## Results / Resultados

- **Files Processed**: {metadata['files_processed']}
- **Total Rows**: {metadata['total_rows']}
- **Columns**: {metadata['n_columns']}

### Output Files / Arquivos de Saída

"""
    
    if metadata.get('output_csv'):
        content += f"- CSV: `{metadata['output_csv']}`\n"
    if metadata.get('output_parquet'):
        content += f"- Parquet: `{metadata['output_parquet']}`\n"
    
    content += f"\n## Column Summary / Resumo de Colunas\n\n"
    content += f"```\n{metadata.get('columns_info', 'N/A')}\n```\n"
    
    content += f"\n## Reproducibility / Reprodutibilidade\n\n"
    content += f"```bash\n"
    content += f"python src/cli/run_library_build.py \\\n"
    content += f"  --in-dir \"{metadata['input_dir']}\" \\\n"
    content += f"  --recursive {metadata['recursive']} \\\n"
    content += f"  --pattern \"{metadata['pattern']}\" \\\n"
    if metadata.get('output_csv'):
        content += f"  --out-csv \"{metadata['output_csv']}\" \\\n"
    if metadata.get('output_parquet'):
        content += f"  --out-parquet \"{metadata['output_parquet']}\" \\\n"
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
        out_root = repo_root / "results" / "meta_library" / "library_build"
    else:
        out_root = args.out_root
    
    # Create run directory
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = out_root / args.experiment / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting library build: {args.experiment}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Input directory: {args.in_dir}")
    logger.info(f"Output directory: {run_dir}")
    
    try:
        # Parse Touchstone files
        df = generate_df.touchstone_to_dataframe(
            folder=str(args.in_dir),
            recursive=args.recursive,
            pattern=args.pattern
        )
        
        logger.info(f"Successfully parsed {len(df)} rows from Touchstone files")
        
        # Determine output paths
        if args.out_csv:
            out_csv = args.out_csv
        else:
            out_csv = run_dir / f"library_{run_id}.csv"
        
        if args.out_parquet:
            out_parquet = args.out_parquet
        else:
            out_parquet = run_dir / f"library_{run_id}.parquet"
        
        # Save outputs
        logger.info(f"Saving CSV to {out_csv}...")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        
        logger.info(f"Saving Parquet to {out_parquet}...")
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_parquet, index=False)
        
        # Create metadata
        metadata = {
            "run_id": run_id,
            "experiment": args.experiment,
            "timestamp": datetime.now().isoformat(),
            "input_dir": str(args.in_dir),
            "recursive": args.recursive,
            "pattern": args.pattern,
            "files_processed": len(df["arquivo"].unique()) if "arquivo" in df.columns else "N/A",
            "total_rows": len(df),
            "n_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "columns_info": "\n".join([f"{col}: {df[col].dtype}" for col in df.columns[:20]]),
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
        print("✅ Library Build Complete / Construção de Biblioteca Completa")
        print("="*60)
        print(f"\nRun ID: {run_id}")
        print(f"Files processed: {metadata['files_processed']}")
        print(f"Total rows: {metadata['total_rows']}")
        print(f"Columns: {metadata['n_columns']}")
        print(f"\nOutputs:")
        print(f"  - CSV: {out_csv}")
        print(f"  - Parquet: {out_parquet}")
        print(f"  - Metadata: {meta_path}")
        print(f"  - README: {readme_path}")
        print(f"\nRun directory: {run_dir}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to build library: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
