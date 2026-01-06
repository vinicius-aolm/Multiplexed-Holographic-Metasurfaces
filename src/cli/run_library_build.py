#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ferramenta CLI para construir biblioteca de metassuperfície a partir de arquivos Touchstone.

Interface de linha de comando para analisar arquivos Touchstone e gerar
DataFrames estruturados de biblioteca com dados de parâmetros S.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Adicionar diretório pai ao path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_library import generate_df

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_repo_root(start: Path = Path.cwd()) -> Path:
    """Encontra a raiz do repositório localizando o diretório .git."""
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    return start


def parse_args():
    """Configura os argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Construir biblioteca de metassuperfície de arquivos Touchstone",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Opções de entrada
    parser.add_argument(
        "--in-dir",
        type=Path,
        required=True,
        help="Diretório de entrada contendo arquivos Touchstone"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Buscar subdiretórios recursivamente"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.ts",
        help="Padrão de arquivo para corresponder"
    )
    
    # Opções de saída
    parser.add_argument(
        "--out-csv",
        type=Path,
        help="Caminho do arquivo CSV de saída"
    )
    parser.add_argument(
        "--out-parquet",
        type=Path,
        help="Caminho do arquivo Parquet de saída"
    )
    
    # Opções de metadados
    parser.add_argument(
        "--experiment",
        type=str,
        default="library_build",
        help="Nome do experimento para organização"
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Diretório raiz de saída (padrão: results/meta_library/library_build)"
    )
    
    return parser.parse_args()


def create_readme(run_dir: Path, metadata: dict) -> Path:
    """Cria README.md com informações da execução."""
    readme_path = run_dir / "README.md"
    
    content = f"""# Execução de Construção de Biblioteca

## Resumo

Biblioteca gerada a partir de arquivos Touchstone.

## Informações da Execução

- **Run ID**: {metadata['run_id']}
- **Experimento**: {metadata['experiment']}
- **Timestamp**: {metadata['timestamp']}
- **Diretório de Entrada**: {metadata['input_dir']}
- **Recursivo**: {metadata['recursive']}
- **Padrão**: {metadata['pattern']}

## Resultados

- **Arquivos Processados**: {metadata['files_processed']}
- **Total de Linhas**: {metadata['total_rows']}
- **Colunas**: {metadata['n_columns']}

### Arquivos de Saída

"""
    
    if metadata.get('output_csv'):
        content += f"- CSV: `{metadata['output_csv']}`\n"
    if metadata.get('output_parquet'):
        content += f"- Parquet: `{metadata['output_parquet']}`\n"
    
    content += f"\n## Resumo de Colunas\n\n"
    content += f"```\n{metadata.get('columns_info', 'N/A')}\n```\n"
    
    content += f"\n## Reprodutibilidade\n\n"
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
    """Função principal de execução."""
    args = parse_args()
    
    # Configurar diretório de saída
    if args.out_root is None:
        repo_root = find_repo_root()
        out_root = repo_root / "results" / "meta_library" / "library_build"
    else:
        out_root = args.out_root
    
    # Criar diretório de execução
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = out_root / args.experiment / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Iniciando construção de biblioteca: {args.experiment}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Diretório de entrada: {args.in_dir}")
    logger.info(f"Diretório de saída: {run_dir}")
    
    try:
        # Analisar arquivos Touchstone
        df = generate_df.touchstone_to_dataframe(
            folder=str(args.in_dir),
            recursive=args.recursive,
            pattern=args.pattern
        )
        
        logger.info(f"Analisadas com sucesso {len(df)} linhas de arquivos Touchstone")
        
        # Determinar caminhos de saída
        if args.out_csv:
            out_csv = args.out_csv
        else:
            out_csv = run_dir / f"library_{run_id}.csv"
        
        if args.out_parquet:
            out_parquet = args.out_parquet
        else:
            out_parquet = run_dir / f"library_{run_id}.parquet"
        
        # Salvar saídas
        logger.info(f"Salvando CSV em {out_csv}...")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        
        logger.info(f"Salvando Parquet em {out_parquet}...")
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_parquet, index=False)
        
        # Criar metadados
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
        
        # Salvar metadados como JSON
        meta_path = run_dir / "run_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadados salvos em {meta_path}")
        
        # Criar README
        readme_path = create_readme(run_dir, metadata)
        logger.info(f"README salvo em {readme_path}")
        
        # Imprimir resumo
        print("\n" + "="*60)
        print("✅ Construção de Biblioteca Completa")
        print("="*60)
        print(f"\nRun ID: {run_id}")
        print(f"Arquivos processados: {metadata['files_processed']}")
        print(f"Total de linhas: {metadata['total_rows']}")
        print(f"Colunas: {metadata['n_columns']}")
        print(f"\nSaídas:")
        print(f"  - CSV: {out_csv}")
        print(f"  - Parquet: {out_parquet}")
        print(f"  - Metadados: {meta_path}")
        print(f"  - README: {readme_path}")
        print(f"\nDiretório de execução: {run_dir}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Falha ao construir biblioteca: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
