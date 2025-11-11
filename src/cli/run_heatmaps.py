#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI tool for generating heatmap visualizations from metasurface library.

Command-line interface to compute and visualize amplitude and phase heatmaps
over the L_x × L_y parameter space.

Interface de linha de comando para computar e visualizar mapas de calor de
amplitude e fase sobre o espaço de parâmetros L_x × L_y.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

# Adicionar diretório pai to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from meta_library import phase_matching

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
    """
    Configure command-line arguments.
    
    Configura argumentos da linha de comando.
    """
    parser = argparse.ArgumentParser(
        description="Generate heatmap visualizations from metasurface library / "
                   "Gerar visualizações de mapas de calor da biblioteca de metassuperfície",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    parser.add_argument(
        "--library",
        type=Path,
        required=True,
        help="Input library file (CSV or Parquet) / "
             "Arquivo de biblioteca de entrada (CSV ou Parquet)"
    )
    
    # Heatmap options
    parser.add_argument(
        "--fields",
        type=str,
        nargs="+",
        default=["phase_TE", "amp_TE", "phase_TM", "amp_TM"],
        help="Fields to generate heatmaps for / "
             "Campos para gerar mapas de calor"
    )
    parser.add_argument(
        "--x-col",
        type=str,
        default="L_x",
        help="Column for x-axis / Coluna para eixo x"
    )
    parser.add_argument(
        "--y-col",
        type=str,
        default="L_y",
        help="Column for y-axis / Coluna para eixo y"
    )
    parser.add_argument(
        "--bins-x",
        type=int,
        default=100,
        help="Number of bins in x direction / Número de bins na direção x"
    )
    parser.add_argument(
        "--bins-y",
        type=int,
        default=100,
        help="Number of bins in y direction / Número de bins na direção y"
    )
    
    # Visualization options
    parser.add_argument(
        "--colormap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name / Nome do mapa de cores do Matplotlib"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution (DPI) / Resolução da figura (DPI)"
    )
    
    # Output options
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory for heatmaps / Diretório de saída para mapas de calor"
    )
    
    # Metadata options
    parser.add_argument(
        "--experiment",
        type=str,
        default="heatmaps",
        help="Experiment name for organization / Nome do experimento para organização"
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Root output directory (default: results/meta_library/heatmaps) / "
             "Diretório raiz de saída (padrão: results/meta_library/heatmaps)"
    )
    
    return parser.parse_args()


def create_readme(run_dir: Path, metadata: dict) -> Path:
    """
    Create README.md with run information.
    
    Cria README.md com informações da execução.
    """
    readme_path = run_dir / "README.md"
    
    content = f"""# Heatmaps Generation Run

## Resumo / Resumo

Heatmap visualizations of metasurface parameter space showing amplitude and phase distributions.

Visualizações de mapas de calor do espaço de parâmetros de metassuperfície mostrando distribuições de amplitude e fase.

## Run Information / Informações da Execução

- **Run ID**: {metadata['run_id']}
- **Experiment**: {metadata['experiment']}
- **Timestamp**: {metadata['timestamp']}
- **Library File**: {metadata['library_file']}

## Configuration / Configuração

- **Fields**: {', '.join(metadata['fields'])}
- **X Column**: {metadata['x_col']}
- **Y Column**: {metadata['y_col']}
- **Grid Size**: {metadata['bins_x']} × {metadata['bins_y']}
- **Colormap**: {metadata['colormap']}

## Results / Resultados

- **Heatmaps Generated**: {metadata['n_heatmaps']}
- **Output Directory**: {metadata['output_dir']}

### Generated Files / Arquivos Gerados

"""
    
    for field in metadata['fields']:
        content += f"- `heatmap_{field}.png` - Visualization\n"
        content += f"- `heatmap_{field}.npy` - Raw data array\n"
    
    content += f"\n## Reprodutibilidade / Reprodutibilidade\n\n"
    content += f"```bash\n"
    content += f"python src/cli/run_heatmaps.py \\\n"
    content += f"  --library \"{metadata['library_file']}\" \\\n"
    content += f"  --fields {' '.join(metadata['fields'])} \\\n"
    content += f"  --x-col {metadata['x_col']} --y-col {metadata['y_col']} \\\n"
    content += f"  --bins-x {metadata['bins_x']} --bins-y {metadata['bins_y']} \\\n"
    content += f"  --colormap {metadata['colormap']} \\\n"
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
        out_root = repo_root / "results" / "meta_library" / "heatmaps"
    else:
        out_root = args.out_root
    
    # Create run directory
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if args.out_dir:
        run_dir = args.out_dir
    else:
        run_dir = out_root / args.experiment / run_id
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting heatmap generation: {args.experiment}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Library file: {args.library}")
    logger.info(f"Output directory: {run_dir}")
    
    try:
        # Read library
        logger.info(f"Reading library file...")
        if str(args.library).endswith('.parquet'):
            df = pd.read_parquet(args.library)
        else:
            df = pd.read_csv(args.library)
        
        logger.info(f"Loaded library with {len(df)} rows")
        
        # Compute heatmaps
        logger.info(f"Computing heatmaps for fields: {args.fields}")
        heatmaps = phase_matching.compute_heatmaps(
            df,
            x=args.x_col,
            y=args.y_col,
            fields=tuple(args.fields),
            bins_x=args.bins_x,
            bins_y=args.bins_y
        )
        
        # Save figures
        logger.info(f"Saving heatmap visualizations...")
        saved_files = phase_matching.save_heatmap_figures(
            heatmaps,
            out_dir=run_dir,
            prefix="heatmap",
            colormap=args.colormap,
            dpi=args.dpi
        )
        
        # Create metadata
        metadata = {
            "run_id": run_id,
            "experiment": args.experiment,
            "timestamp": datetime.now().isoformat(),
            "library_file": str(args.library),
            "fields": args.fields,
            "x_col": args.x_col,
            "y_col": args.y_col,
            "bins_x": args.bins_x,
            "bins_y": args.bins_y,
            "colormap": args.colormap,
            "n_heatmaps": len(args.fields),
            "output_dir": str(run_dir),
            "saved_files": [str(f) for f in saved_files],
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
        print("✅ Heatmap Generation Complete / Geração de Mapas de Calor Completa")
        print("="*60)
        print(f"\nRun ID: {run_id}")
        print(f"Heatmaps generated: {len(args.fields)}")
        print(f"Grid size: {args.bins_x} × {args.bins_y}")
        print(f"\nGenerated files:")
        for field in args.fields:
            print(f"  - heatmap_{field}.png")
            print(f"  - heatmap_{field}.npy")
        print(f"\nMetadata: {meta_path}")
        print(f"README: {readme_path}")
        print(f"\nOutput directory: {run_dir}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to generate heatmaps: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
