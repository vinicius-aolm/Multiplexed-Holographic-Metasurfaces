#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI tool for performing phase matching optimization.

Command-line interface to find optimal metasurface layouts by matching
target TE/TM phase profiles from the library.

Interface de linha de comando para encontrar layouts ótimos de metassuperfície
casando perfis de fase TE/TM alvo da biblioteca.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        description="Perform phase matching optimization for metasurface layout / "
                   "Realizar otimização de casamento de fase para layout de metassuperfície",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    parser.add_argument(
        "--library",
        type=Path,
        required=True,
        help="Input library file (CSV or Parquet) with derived columns / "
             "Arquivo de biblioteca de entrada (CSV ou Parquet) com colunas derivadas"
    )
    parser.add_argument(
        "--target-te",
        type=Path,
        required=True,
        help="Target TE phase map (.npy or .npz file) / "
             "Mapa de fase TE alvo (arquivo .npy ou .npz)"
    )
    parser.add_argument(
        "--target-tm",
        type=Path,
        required=True,
        help="Target TM phase map (.npy or .npz file) / "
             "Mapa de fase TM alvo (arquivo .npy ou .npz)"
    )
    
    # Phase matching options
    parser.add_argument(
        "--use-height",
        action="store_true",
        help="Filter library entries by height / Filtrar entradas da biblioteca por altura"
    )
    parser.add_argument(
        "--height-col",
        type=str,
        default="H",
        help="Column name for height parameter / Nome da coluna para parâmetro de altura"
    )
    parser.add_argument(
        "--target-height",
        type=float,
        help="Target height value (if not specified, uses median) / "
             "Valor de altura alvo (se não especificado, usa mediana)"
    )
    
    # Output options
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory for results / Diretório de saída para resultados"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate preview figure with target and result comparison / "
             "Gerar figura de prévia com comparação entre alvo e resultado"
    )
    
    # Metadata options
    parser.add_argument(
        "--experiment",
        type=str,
        default="phase_matching",
        help="Experiment name for organization / Nome do experimento para organização"
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Root output directory (default: results/meta_library/phase_matching) / "
             "Diretório raiz de saída (padrão: results/meta_library/phase_matching)"
    )
    
    return parser.parse_args()


def load_target_phase(path: Path, key: str = None) -> np.ndarray:
    """
    Load target phase map from file.
    
    Carrega mapa de fase alvo de arquivo.
    """
    if path.suffix == '.npz':
        data = np.load(path)
        if key is None:
            # Try common keys
            for k in ['phase', 'arr_0', 'data']:
                if k in data:
                    return data[k]
            # Use first available key
            return data[list(data.keys())[0]]
        return data[key]
    else:  # .npy
        return np.load(path)


def create_preview_figure(
    target_te: np.ndarray,
    target_tm: np.ndarray,
    layout_lx: np.ndarray,
    layout_ly: np.ndarray,
    error_map: np.ndarray,
    out_path: Path
):
    """
    Create comprehensive preview figure.
    
    Cria figura de prévia abrangente.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Targets
    im0 = axes[0, 0].imshow(target_te, cmap='twilight')
    axes[0, 0].set_title('Target Phase TE')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(target_tm, cmap='twilight')
    axes[0, 1].set_title('Target Phase TM')
    plt.colorbar(im1, ax=axes[0, 1])
    
    axes[0, 2].axis('off')
    axes[0, 2].text(0.5, 0.5, 'Phase Matching\nOptimization',
                   ha='center', va='center', fontsize=16, weight='bold')
    
    # Row 2: Results
    im3 = axes[1, 0].imshow(layout_lx, cmap='viridis')
    axes[1, 0].set_title('Layout L_x')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(layout_ly, cmap='viridis')
    axes[1, 1].set_title('Layout L_y')
    plt.colorbar(im4, ax=axes[1, 1])
    
    im5 = axes[1, 2].imshow(error_map, cmap='hot')
    axes[1, 2].set_title(f'RMS Error (mean={error_map.mean():.4f})')
    plt.colorbar(im5, ax=axes[1, 2])
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def create_readme(run_dir: Path, metadata: dict) -> Path:
    """
    Create README.md with run information.
    
    Cria README.md com informações da execução.
    """
    readme_path = run_dir / "README.md"
    
    content = f"""# Execução de Casamento de Fase

## Resumo / Resumo

Phase matching optimization to find optimal metasurface layout for target TE/TM phase profiles.

Otimização de casamento de fase para encontrar layout ótimo de metassuperfície para perfis de fase TE/TM alvo.

## Run Information / Informações da Execução

- **Run ID**: {metadata['run_id']}
- **Experiment**: {metadata['experiment']}
- **Timestamp**: {metadata['timestamp']}
- **Library File**: {metadata['library_file']}
- **Target TE**: {metadata['target_te_file']}
- **Target TM**: {metadata['target_tm_file']}

## Configuration / Configuração

- **Target Shape**: {metadata['target_shape']}
- **Use Height Filter**: {metadata['use_height']}
"""
    
    if metadata['use_height']:
        content += f"- **Target Height**: {metadata.get('target_height', 'auto')}\n"
        content += f"- **Height Column**: {metadata['height_col']}\n"
    
    content += f"\n## Resultados / Resultados\n\n"
    content += f"- **Mean RMS Error**: {metadata['mean_error']:.6f} rad\n"
    content += f"- **Max RMS Error**: {metadata['max_error']:.6f} rad\n"
    content += f"- **Min RMS Error**: {metadata['min_error']:.6f} rad\n"
    
    content += f"\n### Output Files / Arquivos de Saída\n\n"
    content += f"- `layout_lx.csv` - L_x values for each pixel\n"
    content += f"- `layout_ly.csv` - L_y values for each pixel\n"
    content += f"- `layout_error_map.csv` - RMS error at each pixel\n"
    content += f"- `layout_summary.png` - Visualization summary\n"
    if metadata.get('preview_generated'):
        content += f"- `preview.png` - Comparison with targets\n"
    
    content += f"\n## Reprodutibilidade / Reprodutibilidade\n\n"
    content += f"```bash\n"
    content += f"python src/cli/run_phase_matching.py \\\n"
    content += f"  --library \"{metadata['library_file']}\" \\\n"
    content += f"  --target-te \"{metadata['target_te_file']}\" \\\n"
    content += f"  --target-tm \"{metadata['target_tm_file']}\" \\\n"
    if metadata['use_height']:
        content += f"  --use-height \\\n"
        content += f"  --height-col {metadata['height_col']} \\\n"
        if metadata.get('target_height'):
            content += f"  --target-height {metadata['target_height']} \\\n"
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
        out_root = repo_root / "results" / "meta_library" / "phase_matching"
    else:
        out_root = args.out_root
    
    # Create run directory
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if args.out_dir:
        run_dir = args.out_dir
    else:
        run_dir = out_root / args.experiment / run_id
    
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting phase matching: {args.experiment}")
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
        
        # Load target phases
        logger.info(f"Loading target phase maps...")
        target_te = load_target_phase(args.target_te)
        target_tm = load_target_phase(args.target_tm)
        
        logger.info(f"Target TE shape: {target_te.shape}")
        logger.info(f"Target TM shape: {target_tm.shape}")
        
        # Perform phase matching
        logger.info(f"Performing phase matching optimization...")
        layout_lx, layout_ly, error_map = phase_matching.perform_phase_matching(
            df,
            target_phase_tm=target_tm,
            target_phase_te=target_te,
            use_height=args.use_height,
            height_col=args.height_col,
            target_height=args.target_height
        )
        
        # Save outputs
        logger.info(f"Saving layout results...")
        saved_paths = phase_matching.save_layout_outputs(
            layout_lx, layout_ly, error_map,
            out_dir=run_dir,
            prefix="layout"
        )
        
        # Generate preview if requested
        preview_generated = False
        if args.preview:
            logger.info(f"Generating preview figure...")
            preview_path = run_dir / "preview.png"
            create_preview_figure(
                target_te, target_tm,
                layout_lx, layout_ly, error_map,
                preview_path
            )
            preview_generated = True
        
        # Create metadata
        metadata = {
            "run_id": run_id,
            "experiment": args.experiment,
            "timestamp": datetime.now().isoformat(),
            "library_file": str(args.library),
            "target_te_file": str(args.target_te),
            "target_tm_file": str(args.target_tm),
            "target_shape": list(target_te.shape),
            "use_height": args.use_height,
            "height_col": args.height_col,
            "target_height": args.target_height,
            "mean_error": float(error_map.mean()),
            "max_error": float(error_map.max()),
            "min_error": float(error_map.min()),
            "preview_generated": preview_generated,
            "output_files": {k: str(v) for k, v in saved_paths.items()},
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
        print("✅ Phase Matching Complete / Casamento de Fase Completo")
        print("="*60)
        print(f"\nRun ID: {run_id}")
        print(f"Target shape: {target_te.shape}")
        print(f"\nError Statistics:")
        print(f"  Mean RMS: {metadata['mean_error']:.6f} rad")
        print(f"  Max RMS:  {metadata['max_error']:.6f} rad")
        print(f"  Min RMS:  {metadata['min_error']:.6f} rad")
        print(f"\nOutput files:")
        print(f"  - layout_lx.csv")
        print(f"  - layout_ly.csv")
        print(f"  - layout_error_map.csv")
        print(f"  - layout_summary.png")
        if preview_generated:
            print(f"  - preview.png")
        print(f"\nMetadata: {meta_path}")
        print(f"README: {readme_path}")
        print(f"\nOutput directory: {run_dir}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to perform phase matching: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
