#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Treina e valida (parte de ML) um surrogate baseado em MLP para metassuperfícies.

NOVIDADE: além de aceitar .csv/.parquet, o script agora aceita também uma PASTA
com arquivos Touchstone (.ts). Nesse caso, ele IMPORTA o módulo 'RawDFGenerator'
(e usa a função find_file) para montar um DataFrame consolidado e salva um
CSV/Parquet da biblioteca antes de seguir para o treino.

Mapeamento de modos:
    • TE = S24
    • TM = S13

Saídas por modo (se colunas existirem):
    • {tag}_{sparam}.metrics.json
    • {tag}_{sparam}.residuos.png
    • {tag}_{sparam}.fase_hist.png
    • {tag}_{sparam}.fase_stats.json
    • {tag}_{sparam}.joblib (se joblib disponível)
E um resumo:
    • {tag}_summary.json

Excluídas: validações ópticas (far-field, SSIM/PSNR, Monte Carlo).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Import robusto do RawDFGenerator.find_file
# - Tenta via sys.path (adicionando "Essential4Libs")
# - Se falhar, procura RawDFGenerator.py e carrega por caminho (importlib)
# ---------------------------------------------------------------------
from importlib import util as _importlib_util
from importlib.machinery import SourceFileLoader as _SourceFileLoader

ts_find_file = None
_RAWDF_IMPORT_ERROR = None

def _try_import_rawdf_by_sys_path() -> bool:
    global ts_find_file, _RAWDF_IMPORT_ERROR
    try:
        from RawDFGenerator import find_file as ts_find_file  # type: ignore
        _RAWDF_IMPORT_ERROR = None
        return True
    except Exception as e:
        _RAWDF_IMPORT_ERROR = e
        return False

def _load_rawdf_from_file(py_path: Path) -> bool:
    """Carrega RawDFGenerator.py diretamente de um caminho, sem depender do sys.path."""
    global ts_find_file, _RAWDF_IMPORT_ERROR
    try:
        spec = _importlib_util.spec_from_file_location("RawDFGenerator", str(py_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"spec_from_file_location falhou para: {py_path}")
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        ts_find_file = getattr(module, "find_file", None)
        if ts_find_file is None:
            raise ImportError(f"'find_file' não encontrado dentro de {py_path}")
        _RAWDF_IMPORT_ERROR = None
        return True
    except Exception as e:
        _RAWDF_IMPORT_ERROR = e
        return False

# 1) tenta importar direto (caso já esteja no PYTHONPATH)
if not _try_import_rawdf_by_sys_path():
    # 2) adiciona "Essential4Libs" ao sys.path e tenta de novo
    _BASE_DIR = Path(__file__).resolve().parent
    _CANDIDATES = [
        _BASE_DIR / "Essential4Libs",
        _BASE_DIR,                         # por via das dúvidas
        _BASE_DIR.parent / "Essential4Libs"
    ]
    for _p in _CANDIDATES:
        if _p.exists() and _p.is_dir():
            sp = str(_p)
            if sp not in sys.path:
                sys.path.append(sp)

    if not _try_import_rawdf_by_sys_path():
        # 3) procura RawDFGenerator.py e carrega por caminho
        search_roots = [_BASE_DIR, _BASE_DIR / "Essential4Libs", _BASE_DIR.parent]
        found_path: Optional[Path] = None
        for root in search_roots:
            try:
                for hit in root.rglob("RawDFGenerator.py"):
                    found_path = hit
                    break
            except Exception:
                pass
            if found_path:
                break

        if found_path is not None:
            _load_rawdf_from_file(found_path)

# Se nada deu certo, ts_find_file continuará None e trataremos mais adiante
# (no ponto em que tentamos usar a função).

# ---------------------------------------------------------------------
# joblib é opcional; se não existir, apenas não salva o pipeline serializado
# ---------------------------------------------------------------------
try:
    import joblib  # type: ignore
except Exception:
    joblib = None  # pragma: no cover

# ============================== Parâmetros globais ==============================

# S-parameters mapeados para TE/TM conforme seu pedido:
SPARAM_TE = "S24"  # TE
SPARAM_TM = "S13"  # TM

# Reprodutibilidade e split
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20

# Filtro de frequência (GHz). 1064 nm ≈ 281_760 GHz
FREQ_GHZ_DEFAULT: float = 281_760.0
FREQ_TOL_DEFAULT: float = 5.0

# Se True, inclui 'frequencia_ghz' como feature do modelo
USE_FREQ_AS_FEATURE_DEFAULT: bool = True

# Hiperparâmetros do MLP
MLP_HIDDEN: Tuple[int, int] = (128, 64)
MLP_ACTIVATION: str = "relu"
MLP_LR_INIT: float = 1e-3
MLP_MAX_ITER: int = 5_000
MLP_EARLY_STOPPING: bool = True


# ============================== Utilitários I/O ==============================

def _basename_noext(path: Path) -> str:
    """Nome-base sem extensão (seguro para .csv/.parquet/.pq)."""
    name = path.name
    lower = name.lower()
    if lower.endswith(".csv"):
        return name[:-4]
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return name.rsplit(".", 1)[0]
    return path.stem


def _ensure_dir(folder: Path) -> Path:
    """Garante a existência do diretório de saída."""
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _save_json(data: dict, path: Path) -> None:
    """Salva um dicionário como JSON legível (UTF-8, indentado)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --------------------------- Carregamento de dados -----------------------------

def _load_dataset_from_path(input_path: Path, results_dir: Path) -> tuple[pd.DataFrame, str, Optional[Path], Optional[Path]]:
    """
    Carrega um DataFrame a partir de:
      • arquivo .csv ou .parquet; OU
      • PASTA contendo arquivos .ts (Touchstone).

    Se for pasta com .ts:
      - usa RawDFGenerator.find_file para montar o DF;
      - salva CSV/Parquet da biblioteca (padronizados) e retorna o DF já carregado.

    Retorna:
      (df, tag, csv_salvo_ou_None, parquet_salvo_ou_None)
    """
    if input_path.is_dir():
        # ----------------------- Caso PASTA de arquivos .ts -----------------------
        if ts_find_file is None:
            raise ImportError(
                "Não foi possível importar RawDFGenerator.find_file. "
                "Coloque 'RawDFGenerator.py' acessível no PYTHONPATH ou ao lado deste script.\n"
                f"Erro original: {_RAWDF_IMPORT_ERROR}"
            )

        # Monta DF a partir da pasta (não-recursivo, conforme sua função)
        df = ts_find_file(str(input_path))
        if df is None or len(df) == 0:
            raise ValueError(f"Nenhum dado .ts foi convertido na pasta: {input_path}")

        # Padroniza nomes dos arquivos de saída da biblioteca
        tag = _basename_noext(input_path)  # nome da pasta
        lib_csv = results_dir / f"{tag}_biblioteca.csv"
        lib_parquet = results_dir / f"{tag}_biblioteca.parquet"

        # Salva o DF gerado pelo RawDFGenerator para reuso
        try:
            df.to_csv(lib_csv, index=False)
        except Exception as e:
            print(f"[WARN] Falha ao salvar CSV da biblioteca: {e}")
            lib_csv = None  # type: ignore

        try:
            df.to_parquet(lib_parquet, index=False)
        except Exception as e:
            print(f"[WARN] Falha ao salvar Parquet da biblioteca: {e}")
            lib_parquet = None  # type: ignore

        return df, tag, lib_csv, lib_parquet

    # ----------------------- Caso arquivo CSV / Parquet -------------------------
    lower = input_path.suffix.lower()
    if lower == ".csv":
        df = pd.read_csv(input_path)
        tag = _basename_noext(input_path)
        return df, tag, input_path, None
    if lower in (".parquet", ".pq"):
        df = pd.read_parquet(input_path)
        tag = _basename_noext(input_path)
        return df, tag, None, input_path

    raise ValueError("Entrada não suportada. Use .csv, .parquet ou uma pasta contendo arquivos .ts.")


# ============================== Métricas e plots ==============================

def _metrics_block(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de regressão para as componentes real e imaginária, e também agregadas.

    y_true / y_pred: arrays (N, 2) com colunas [real, imag]

    Retorna:
        - r2_re, r2_im, r2_agg
        - rmse_re, rmse_im, rmse_agg
        - mae_re, mae_im, mae_agg
    """
    ytr, yti = y_true[:, 0], y_true[:, 1]
    ypr, ypi = y_pred[:, 0], y_pred[:, 1]

    r2_re = r2_score(ytr, ypr)
    r2_im = r2_score(yti, ypi)
    r2_agg = r2_score(y_true.reshape(-1), y_pred.reshape(-1))

    rmse_re = math.sqrt(mean_squared_error(ytr, ypr))
    rmse_im = math.sqrt(mean_squared_error(yti, ypi))
    rmse_agg = float(np.sqrt(np.mean((ytr - ypr) ** 2 + (yti - ypi) ** 2)))

    mae_re = mean_absolute_error(ytr, ypr)
    mae_im = mean_absolute_error(yti, ypi)
    mae_agg = 0.5 * (float(np.mean(np.abs(ytr - ypr))) + float(np.mean(np.abs(yti - ypi))))

    return {
        "r2_re": float(r2_re),
        "r2_im": float(r2_im),
        "r2_agg": float(r2_agg),
        "rmse_re": float(rmse_re),
        "rmse_im": float(rmse_im),
        "rmse_agg": float(rmse_agg),
        "mae_re": float(mae_re),
        "mae_im": float(mae_im),
        "mae_agg": float(mae_agg),
    }


def _plot_residuals_and_phase(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    out_prefix: Path,
    sparam_label: str,
) -> None:
    """
    Gera:
      1) {prefix}.residuos.png — resíduos (Re/Im) vs. L_x/L_y/(freq se existir)
      2) {prefix}.fase_hist.png — histograma do erro de fase (°)
      3) {prefix}.fase_stats.json — estatísticas do erro de fase
    """
    import matplotlib.pyplot as plt

    # Resíduos Re/Im
    res_re = y_test[:, 0] - y_pred[:, 0]
    res_im = y_test[:, 1] - y_pred[:, 1]

    feats: List[str] = ["L_x", "L_y"]
    if "frequencia_ghz" in X_test.columns:
        feats.append("frequencia_ghz")

    ncols = len(feats)
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 6), sharey="row")
    axes = np.atleast_2d(axes)

    for j, col in enumerate(feats):
        # Re
        axes[0, j].scatter(X_test[col], res_re, s=4, alpha=0.3)
        axes[0, j].axhline(0, color="k", lw=1, ls="--")
        axes[0, j].set_title(f"Resíduo Re({sparam_label}) vs {col}")
        axes[0, j].set_xlabel(col)
        axes[0, j].set_ylabel("resíduo (Re)")
        axes[0, j].grid(True, alpha=0.25)

        # Im
        axes[1, j].scatter(X_test[col], res_im, s=4, alpha=0.3)
        axes[1, j].axhline(0, color="k", lw=1, ls="--")
        axes[1, j].set_title(f"Resíduo Im({sparam_label}) vs {col}")
        axes[1, j].set_xlabel(col)
        axes[1, j].set_ylabel("resíduo (Im)")
        axes[1, j].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".residuos.png"), dpi=200)
    plt.close(fig)

    # Erro de fase
    s_true = y_test[:, 0] + 1j * y_test[:, 1]
    s_pred = y_pred[:, 0] + 1j * y_pred[:, 1]

    dphi_deg = np.degrees(np.angle(s_pred) - np.angle(s_true))
    dphi_deg = (dphi_deg + 180.0) % 360.0 - 180.0  # wrap para [-180, 180)
    abs_dphi = np.abs(dphi_deg)

    stats = {
        "mean_deg": float(dphi_deg.mean()),
        "std_deg": float(dphi_deg.std()),
        "mae_deg": float(abs_dphi.mean()),
        "p95_abs_deg": float(np.percentile(abs_dphi, 95)),
    }
    _save_json(stats, out_prefix.with_suffix(".fase_stats.json"))

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(dphi_deg, bins=100, alpha=0.85)
    ax2.set_title(f"Erro de fase (°) — {sparam_label}")
    ax2.set_xlabel("Δφ (°)")
    ax2.set_ylabel("Contagem")
    ax2.grid(True, alpha=0.30)
    fig2.tight_layout()
    fig2.savefig(out_prefix.with_suffix(".fase_hist.png"), dpi=200)
    plt.close(fig2)


# ============================== Treino de um modo ==============================

def _train_one_mode(
    df_filtered: pd.DataFrame,
    sparam: str,
    use_freq_feature: bool,
    results_dir: Path,
    tag: str,
) -> Dict[str, object]:
    """
    Treina um pipeline (Scaler + MLP) para um modo/polarização.
    - sparam: "S24" (TE) ou "S13" (TM).
    - use_freq_feature: se True, inclui 'frequencia_ghz' como feature.

    Saídas salvas (quando possível):
      • {tag}_{sparam.lower()}.metrics.json
      • {tag}_{sparam.lower()}.residuos.png
      • {tag}_{sparam.lower()}.fase_hist.png
      • {tag}_{sparam.lower()}.fase_stats.json
      • {tag}_{sparam.lower()}.joblib (se joblib disponível)
    """
    # Verifica colunas do alvo
    re_col = f"{sparam}_real"
    im_col = f"{sparam}_imag"
    if re_col not in df_filtered.columns or im_col not in df_filtered.columns:
        return {"available": False, "message": f"Colunas ausentes para {sparam} ({re_col}/{im_col})."}

    # Define features
    features: List[str] = ["L_x", "L_y"]
    if use_freq_feature and ("frequencia_ghz" in df_filtered.columns):
        features.append("frequencia_ghz")

    # Segurança: checa presença das features
    missing_feats = [c for c in features if c not in df_filtered.columns]
    if missing_feats:
        return {"available": False, "message": f"Features faltantes: {missing_feats}"}

    # Constrói X e y
    X = df_filtered[features].astype(float).copy()
    y = df_filtered[[re_col, im_col]].astype(float).to_numpy()

    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Pipeline: normalização + MLP multioutput
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), features)],
        remainder="drop",
    )
    mlp = MLPRegressor(
        hidden_layer_sizes=MLP_HIDDEN,
        activation=MLP_ACTIVATION,
        solver="adam",
        learning_rate_init=MLP_LR_INIT,
        max_iter=MLP_MAX_ITER,
        early_stopping=MLP_EARLY_STOPPING,
        random_state=RANDOM_STATE,
    )
    pipe = Pipeline(steps=[("pre", pre), ("mlp", mlp)])

    # Treina
    pipe.fit(X_train, y_train)

    # Predições
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    # Métricas
    metrics_train = _metrics_block(y_train, y_pred_train)
    metrics_test = _metrics_block(y_test, y_pred_test)

    metrics_out = {
        "mode": sparam,
        "features": features,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train": metrics_train,
        "test": metrics_test,
    }

    # Prefixo de saída baseado na tag da biblioteca
    out_prefix = results_dir / f"{tag}_{sparam.lower()}"

    _save_json(metrics_out, out_prefix.with_suffix(".metrics.json"))
    _plot_residuals_and_phase(X_test, y_test, y_pred_test, out_prefix, sparam_label=sparam)

    pipeline_path: Optional[str] = None
    if joblib is not None:
        pipeline_path = str(out_prefix.with_suffix(".joblib"))
        joblib.dump(pipe, pipeline_path)

    return {"available": True, "metrics": metrics_out, "pipeline_path": pipeline_path}


# ============================== Main (CLI) ==============================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Treina surrogate MLP (sem validações ópticas). "
            "Aceita .csv/.parquet ou uma PASTA com arquivos Touchstone (.ts)."
        )
    )
    parser.add_argument("input_path", type=str, help="Caminho para .csv/.parquet ou PASTA com .ts")
    parser.add_argument("--results-dir", type=str, default="results", help="Pasta de saída")
    parser.add_argument("--freq-ghz", type=float, default=FREQ_GHZ_DEFAULT, help="Frequência central (GHz)")
    parser.add_argument("--freq-tol", type=float, default=FREQ_TOL_DEFAULT, help="Tolerância ± (GHz)")
    parser.add_argument(
        "--use-freq-feature",
        action="store_true",
        default=USE_FREQ_AS_FEATURE_DEFAULT,
        help="Incluir 'frequencia_ghz' como feature",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limite opcional de amostras (para rodar mais rápido)",
    )
    args = parser.parse_args()

    results_dir = _ensure_dir(Path(args.results_dir))

    # 1) Carrega dataset (de arquivo ou de pasta .ts -> via RawDFGenerator)
    input_path = Path(args.input_path)
    df, tag, lib_csv, lib_parquet = _load_dataset_from_path(input_path, results_dir)

    # 2) Checagem mínima de colunas base
    for col in ("L_x", "L_y", "frequencia_ghz"):
        if col not in df.columns:
            raise ValueError(f"Coluna obrigatória ausente: {col}")

    # 3) Filtro de frequência
    f0 = float(args.freq_ghz)
    tol = float(args.freq_tol)
    mask = (df["frequencia_ghz"] >= f0 - tol) & (df["frequencia_ghz"] <= f0 + tol)
    df_filtered = df.loc[mask].copy()

    # 4) Amostragem opcional
    if args.max_samples is not None and len(df_filtered) > args.max_samples:
        df_filtered = df_filtered.sample(n=args.max_samples, random_state=RANDOM_STATE)

    if len(df_filtered) < 10:
        raise ValueError(f"Poucos dados após o filtro de frequência: {len(df_filtered)} (mín. recomendado: 10).")

    # 5) Treino TE/TM
    summary: Dict[str, object] = {
        "input": str(input_path),
        "rows_total": int(len(df)),
        "rows_filtered": int(len(df_filtered)),
        "center_freq_ghz": f0,
        "tol_ghz": tol,
        "tag": tag,
    }
    if lib_csv is not None:
        summary["library_csv"] = str(lib_csv)
    if lib_parquet is not None:
        summary["library_parquet"] = str(lib_parquet)

    te_result = _train_one_mode(
        df_filtered=df_filtered,
        sparam=SPARAM_TE,
        use_freq_feature=args.use_freq_feature,
        results_dir=results_dir,
        tag=tag,
    )

    tm_result = _train_one_mode(
        df_filtered=df_filtered,
        sparam=SPARAM_TM,
        use_freq_feature=args.use_freq_feature,
        results_dir=results_dir,
        tag=tag,
    )

    summary["TE"] = te_result
    summary["TM"] = tm_result

    _save_json(summary, results_dir / f"{tag}_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
