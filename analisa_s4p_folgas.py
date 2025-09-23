#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analisa_s4p_folgas.py (versão robusta c/ baseline corrigida)

- Lê .s4p (Touchstone v2 OU v1; formatos RI/MA/DB; encodings utf-8/utf-16/le/be/latin-1);
- Extrai Txx, Tyy, Txy, Tyx da matriz S 4x4;
- Interpola em malha comum (interseção de banda);
- Compara com baseline (maior folga t_air/t_sub);
- Escolhe o menor par (t_air, t_sub) cujo erro abs no módulo de Txx/Tyy ≤ TOL_ABS.

Saídas (OUT_DIR):
- tij_interpolados.csv
- convergencia_vs_baseline.csv
- folga_recomendada.txt (se houver)
- stack_*.png, diff_*.png
"""

# =========================
# CONFIGURAÇÕES (edite aqui)
# =========================
DIRECTORY    = "./Teste_folga"    # pasta onde estão os .s4p
FILE_PATTERN = "MWS-*.s4p"        # padrão de arquivo (glob)
OUT_DIR      = "out_folgas"       # pasta de saída

FMIN_THz     = 250.0              # THz
FMAX_THz     = 330.0              # THz
NSAMPLES     = 401                # pontos da malha comum
TOL_ABS      = 0.01               # 1% de tolerância absoluta no módulo
SAVE_FIGS    = True               # salvar .png

# =========================
# IMPORTS
# =========================
import os, glob, re, csv, math, cmath
import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
    USE_PANDAS = True
except Exception:
    USE_PANDAS = False

# =========================
# UTILIDADES DE E/S
# =========================
def listar_arquivos(dirpath, pattern):
    return sorted(glob.glob(os.path.join(dirpath, pattern)))

def _try_read_text(path):
    encodings = ["utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.readlines()
        except Exception as e:
            last_err = e
    with open(path, "r", encoding="latin-1", errors="ignore") as f:
        return f.readlines()

def _normalize_line(ln: str) -> str:
    ln = ln.replace("\u2212", "-")   # minus unicode → ASCII
    ln = ln.replace(",", ".")        # vírgula decimal → ponto
    return ln.strip()

# =========================
# PARSE TOUCHSTONE
# =========================
def unidade_para_THz(token_unidade: str):
    u = token_unidade.strip().upper()
    return {"HZ":1e-12, "KHZ":1e-9, "MHZ":1e-6, "GHZ":1e-3, "THZ":1.0}.get(u, None)

def ler_params_do_cabecalho(linhas):
    params = {}
    padrao = re.compile(r"^!\s*Parameters\s*=\s*\{(.+)\}")
    par_kv = re.compile(r"([A-Za-z0-9_]+)\s*=\s*([^;]+)")
    for ln in linhas:
        m = padrao.match(ln)
        if m:
            blob = m.group(1)
            for k, v in par_kv.findall(blob):
                v = v.strip()
                try:
                    params[k] = float(v)
                except Exception:
                    params[k] = v
    return params

def _complex_from_pair(a: float, b: float, fmt: str) -> complex:
    fmt = fmt.upper()
    if fmt == "RI":      # real/imag
        return complex(a, b)
    if fmt == "MA":      # magnitude/ângulo (graus)
        return a * cmath.exp(1j * math.radians(b))
    if fmt == "DB":      # dB/ângulo (graus)
        mag = 10.0 ** (a/20.0)
        return mag * cmath.exp(1j * math.radians(b))
    raise ValueError(f"Formato não suportado: {fmt}")

def _parse_header_hash(linhas):
    hdr_re = re.compile(r"^#\s+(\w+)\s+S\s+(\w+)\s+R\s+([0-9eE\.\+\-]+)", re.IGNORECASE)
    for ln in linhas:
        if ln.lstrip().startswith("#"):
            m = hdr_re.match(ln.strip())
            if not m:
                raise ValueError(f"Header Touchstone inválido: {ln.strip()}")
            unit, fmt, zdef = m.groups()
            return unit, fmt.upper(), float(zdef)
    raise ValueError("Header '#' não encontrado.")

def ler_touchstone_generico(caminho):
    """
    Lê .s4p em:
    - v2 (com [Network Data]) OU v1 (sem blocos),
    - RI/MA/DB,
    - encodings comuns.
    Retorna: freqs_thz (Nf,), S (Nf,4,4), Zref (4,), params (dict).
    """
    raw_lines = _try_read_text(caminho)
    linhas = [_normalize_line(ln) for ln in raw_lines if ln is not None]

    params = ler_params_do_cabecalho(linhas)
    unit, fmt, Zdefault = _parse_header_hash(linhas)
    scale = unidade_para_THz(unit)
    if scale is None:
        raise ValueError(f"Unidade não suportada: {unit}")

    have_blocks = any(ln.lower().startswith("[network data]") for ln in linhas)
    if have_blocks:
        # [Reference]
        Zref = [math.nan]*4
        in_ref = False
        for ln in linhas:
            key = ln.lower()
            if key.startswith("[reference]"):
                in_ref = True; continue
            if in_ref:
                if key.startswith("["):
                    in_ref = False
                else:
                    toks = ln.split()
                    for i, tok in enumerate(toks[:4]):
                        try: Zref[i] = float(tok)
                        except Exception: pass
        Zref = np.array([Zdefault if math.isnan(z) else z for z in Zref], float)

        # [Network Data]
        in_net = False
        freqs, S_list, buf = [], [], []
        for ln in linhas:
            key = ln.lower()
            if key.startswith("[network data]"):
                in_net = True; continue
            if in_net:
                if key.startswith("[") and not key.startswith("[network data]"):
                    break
                if (not ln) or ln.startswith("!"):
                    continue
                ln_nocomment = ln.split("!")[0].strip()
                if not ln_nocomment:
                    continue
                for tok in ln_nocomment.split():
                    try: buf.append(float(tok))
                    except Exception: pass
                while len(buf) >= 33:
                    f_raw  = buf[0]
                    vals32 = buf[1:33]
                    buf    = buf[33:]
                    f_thz  = f_raw * scale
                    cpx = []
                    for i in range(0, 32, 2):
                        cpx.append(_complex_from_pair(vals32[i], vals32[i+1], fmt))
                    S = np.array(cpx, dtype=complex).reshape(4,4)
                    freqs.append(f_thz); S_list.append(S)

        freqs = np.array(freqs, float)
        if freqs.size > 0:
            return freqs, np.stack(S_list, axis=0), Zref, params
        # se não deu, cai para v1

    # v1 (sem blocos), possivelmente “quebrado”
    freqs, Smats, buf = [], [], []
    for ln in linhas:
        if (not ln) or ln.startswith("!") or ln.startswith("#") or ln.startswith("["):
            continue
        ln_nocomment = ln.split("!")[0].strip()
        if not ln_nocomment:
            continue
        for tok in ln_nocomment.split():
            try: buf.append(float(tok))
            except Exception: pass
        while len(buf) >= 33:
            f_raw  = buf[0]
            vals32 = buf[1:33]
            buf    = buf[33:]
            f_thz  = f_raw * scale
            cpx = []
            for i in range(0, 32, 2):
                cpx.append(_complex_from_pair(vals32[i], vals32[i+1], fmt))
            S = np.array(cpx, dtype=complex).reshape(4,4)
            freqs.append(f_thz); Smats.append(S)

    freqs = np.array(freqs, float)
    if freqs.size == 0:
        raise RuntimeError(
            "Não achei dados em v2 nem v1 após normalização.\n"
            "- No CST: Touchstone Version=2.0, Format=RI (ou MA/DB), Include All Ports/Modes.\n"
            "- Se for single frequency, garanta que FMIN/FMAX abrangem esse ponto (ex.: 282–282)."
        )
    Zref = np.array([Zdefault]*4, float)
    return freqs, np.stack(Smats, axis=0), Zref, params

# =========================
# PÓS-PROCESSAMENTO
# =========================
def extrair_Tij(S):
    """
    Índices (CST, 2 modos por porta):
      0: Zmin modo1 (x), 1: Zmin modo2 (y),
      2: Zmax modo1 (x), 3: Zmax modo2 (y)
    """
    return {
        "Txx": S[:, 2, 0],
        "Tyy": S[:, 3, 1],
        "Txy": S[:, 2, 1],
        "Tyx": S[:, 3, 0],
    }

def grade_comum(freqs_list, fmin_req, fmax_req, nsamples):
    mins = [max(fmin_req, float(np.min(f))) for f in freqs_list if len(f)]
    maxs = [min(fmax_req, float(np.max(f))) for f in freqs_list if len(f)]
    if not mins or not maxs:
        raise RuntimeError("Não há pontos de frequência válidos nos arquivos.")
    f_lo = max(mins); f_hi = min(maxs)
    if f_hi <= f_lo:
        raise RuntimeError(f"Interseção de banda vazia: [{f_lo:.3f}, {f_hi:.3f}] THz")
    return np.linspace(f_lo, f_hi, nsamples)

def interp_complex(x_src, y_src, x_dst):
    yr = np.interp(x_dst, x_src, np.real(y_src))
    yi = np.interp(x_dst, x_src, np.imag(y_src))
    return yr + 1j*yi

def escolher_baseline_por_folga(lista_runs):
    """
    Baseline = maior (t_air, t_sub).
    Implementação puramente Python (evita ambiguidade de tipos do NumPy).
    """
    def safe(v):  # NaN vira -inf para nunca virar baseline
        return v if (v is not None and not (isinstance(v, float) and math.isnan(v))) else -1e30
    return max(range(len(lista_runs)),
               key=lambda i: (safe(lista_runs[i]['t_air']), safe(lista_runs[i]['t_sub'])))

def salvar_fig_stack(freqs, curvas, titulo, ylabel, caminho_png):
    plt.figure(figsize=(8,4))
    for rot, y in curvas.items():
        plt.plot(freqs, np.abs(y), lw=1.2, label=rot)
    plt.xlabel("Frequência (THz)"); plt.ylabel(ylabel)
    plt.title(titulo); plt.grid(True, ls=":")
    plt.legend(ncol=2, fontsize=8); plt.tight_layout()
    plt.savefig(caminho_png, dpi=160); plt.close()

def salvar_fig_diff(freqs, diffs, tol, titulo, caminho_png):
    plt.figure(figsize=(8,4))
    for rot, y in diffs.items():
        plt.plot(freqs, y, lw=1.2, label=rot)
    plt.axhline(tol, color="k", ls="--", lw=1.0, label=f"tol={tol:g}")
    plt.xlabel("Frequência (THz)"); plt.ylabel("|Δ| absoluto no módulo")
    plt.title(titulo); plt.grid(True, ls=":")
    plt.legend(ncol=2, fontsize=8); plt.tight_layout()
    plt.savefig(caminho_png, dpi=160); plt.close()

# =========================
# PIPELINE PRINCIPAL
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    arquivos = listar_arquivos(DIRECTORY, FILE_PATTERN)
    if not arquivos:
        print("Nenhum .s4p encontrado. Confira DIRECTORY e FILE_PATTERN no cabeçalho.")
        return
    print(f"Arquivos encontrados: {len(arquivos)}")

    runs = []
    for path in arquivos:
        base = os.path.basename(path)
        try:
            freqs, S, Zref, params = ler_touchstone_generico(path)
            if freqs.size == 0:
                print(f"[vazio] {base}: sem pontos de frequência."); continue
        except Exception as e:
            print(f"[ERRO] {base}: {e}"); continue

        Tij = extrair_Tij(S)
        t_air = float(params.get("t_air", np.nan))
        t_sub = float(params.get("t_sub", np.nan))
        rotulo = f"{base} (tair={t_air:.0f},tsub={t_sub:.0f})"
        runs.append({
            "arquivo": path, "rotulo": rotulo, "freqs": freqs, "S": S,
            "Zref": Zref, "params": params, "Tij": Tij, "t_air": t_air, "t_sub": t_sub
        })
        print(f"[OK] {base}: N={freqs.size} pts, faixa=[{freqs.min():.3f},{freqs.max():.3f}] THz, "
              f"t_air={t_air}, t_sub={t_sub}")

    if not runs:
        print("Nenhum arquivo válido foi processado.")
        return

    # malha comum
    try:
        fgrid = grade_comum([r["freqs"] for r in runs], FMIN_THz, FMAX_THz, NSAMPLES)
    except RuntimeError as e:
        print(f"[ERRO] {e}")
        print("Dica: se arquivos são single-frequency (ex.: 282 THz), ajuste FMIN_THz/FMAX_THz para 282–282.")
        return

    # interpola
    curvas_por_run = {}
    for r in runs:
        Tij = r["Tij"]
        curvas_por_run[r["rotulo"]] = {
            "Txx": interp_complex(r["freqs"], Tij["Txx"], fgrid),
            "Tyy": interp_complex(r["freqs"], Tij["Tyy"], fgrid),
            "Txy": interp_complex(r["freqs"], Tij["Txy"], fgrid),
            "Tyx": interp_complex(r["freqs"], Tij["Tyx"], fgrid),
        }

    # salva CSV (malha comum)
    out_csv = os.path.join(OUT_DIR, "tij_interpolados.csv")
    if USE_PANDAS:
        linhas = []
        for rot, d in curvas_por_run.items():
            for i, f in enumerate(fgrid):
                linhas.append({
                    "file": rot, "f_THz": f,
                    "Txx_mag": np.abs(d["Txx"][i]),
                    "Tyy_mag": np.abs(d["Tyy"][i]),
                    "Txy_mag": np.abs(d["Txy"][i]),
                    "Tyx_mag": np.abs(d["Tyx"][i]),
                })
        pd.DataFrame(linhas).to_csv(out_csv, index=False)
    else:
        with open(out_csv, "w", newline="", encoding="utf-8") as g:
            w = csv.writer(g)
            w.writerow(["file","f_THz","Txx_mag","Tyy_mag","Txy_mag","Tyx_mag"])
            for rot, d in curvas_por_run.items():
                for i, f in enumerate(fgrid):
                    w.writerow([rot, f,
                                np.abs(d["Txx"][i]), np.abs(d["Tyy"][i]),
                                np.abs(d["Txy"][i]), np.abs(d["Tyx"][i])])

    # baseline = maior (t_air, t_sub) — implementação Python pura (corrigida)
    idx_base = escolher_baseline_por_folga(runs)
    rot_base = runs[idx_base]["rotulo"]; base = curvas_por_run[rot_base]
    print(f"Baseline: {rot_base}")

    # erros vs baseline
    diffs_Txx, diffs_Tyy = {}, {}
    linhas_rel, candidatos = [], []
    for r in runs:
        rot = r["rotulo"]; cur = curvas_por_run[rot]
        dTxx = np.abs(np.abs(cur["Txx"]) - np.abs(base["Txx"]))
        dTyy = np.abs(np.abs(cur["Tyy"]) - np.abs(base["Tyy"]))
        if rot != rot_base:
            diffs_Txx[rot] = dTxx; diffs_Tyy[rot] = dTyy
        eTxx = float(np.max(dTxx)); eTyy = float(np.max(dTyy))
        ok = (eTxx <= TOL_ABS) and (eTyy <= TOL_ABS)
        linhas_rel.append([rot, r["t_air"], r["t_sub"], eTxx, eTyy, int(ok)])
        if ok and np.isfinite(r["t_air"]) and np.isfinite(r["t_sub"]):
            candidatos.append((r["t_air"], r["t_sub"], rot))

    # salva relatório
    out_rel = os.path.join(OUT_DIR, "convergencia_vs_baseline.csv")
    header = ["file","t_air_nm","t_sub_nm","max_abs_diff_|Txx|","max_abs_diff_|Tyy|","pass"]
    if USE_PANDAS:
        (pd.DataFrame(linhas_rel, columns=header)
           .sort_values(["pass","t_air_nm","t_sub_nm"], ascending=[False, True, True])
           .to_csv(out_rel, index=False))
    else:
        linhas_rel_sorted = sorted(
            linhas_rel,
            key=lambda x: (1-x[5], x[1] if np.isfinite(x[1]) else 1e12,
                           x[2] if np.isfinite(x[2]) else 1e12)
        )
        with open(out_rel, "w", newline="", encoding="utf-8") as g:
            w = csv.writer(g); w.writerow(header)
            for row in linhas_rel_sorted: w.writerow(row)

    # melhor menor par que passou
    if candidatos:
        candidatos.sort(key=lambda x: (x[0], x[1]))
        t_air_best, t_sub_best, rot_best = candidatos[0]
        print(f"[OK] Folga recomendada: t_air={t_air_best:.0f} nm, t_sub={t_sub_best:.0f} nm  ({rot_best})")
        with open(os.path.join(OUT_DIR, "folga_recomendada.txt"), "w", encoding="utf-8") as g:
            g.write(f"Par recomendado: t_air={t_air_best:.0f} nm, t_sub={t_sub_best:.0f} nm  ({rot_best})\n")
    else:
        print("[ATENÇÃO] Nenhum caso cumpriu o critério de convergência. Veja convergencia_vs_baseline.csv.")

    # gráficos
    if SAVE_FIGS:
        salvar_fig_stack(fgrid, {rot:d["Txx"] for rot,d in curvas_por_run.items()},
                         "|Txx| (todas as runs)", "|Txx|", os.path.join(OUT_DIR, "stack_Txx.png"))
        salvar_fig_stack(fgrid, {rot:d["Tyy"] for rot,d in curvas_por_run.items()},
                         "|Tyy| (todas as runs)", "|Tyy|", os.path.join(OUT_DIR, "stack_Tyy.png"))
        salvar_fig_stack(fgrid, {rot:d["Txy"] for rot,d in curvas_por_run.items()},
                         "|Txy| (todas as runs)", "|Txy|", os.path.join(OUT_DIR, "stack_Txy.png"))
        salvar_fig_stack(fgrid, {rot:d["Tyx"] for rot,d in curvas_por_run.items()},
                         "|Tyx| (todas as runs)", "|Tyx|", os.path.join(OUT_DIR, "stack_Tyx.png"))

        if diffs_Txx:
            salvar_fig_diff(fgrid, diffs_Txx, TOL_ABS,
                            f"|Δ| absoluto no módulo de Txx (baseline: {rot_base})",
                            os.path.join(OUT_DIR, "diff_Txx_vs_baseline.png"))
        if diffs_Tyy:
            salvar_fig_diff(fgrid, diffs_Tyy, TOL_ABS,
                            f"|Δ| absoluto no módulo de Tyy (baseline: {rot_base})",
                            os.path.join(OUT_DIR, "diff_Tyy_vs_baseline.png"))

    print(f"Arquivos salvos em: {os.path.abspath(OUT_DIR)}")

# =========================
# RODAR
# =========================
if __name__ == "__main__":
    main()
