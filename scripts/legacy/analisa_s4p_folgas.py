#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analisa_s4p_folgas.py — Folgas (B), Malha Fixa (C) e Malha Adaptativa (D)

- Lê .s4p/.ts (Touchstone v2 ou v1; RI/MA/DB; encodings comuns).
- Extrai Txx,Tyy,Txy,Tyx; interpola para grade comum (250–330 THz, 401 pts).
- Passo B (folgas): baseline = maior (t_air,t_sub). Relatórios e pilhas.
- Passo C (malha fixa): baseline = maior Nedge. Assíntota erro×Nedge (máx e percentil).
- Passo D (malha adaptativa): compara FIXO x ADAPT. Métricas: RMS, máx95, máx.
"""

# ========================= CONFIG =========================
# Pastas/arquivos:
DIR_FOLGAS        = "./Teste_folga"
PAT_FOLGAS        = ["*.s4p", "*.ts"]

DIR_MALHA         = "./Teste_malha_1"
PAT_MALHA_FIXA    = ["250-330-28-09-4_*.*s"]   # ajuste conforme seu padrão
PAT_MALHA_ADAPT   = ["meshD_adapt_banda_*.*s"]

OUT_FOLGAS        = "out_folgas"
OUT_MALHA         = "out_malha"
OUT_MALHA_ADAPT   = "out_malha_adapt"

FMIN_THz, FMAX_THz, NSAMPLES = 250.0, 330.0, 401
TOL_ABS = 0.01
SAVE_FIGS = True

# Convergência (qual métrica usar para estimar p)
#   "max_abs"  -> usa máximo absoluto
#   "pctl"     -> usa percentil (ex.: 99%) para reduzir influência de spikes estreitos
CONV_METRIC = "pctl"
CONV_PCTL   = 99.0

# ========================= IMPORTS =========================
import os, glob, re, csv, math, cmath, warnings
import numpy as np
import matplotlib.pyplot as plt
try:
    import pandas as pd
    USE_PANDAS = True
except Exception:
    USE_PANDAS = False

# ========================= UTILS =========================
def listar_arquivos_multi(dirpath, patterns):
    files = []
    for pat in patterns:
        files += glob.glob(os.path.join(dirpath, pat))
    return sorted(set(files))

def _try_read_text(path):
    encs = ["utf-8-sig","utf-8","utf-16","utf-16-le","utf-16-be","latin-1"]
    for enc in encs:
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.readlines()
        except Exception:
            pass
    with open(path, "r", encoding="latin-1", errors="ignore") as f:
        return f.readlines()

def _normalize_line(ln: str) -> str:
    return ln.replace("\u2212","-").replace(",",".").strip()

def unidade_para_THz(token_unidade: str):
    return {"HZ":1e-12,"KHZ":1e-9,"MHZ":1e-6,"GHZ":1e-3,"THZ":1.0}.get(token_unidade.strip().upper(), None)

def ler_params_do_cabecalho(linhas):
    params = {}
    padrao = re.compile(r"^!\s*Parameters\s*=\s*\{(.+)\}")
    par_kv = re.compile(r"([A-Za-z0-9_]+)\s*=\s*([^;]+)")
    for ln in linhas:
        m = padrao.match(ln)
        if m:
            for k,v in par_kv.findall(m.group(1)):
                v = v.strip()
                try: params[k] = float(v)
                except Exception: params[k] = v
    return params

def _complex_from_pair(a,b,fmt):
    fmt = fmt.upper()
    if fmt=="RI": return complex(a,b)
    if fmt=="MA": return a*cmath.exp(1j*math.radians(b))
    if fmt=="DB": return (10.0**(a/20.0))*cmath.exp(1j*math.radians(b))
    raise ValueError(f"Formato {fmt} não suportado")

def _parse_header_hash(linhas):
    hdr = re.compile(r"^#\s+(\w+)\s+S\s+(\w+)\s+R\s+([0-9eE\.\+\-]+)", re.IGNORECASE)
    for ln in linhas:
        if ln.lstrip().startswith("#"):
            m = hdr.match(ln.strip())
            if not m: raise ValueError(f"Header inválido: {ln.strip()}")
            unit, fmt, z = m.groups()
            return unit, fmt.upper(), float(z)
    raise ValueError("Header '#' não encontrado")

def ler_touchstone_generico(path):
    raw = _try_read_text(path)
    linhas = [_normalize_line(ln) for ln in raw if ln is not None]
    params = ler_params_do_cabecalho(linhas)
    unit, fmt, Zdefault = _parse_header_hash(linhas)
    scale = unidade_para_THz(unit)
    if scale is None: raise ValueError(f"Unidade não suportada: {unit}")
    have_blocks = any(ln.lower().startswith("[network data]") for ln in linhas)

    # v2
    if have_blocks:
        # [Reference]
        Zref=[math.nan]*4; in_ref=False
        for ln in linhas:
            key=ln.lower()
            if key.startswith("[reference]"): in_ref=True; continue
            if in_ref:
                if key.startswith("["): in_ref=False
                else:
                    toks=ln.split()
                    for i,tok in enumerate(toks[:4]):
                        try: Zref[i]=float(tok)
                        except: pass
        Zref=np.array([Zdefault if math.isnan(z) else z for z in Zref], float)

        # [Network Data]
        in_net=False; freqs=[]; S_list=[]; buf=[]
        for ln in linhas:
            key=ln.lower()
            if key.startswith("[network data]"): in_net=True; continue
            if in_net:
                if key.startswith("[") and not key.startswith("[network data]"): break
                if (not ln) or ln.startswith("!"): continue
                ln_nc=ln.split("!")[0].strip()
                if not ln_nc: continue
                for tok in ln_nc.split():
                    try: buf.append(float(tok))
                    except: pass
                # 4 portas -> 32 números por linha de dados
                while len(buf)>=33:
                    f_raw=buf[0]; vals32=buf[1:33]; buf=buf[33:]
                    cpx=[_complex_from_pair(vals32[i],vals32[i+1],fmt) for i in range(0,32,2)]
                    S=np.array(cpx,dtype=complex).reshape(4,4)
                    freqs.append(f_raw*scale); S_list.append(S)
        freqs=np.array(freqs,float)
        if freqs.size>0:
            # ordena por frequência (só por segurança)
            ord_idx=np.argsort(freqs)
            return freqs[ord_idx], np.stack(S_list,axis=0)[ord_idx], Zref, params

    # v1
    freqs=[]; Smats=[]; buf=[]
    for ln in linhas:
        if (not ln) or ln.startswith("!") or ln.startswith("#") or ln.startswith("["): continue
        ln_nc=ln.split("!")[0].strip()
        if not ln_nc: continue
        for tok in ln_nc.split():
            try: buf.append(float(tok))
            except: pass
        while len(buf)>=33:
            f_raw=buf[0]; vals32=buf[1:33]; buf=buf[33:]
            cpx=[_complex_from_pair(vals32[i],vals32[i+1],fmt) for i in range(0,32,2)]
            S=np.array(cpx,dtype=complex).reshape(4,4)
            freqs.append(f_raw*scale); Smats.append(S)
    freqs=np.array(freqs,float)
    if freqs.size==0: raise RuntimeError("Não achei dados em v1 nem v2.")
    ord_idx=np.argsort(freqs)
    return freqs[ord_idx], np.stack(Smats,axis=0)[ord_idx], np.array([Zdefault]*4,float), params

def extrair_Tij(S):
    # Mapeamento CST (4-port Floquet): 0,1 = Zmin (TE/TM); 2,3 = Zmax (TE/TM)
    # Txx: Zmax modo X <- Zmin modo X  => S[2,0]
    # Tyy: Zmax modo Y <- Zmin modo Y  => S[3,1]
    # Txy: Zmax X <- Zmin Y            => S[2,1]
    # Tyx: Zmax Y <- Zmin X            => S[3,0]
    return {"Txx": S[:,2,0], "Tyy": S[:,3,1], "Txy": S[:,2,1], "Tyx": S[:,3,0]}

def grade_comum(freqs_list, fmin_req, fmax_req, nsamples):
    mins = [max(fmin_req, float(np.min(f))) for f in freqs_list if len(f)]
    maxs = [min(fmax_req, float(np.max(f))) for f in freqs_list if len(f)]
    if not mins or not maxs: raise RuntimeError("Sem pontos de frequência válidos.")
    f_lo = max(mins); f_hi = min(maxs)
    if f_hi <= f_lo: raise RuntimeError(f"Interseção de banda vazia: [{f_lo:.3f},{f_hi:.3f}] THz")
    return np.linspace(f_lo, f_hi, nsamples)

def interp_complex(x_src, y_src, x_dst):
    # interpola parte real e imaginária separadamente
    return np.interp(x_dst, x_src, np.real(y_src)) + 1j*np.interp(x_dst, x_src, np.imag(y_src))

def escolher_baseline_por_folga(lista_runs):
    def safe(v): 
        try:
            return float(v)
        except Exception:
            return -1e30
    return max(range(len(lista_runs)),
               key=lambda i: (safe(lista_runs[i].get('t_air',-1e30)), safe(lista_runs[i].get('t_sub',-1e30))))

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

def listar_runs(dirpath, patterns, label_fn=None):
    files = listar_arquivos_multi(dirpath, patterns)
    runs = []
    for path in files:
        base = os.path.basename(path)
        try:
            freqs, S, Zref, params = ler_touchstone_generico(path)
        except Exception as e:
            print(f"[ERRO] {base}: {e}"); continue
        info = {"arquivo": path, "base": base, "freqs": freqs,
                "S": S, "Zref": Zref, "params": params, "Tij": extrair_Tij(S)}
        if label_fn: info.update(label_fn(base, params))
        else: info["rotulo"] = base
        runs.append(info)
        print(f"[OK] {base}: N={freqs.size} pts, faixa=[{freqs.min():.3f},{freqs.max():.3f}] THz")
    return runs

def interpolar_todos(runs, fgrid):
    curvas = {}
    for r in runs:
        Tij = r["Tij"]
        curvas[r["rotulo"]] = {
            "Txx": interp_complex(r["freqs"], Tij["Txx"], fgrid),
            "Tyy": interp_complex(r["freqs"], Tij["Tyy"], fgrid),
            "Txy": interp_complex(r["freqs"], Tij["Txy"], fgrid),
            "Tyx": interp_complex(r["freqs"], Tij["Tyx"], fgrid),
        }
    return curvas

def _max_with_freq(diff_vec, fgrid):
    idx = int(np.argmax(diff_vec))
    return float(diff_vec[idx]), float(fgrid[idx])

# ========================= PASSO B — FOLGAS =========================
def fluxo_folgas():
    if not os.path.isdir(DIR_FOLGAS): 
        print(f"[AVISO] Pasta de folgas não encontrada: {DIR_FOLGAS}"); return
    print("\n=== Passo B: Folgas ===")
    os.makedirs(OUT_FOLGAS, exist_ok=True)

    def add_labels(base, params):
        t_air = float(params.get("t_air", np.nan)) if str(params.get("t_air","")).strip()!="" else np.nan
        t_sub = float(params.get("t_sub", np.nan)) if str(params.get("t_sub","")).strip()!="" else np.nan
        return {"t_air": t_air, "t_sub": t_sub, "rotulo": f"{base} (tair={t_air:.0f},tsub={t_sub:.0f})"}

    runs = listar_runs(DIR_FOLGAS, PAT_FOLGAS, add_labels)
    if not runs: return

    fgrid = grade_comum([r["freqs"] for r in runs], FMIN_THz, FMAX_THz, NSAMPLES)
    curvas = interpolar_todos(runs, fgrid)

    idx_base = escolher_baseline_por_folga(runs)
    rot_base = runs[idx_base]["rotulo"]; base = curvas[rot_base]
    print(f"Baseline (folgas): {rot_base}")

    diffs_Txx, diffs_Tyy, linhas_rel, candidatos = {}, {}, [], []
    for r in runs:
        rot = r["rotulo"]; cur = curvas[rot]
        dTxx = np.abs(np.abs(cur["Txx"]) - np.abs(base["Txx"]))
        dTyy = np.abs(np.abs(cur["Tyy"]) - np.abs(base["Tyy"]))
        if rot != rot_base: diffs_Txx[rot] = dTxx; diffs_Tyy[rot] = dTyy
        eTxx, fworst_x = _max_with_freq(dTxx, fgrid)
        eTyy, fworst_y = _max_with_freq(dTyy, fgrid)
        ok = (eTxx <= TOL_ABS) and (eTyy <= TOL_ABS)
        linhas_rel.append([rot, r["t_air"], r["t_sub"], eTxx, fworst_x, eTyy, fworst_y, int(ok)])
        if ok and np.isfinite(r["t_air"]) and np.isfinite(r["t_sub"]):
            candidatos.append((r["t_air"], r["t_sub"], rot))

    if USE_PANDAS:
        df = pd.DataFrame(linhas_rel, columns=["file","t_air_nm","t_sub_nm","max_|ΔTxx|","fworst_Txx_THZ","max_|ΔTyy|","fworst_Tyy_THZ","pass"])
        df.sort_values(["pass","t_air_nm","t_sub_nm"], ascending=[False,True,True]).to_csv(
            os.path.join(OUT_FOLGAS,"convergencia_vs_baseline.csv"), index=False)

    if candidatos:
        candidatos.sort(key=lambda x: (x[0],x[1]))
        with open(os.path.join(OUT_FOLGAS,"folga_recomendada.txt"),"w",encoding="utf-8") as g:
            g.write(f"Par recomendado: t_air={candidatos[0][0]:.0f} nm, t_sub={candidatos[0][1]:.0f} nm  ({candidatos[0][2]})\n")

    if SAVE_FIGS:
        salvar_fig_stack(fgrid, {rot:d["Txx"] for rot,d in curvas.items()},
                         "|Txx| (folgas)", "|Txx|", os.path.join(OUT_FOLGAS,"stack_Txx.png"))
        salvar_fig_stack(fgrid, {rot:d["Tyy"] for rot,d in curvas.items()},
                         "|Tyy| (folgas)", "|Tyy|", os.path.join(OUT_FOLGAS,"stack_Tyy.png"))
        salvar_fig_stack(fgrid, {rot:d["Txy"] for rot,d in curvas.items()},
                         "|Txy| (folgas)", "|Txy|", os.path.join(OUT_FOLGAS,"stack_Txy.png"))
        salvar_fig_stack(fgrid, {rot:d["Tyx"] for rot,d in curvas.items()},
                         "|Tyx| (folgas)", "|Tyx|", os.path.join(OUT_FOLGAS,"stack_Tyx.png"))
        if diffs_Txx:
            salvar_fig_diff(fgrid, diffs_Txx, TOL_ABS,
                            f"|Δ| em |Txx| vs baseline {rot_base}",
                            os.path.join(OUT_FOLGAS,"diff_Txx_vs_baseline.png"))
        if diffs_Tyy:
            salvar_fig_diff(fgrid, diffs_Tyy, TOL_ABS,
                            f"|Δ| em |Tyy| vs baseline {rot_base}",
                            os.path.join(OUT_FOLGAS,"diff_Tyy_vs_baseline.png"))

# ========================= PASSO C — MALHA FIXA =========================
def _inferir_Nedge(base, params):
    # 1) Cabeçalho Parameters (preferência)
    n = params.get("Nedge", None)
    try:
        if n is not None:
            return int(round(float(n)))
    except Exception:
        pass
    # 2) Nome do arquivo: aceita "Nedge=12", "Nedge-12", "_Nedge12", "Nedge12", "_N12"
    pats = [
        r"[(_-]Nedge[=_-]?(\d+)",
        r"\bNedge[=_-]?(\d+)\b",
        r"[(_-]N(\d+)\b"
    ]
    for pat in pats:
        m = re.search(pat, base, flags=re.IGNORECASE)
        if m:
            try: return int(m.group(1))
            except: pass
    return None

def _erro_metricas(cur, base, fgrid):
    # diferenças ponto a ponto
    dTxx = np.abs(np.abs(cur["Txx"]) - np.abs(base["Txx"]))
    dTyy = np.abs(np.abs(cur["Tyy"]) - np.abs(base["Tyy"]))
    worst = np.maximum(dTxx, dTyy)

    # máximos e frequências
    eTxx, fworst_x = _max_with_freq(dTxx, fgrid)
    eTyy, fworst_y = _max_with_freq(dTyy, fgrid)
    eworst, fworst = _max_with_freq(worst, fgrid)

    # percentil (para reduzir impacto de spikes)
    pctl = float(np.percentile(worst, CONV_PCTL))

    return dict(
        dTxx=dTxx, dTyy=dTyy, worst=worst,
        eTxx=eTxx, fworst_x=fworst_x,
        eTyy=eTyy, fworst_y=fworst_y,
        eworst=eworst, fworst=fworst,
        pctl=pctl
    )

def fluxo_malha_fixa():
    if not os.path.isdir(DIR_MALHA):
        print(f"[AVISO] Pasta de malha não encontrada: {DIR_MALHA}"); return
    print("\n=== Passo C: Convergência de malha (FIXA) ===")
    os.makedirs(OUT_MALHA, exist_ok=True)

    def add_labels(base, params):
        nedge = _inferir_Nedge(base, params)
        return {"Nedge": nedge, "rotulo": f"{base}" if nedge is None else f"{base} (Nedge={nedge})"}

    runs = listar_runs(DIR_MALHA, PAT_MALHA_FIXA, add_labels)
    if not runs: return

    fgrid = grade_comum([r["freqs"] for r in runs], FMIN_THz, FMAX_THz, NSAMPLES)
    curvas = interpolar_todos(runs, fgrid)

    # baseline = maior Nedge (se faltar, pega o último)
    n_list = [(i, r.get("Nedge", -1)) for i,r in enumerate(runs)]
    n_list.sort(key=lambda x: x[1])
    idx_base = n_list[-1][0]
    rot_base = runs[idx_base]["rotulo"]; base = curvas[rot_base]
    print(f"Baseline (malha fixa): {rot_base}")

    diffs_Txx, diffs_Tyy, linhas = {}, {}, []
    for r in runs:
        rot=r["rotulo"]; cur=curvas[rot]
        met = _erro_metricas(cur, base, fgrid)
        if rot!=rot_base:
            diffs_Txx[rot]=met["dTxx"]; diffs_Tyy[rot]=met["dTyy"]
        linhas.append({
            "file":rot,
            "Nedge":r.get("Nedge",np.nan),
            "max_abs_diff_|Txx|":met["eTxx"], "fworst_Txx_THZ":met["fworst_x"],
            "max_abs_diff_|Tyy|":met["eTyy"], "fworst_Tyy_THZ":met["fworst_y"],
            "max_abs_diff_worst":met["eworst"], "fworst_THZ":met["fworst"],
            f"pctl{CONV_PCTL:.0f}_worst":met["pctl"]
        })

    if USE_PANDAS:
        pd.DataFrame(linhas).sort_values("Nedge").to_csv(os.path.join(OUT_MALHA,"convergencia_malha.csv"),index=False)

    # assíntota erro×Nedge (escolha da métrica)
    n_vals=[]; e_vals=[]
    for d in linhas:
        n=d["Nedge"]
        if not (isinstance(n,(int,float)) and np.isfinite(n)): 
            continue
        if CONV_METRIC=="pctl":
            e = d[f"pctl{CONV_PCTL:.0f}_worst"]
        else:
            e = d["max_abs_diff_worst"]
        if e>0: 
            n_vals.append(float(n)); e_vals.append(float(e))

    p_est=np.nan; a_est=np.nan
    if len(n_vals)>=2 and np.max(e_vals) > 1.001*np.min(e_vals):
        x=np.log(n_vals); y=np.log(e_vals)
        m,c=np.polyfit(x,y,1); p_est=-m; a_est=np.exp(c)
    else:
        warnings.warn("Erros quase constantes entre rodadas — não é possível estimar p de forma confiável.")

    if SAVE_FIGS:
        salvar_fig_stack(fgrid, {rot:d["Txx"] for rot,d in curvas.items()},
                         "|Txx| (malha fixa)", "|Txx|", os.path.join(OUT_MALHA,"stack_Txx.png"))
        salvar_fig_stack(fgrid, {rot:d["Tyy"] for rot,d in curvas.items()},
                         "|Tyy| (malha fixa)", "|Tyy|", os.path.join(OUT_MALHA,"stack_Tyy.png"))
        if diffs_Txx:
            salvar_fig_diff(fgrid, diffs_Txx, TOL_ABS,
                            f"|Δ| em |Txx| vs baseline {rot_base}",
                            os.path.join(OUT_MALHA,"diff_Txx_vs_baseline.png"))
        if diffs_Tyy:
            salvar_fig_diff(fgrid, diffs_Tyy, TOL_ABS,
                            f"|Δ| em |Tyy| vs baseline {rot_base}",
                            os.path.join(OUT_MALHA,"diff_Tyy_vs_baseline.png"))
        if len(n_vals)>=2:
            xs=np.linspace(min(n_vals),max(n_vals),100)
            fit=a_est*xs**(-p_est) if np.isfinite(p_est) else None
            plt.figure(figsize=(7.2,4.2))
            plt.loglog(n_vals,e_vals,'o-',lw=1.2,label=f"erro ({'pctl' if CONV_METRIC=='pctl' else 'máx'})")
            if fit is not None: plt.loglog(xs,fit,'--',label=f"ajuste ~ a·Nedge^(-p), p={p_est:.2f}")
            plt.axhline(TOL_ABS,color='k',ls=':',lw=1.0,label=f"tol={TOL_ABS:g}")
            plt.xlabel("Nedge"); plt.ylabel("erro na banda (|Δ|)")
            plt.title(f"Assíntota de convergência (malha fixa) — métrica: {CONV_METRIC}")
            plt.grid(True,which='both',ls=":"); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(OUT_MALHA,"assintota_erro_vs_Nedge.png"),dpi=160); plt.close()

    with open(os.path.join(OUT_MALHA,"resumo_malha.txt"),"w",encoding="utf-8") as g:
        g.write(f"Baseline (maior Nedge): {rot_base}\n")
        if np.isfinite(p_est):
            g.write(f"Ordem de convergência estimada (métrica={CONV_METRIC}): p ≈ {p_est:.2f}\n")
        for d in sorted(linhas,key=lambda x: (np.inf if math.isnan(x['Nedge']) else x['Nedge'])):
            g.write(f"Nedge={d['Nedge']}, eTxx={d['max_abs_diff_|Txx|']:.5g} @ {d['fworst_Txx_THZ']:.3f} THz, "
                    f"eTyy={d['max_abs_diff_|Tyy|']:.5g} @ {d['fworst_Tyy_THZ']:.3f} THz, "
                    f"worst={d['max_abs_diff_worst']:.5g} @ {d['fworst_THZ']:.3f} THz, "
                    f"pctl{CONV_PCTL:.0f}={d[f'pctl{CONV_PCTL:.0f}_worst']:.5g}\n")

# ========================= PASSO D — MALHA ADAPTATIVA =========================
def fluxo_malha_adapt():
    if not os.path.isdir(DIR_MALHA):
        print(f"[AVISO] Pasta de malha não encontrada: {DIR_MALHA}"); return
    print("\n=== Passo D: Malha adaptativa ===")
    os.makedirs(OUT_MALHA_ADAPT, exist_ok=True)

    def add_labels_c(base, params):
        nedge = _inferir_Nedge(base, params)
        return {"tipo":"FIXO","Nedge": nedge, "rotulo": f"{base} (Nedge={nedge}, FIXO)"}
    def add_labels_d(base, params):
        nedge = _inferir_Nedge(base, params)
        return {"tipo":"ADAPT","Nedge": nedge, "rotulo": f"{base} (ADAPT)" if nedge is None else f"{base} (Nedge={nedge}, ADAPT)"}

    runs_c = listar_runs(DIR_MALHA, PAT_MALHA_FIXA, add_labels_c)
    runs_d = listar_runs(DIR_MALHA, PAT_MALHA_ADAPT, add_labels_d)
    if not runs_c or not runs_d: 
        print("[AVISO] Faltam arquivos FIXO ou ADAPT nesta pasta.")
        return

    fgrid = grade_comum([r["freqs"] for r in runs_c+runs_d], FMIN_THz, FMAX_THz, NSAMPLES)
    curvas_c = interpolar_todos(runs_c, fgrid)
    curvas_d = interpolar_todos(runs_d, fgrid)

    # baseline = maior Nedge FIXO
    n_list = [(i, r.get("Nedge", -1)) for i,r in enumerate(runs_c)]
    n_list.sort(key=lambda x: x[1])
    rot_base = runs_c[n_list[-1][0]]["rotulo"]; base = curvas_c[rot_base]
    print(f"Baseline Passo D: {rot_base}")

    def metricas(cur):
        dxx = np.abs(np.abs(cur["Txx"]) - np.abs(base["Txx"]))
        dyy = np.abs(np.abs(cur["Tyy"]) - np.abs(base["Tyy"]))
        worst = np.maximum(dxx, dyy)
        rms = float(np.sqrt(np.mean(worst**2)))
        max95 = float(np.percentile(worst, 95.0))
        maxabs = float(np.max(worst))
        return rms, max95, maxabs, dxx, dyy, worst

    # escolher FIXO(s) alvo: preferir 24 e 30; senão, dois maiores
    nomes_fixo_alvo = []
    mapa = {r.get("Nedge",None): r["rotulo"] for r in runs_c if r.get("Nedge",None) is not None}
    for alvo in (24, 30):
        if alvo in mapa: nomes_fixo_alvo.append(mapa[alvo])
    if len(nomes_fixo_alvo) < 2:
        top = [rot for (_,rot) in sorted([(r.get("Nedge",-1), r["rotulo"]) for r in runs_c])[-2:]]
        for rot in top:
            if rot not in nomes_fixo_alvo: nomes_fixo_alvo.append(rot)

    registros = []

    # FIXO(s)
    for rot in nomes_fixo_alvo:
        cur = curvas_c[rot]; rms,max95,maxabs,_,_,_ = metricas(cur)
        registros.append(("FIXO", rot, rms, max95, maxabs))
    # ADAPT(s)
    for r in runs_d:
        rot = r["rotulo"]; cur = curvas_d[rot]
        rms,max95,maxabs,_,_,_ = metricas(cur)
        registros.append(("ADAPT", rot, rms, max95, maxabs))

    if USE_PANDAS:
        df = pd.DataFrame(registros, columns=["tipo","rotulo","rms_worst","max95_worst","max_abs_worst"])
        df.to_csv(os.path.join(OUT_MALHA_ADAPT,"metricas_adapt.csv"), index=False)

        # barras
        labels = df["rotulo"].tolist()
        x = np.arange(len(labels)); w = 0.28
        plt.figure(figsize=(10.5,4.5))
        plt.bar(x- w, df["rms_worst"],   width=w, label="RMS")
        plt.bar(x    , df["max95_worst"],width=w, label="máx95")
        plt.bar(x+ w, df["max_abs_worst"],width=w, label="máx abs")
        plt.xticks(x, labels, rotation=25, ha="right", fontsize=8)
        plt.ylabel("erro em |T|"); plt.title("Passo D: métricas vs baseline (FIXO Nedge máximo)")
        plt.grid(True, axis="y", ls=":"); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_MALHA_ADAPT,"metricas_barras.png"), dpi=160)
        plt.close()

    # curvas de diferença (opcional)
    if SAVE_FIGS:
        # |Δ| em Txx
        plt.figure(figsize=(9,4))
        for rot in nomes_fixo_alvo:
            cur = curvas_c[rot]
            dxx = np.abs(np.abs(cur["Txx"])-np.abs(base["Txx"]))
            plt.plot(fgrid, dxx, lw=1.2, label=f"{rot}")
        for r in runs_d:
            rot=r["rotulo"]; cur=curvas_d[rot]
            dxx = np.abs(np.abs(cur["Txx"])-np.abs(base["Txx"]))
            plt.plot(fgrid, dxx, lw=1.2, ls="--", label=f"{rot}")
        plt.axhline(TOL_ABS,color="k",ls=":",label=f"tol={TOL_ABS:g}")
        plt.xlabel("Frequência (THz)"); plt.ylabel("|Δ| em |Txx|")
        plt.title("Passo D: |Δ| em |Txx| (contínuo=FIXO, tracejado=ADAPT)")
        plt.grid(True, ls=":"); plt.legend(ncol=2, fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(OUT_MALHA_ADAPT,"delta_Txx.png"), dpi=160); plt.close()

        # |Δ| em Tyy
        plt.figure(figsize=(9,4))
        for rot in nomes_fixo_alvo:
            cur = curvas_c[rot]
            dyy = np.abs(np.abs(cur["Tyy"])-np.abs(base["Tyy"]))
            plt.plot(fgrid, dyy, lw=1.2, label=f"{rot}")
        for r in runs_d:
            rot=r["rotulo"]; cur=curvas_d[rot]
            dyy = np.abs(np.abs(cur["Tyy"])-np.abs(base["Tyy"]))
            plt.plot(fgrid, dyy, lw=1.2, ls="--", label=f"{rot}")
        plt.axhline(TOL_ABS,color="k",ls=":",label=f"tol={TOL_ABS:g}")
        plt.xlabel("Frequência (THz)"); plt.ylabel("|Δ| em |Tyy|")
        plt.title("Passo D: |Δ| em |Tyy| (contínuo=FIXO, tracejado=ADAPT)")
        plt.grid(True, ls=":"); plt.legend(ncol=2, fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(OUT_MALHA_ADAPT,"delta_Tyy.png"), dpi=160); plt.close()

# ========================= MAIN =========================
def main():
    fluxo_folgas()
    fluxo_malha_fixa()
    fluxo_malha_adapt()

if __name__ == "__main__":
    main()
