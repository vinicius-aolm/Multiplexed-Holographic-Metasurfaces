#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analise_convergencia_cst.py

Script aprimorado para análise de convergência de simulações do CST Studio Suite,
com foco em estudos de malha local para estruturas ressonantes.

FUNCIONALIDADES:
- Lê arquivos Touchstone (.s4p, .ts) de forma robusta (v1/v2, RI/MA/DB).
- Extrai parâmetros de transmissão (Txx, Tyy, Txy, Tyx) e os interpola em uma grade de frequência comum.
- **Passo B (Folgas):** Analisa a convergência em relação às folgas de simulação (t_air, t_sub).
- **Passo C (Malha Local):** Analisa a convergência da malha local ('mesh_pilar'), que é a estratégia recomendada.
- **Novas Métricas de Convergência:**
  - Detecta e plota a frequência de ressonância (f_res) vs. refinamento da malha.
  - Detecta e plota a amplitude máxima da transmissão (T_max) vs. refinamento da malha.
- **Relatórios Completos:** Gera gráficos para todas as componentes de transmissão (co-pol e cross-pol),
  gráficos de erro, e relatórios em CSV e TXT com todas as métricas.
"""

# ========================= CONFIGURAÇÃO =========================
# Configure as pastas e os padrões de nome de arquivo para cada passo.

# --- PASSO B: Estudo de Convergência das Folgas ---
DIR_FOLGAS = "./Teste_folga"
PAT_FOLGAS = ["*.s4p", "*.ts"]
OUT_FOLGAS = "out_analise_folgas"

# --- PASSO C: Estudo de Convergência da Malha Local (Pilar) ---
# O script agora é focado em varrer a malha local do pilar.
DIR_MALHA_LOCAL = "./Teste_malha_local" # Crie esta pasta para seus novos resultados
PAT_MALHA_LOCAL = ["*.s4p", "*.ts"]
PARAM_MALHA_LOCAL = "mesh_pilar_nm" # Nome do parâmetro que você varre no CST (e.g., 'mesh_pilar')
OUT_MALHA_LOCAL = "out_analise_malha_local"

# --- Configurações Gerais ---
FMIN_THz, FMAX_THz, NSAMPLES = 250.0, 330.0, 401
TOL_ABS = 0.01   # Tolerância para gráficos de erro
SAVE_FIGS = True
PLOT_STYLE = 'seaborn-v0_8-darkgrid' # Estilo visual dos gráficos

# ========================= IMPORTS =========================
import os
import glob
import re
import csv
import math
import cmath
import warnings
import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
    USE_PANDAS = True
except ImportError:
    USE_PANDAS = False

# ========================= FUNÇÕES AUXILIARES =========================
def listar_arquivos_multi(dirpath, patterns):
    """Lista arquivos em um diretório que correspondem a múltiplos padrões."""
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(dirpath, pat)))
    return sorted(set(files))

def ler_touchstone_generico(path):
    """Lê um arquivo Touchstone (v1 ou v2) de forma robusta."""
    def _try_read_text(filepath):
        encs = ["utf-8-sig", "utf-8", "utf-16", "latin-1"]
        for enc in encs:
            try:
                with open(filepath, "r", encoding=enc, errors="strict") as f:
                    return f.readlines()
            except Exception:
                pass
        raise IOError(f"Não foi possível ler o arquivo {filepath} com os encodings testados.")

    def _normalize_line(ln):
        return ln.replace("\u2212", "-").replace(",", ".").strip()

    def _unidade_para_THz(token):
        return {"HZ": 1e-12, "KHZ": 1e-9, "MHZ": 1e-6, "GHZ": 1e-3, "THZ": 1.0}.get(token.strip().upper())

    def _complex_from_pair(a, b, fmt):
        if fmt == "RI": return complex(a, b)
        if fmt == "MA": return a * cmath.exp(1j * math.radians(b))
        if fmt == "DB": return (10.0**(a / 20.0)) * cmath.exp(1j * math.radians(b))
        raise ValueError(f"Formato {fmt} não suportado.")

    raw_lines = _try_read_text(path)
    lines = [_normalize_line(ln) for ln in raw_lines if ln.strip()]
    
    params = {}
    param_regex = re.compile(r"^!\s*Parameters\s*=\s*\{(.+)\}")
    kv_regex = re.compile(r"([A-Za-z0-9_]+)\s*=\s*([^;]+)")
    for line in lines:
        match = param_regex.match(line)
        if match:
            for k, v in kv_regex.findall(match.group(1)):
                try:
                    params[k.strip()] = float(v.strip())
                except ValueError:
                    params[k.strip()] = v.strip()
            break
            
    header_regex = re.compile(r"^#\s+(\w+)\s+S\s+(\w+)\s+R\s+([0-9eE\.\+\-]+)", re.IGNORECASE)
    unit, fmt, z_default = None, None, 50.0
    for line in lines:
        if line.startswith("#"):
            match = header_regex.match(line)
            if match:
                unit, fmt, z_str = match.groups()
                z_default = float(z_str)
                break
    if not unit: raise ValueError("Cabeçalho '#' do Touchstone não encontrado.")
    
    scale = _unidade_para_THz(unit)
    if scale is None: raise ValueError(f"Unidade de frequência não suportada: {unit}")

    # --- CORREÇÃO AQUI ---
    # Adicionada a condição "and not line.strip().startswith('[')" para ignorar
    # linhas de seção do Touchstone v2.0, como [Version] e [Network Data].
    data_lines = [
        line.split("!")[0] for line in lines 
        if not line.startswith("!") and not line.startswith("#") and not line.strip().startswith("[")
    ]
    
    freqs, s_mats = [], []
    for line in data_lines:
        # Adiciona uma verificação para garantir que a linha não está vazia após remover comentários
        if not line.strip():
            continue
        tokens = [float(t) for t in line.split()]
        if not tokens: continue
        # Assumindo 4 portas, 1 (freq) + 4*4*2 (dados) = 33 valores por ponto.
        freqs.append(tokens[0] * scale)
        cplx_data = [_complex_from_pair(tokens[i], tokens[i+1], fmt) for i in range(1, len(tokens), 2)]
        
        # Garante que temos dados suficientes para remodelar a matriz S
        num_ports = int(np.sqrt(len(cplx_data)))
        if num_ports * num_ports != len(cplx_data):
            warnings.warn(f"Número inesperado de pontos de dados na linha em {path}. Ignorando linha.")
            freqs.pop() # Remove a frequência adicionada
            continue

        s_mats.append(np.array(cplx_data, dtype=complex).reshape(num_ports, num_ports))

    if not freqs: raise ValueError("Nenhum dado de rede encontrado no arquivo.")
    
    freqs = np.array(freqs)
    s_mats = np.stack(s_mats, axis=0)
    
    sort_idx = np.argsort(freqs)
    # Assume 4 portas se o número de portas não puder ser determinado a partir dos dados
    num_ports_final = s_mats.shape[1] if s_mats.ndim > 1 else 4
    return freqs[sort_idx], s_mats[sort_idx], np.array([z_default] * num_ports_final), params

def extrair_Tij(S_array):
    """Extrai as componentes da matriz de transmissão do array S."""
    if S_array.shape[1] < 4 or S_array.shape[2] < 4:
         warnings.warn(f"Matriz S tem formato inesperado {S_array.shape}, retornando zeros para T.")
         return {"Txx": 0, "Tyy": 0, "Txy": 0, "Tyx": 0}

    # Mapeamento CST (4-port Floquet): 0,1=Zmin(TE/TM); 2,3=Zmax(TE/TM)
    # Txx: Zmax(x) <- Zmin(x) => S[2,0]
    # Tyy: Zmax(y) <- Zmin(y) => S[3,1]
    # Txy: Zmax(x) <- Zmin(y) => S[2,1]
    # Tyx: Zmax(y) <- Zmin(x) => S[3,0]
    return {
        "Txx": S_array[:, 2, 0], "Tyy": S_array[:, 3, 1],
        "Txy": S_array[:, 2, 1], "Tyx": S_array[:, 3, 0]
    }

def grade_comum(freqs_list, fmin_req, fmax_req, nsamples):
    """Cria uma grade de frequência comum para interpolação."""
    valid_freqs = [f for f in freqs_list if len(f) > 1]
    if not valid_freqs:
        raise ValueError("Nenhum array de frequência válido para criar grade comum.")
    f_start = max(fmin_req, max(np.min(f) for f in valid_freqs))
    f_end = min(fmax_req, min(np.max(f) for f in valid_freqs))
    if f_end <= f_start:
        raise ValueError(f"Interseção de banda vazia: [{f_start:.2f}, {f_end:.2f}] THz")
    return np.linspace(f_start, f_end, nsamples)

def interp_complex(x_src, y_src, x_dst):
    """Interpola um array complexo."""
    y_real = np.interp(x_dst, x_src, np.real(y_src))
    y_imag = np.interp(x_dst, x_src, np.imag(y_src))
    return y_real + 1j * y_imag

def encontrar_ressonancia(freqs, T_mag):
    """Encontra a frequência e a amplitude do pico de ressonância."""
    if len(T_mag) == 0:
        return {'f_res': np.nan, 'T_max': np.nan}
    idx_max = np.argmax(T_mag)
    f_res = freqs[idx_max]
    T_max = T_mag[idx_max]
    return {'f_res': f_res, 'T_max': T_max}

def _inferir_param_varrido(basename, params, param_name):
    """Tenta inferir o valor de um parâmetro varrido a partir do header ou nome do arquivo."""
    # 1. Tenta ler do header de parâmetros do CST
    if param_name in params:
        try:
            return float(params[param_name])
        except (ValueError, TypeError):
            pass
    # 2. Tenta extrair do nome do arquivo (e.g., "run_mesh_pilar_15.s4p", "sim_mesh_pilar=15.ts")
    match = re.search(fr"{param_name}[_=]?(\d+\.?\d*)", basename, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, TypeError):
            pass
    return np.nan

# ========================= FUNÇÃO ADICIONADA =========================
def interpolar_todos(runs, f_grid):
    """
    Interpola os dados de transmissão de todas as simulações para uma grade de frequência comum.
    """
    curvas_interp = {}
    for run in runs:
        rotulo = run["rotulo"]
        freqs_orig = run["freqs"]
        
        # Extrai os parâmetros T da matriz S
        params_T = extrair_Tij(run["s_mats"])
        
        # Interpola cada parâmetro T para a grade comum
        curvas_interp[rotulo] = {
            key: interp_complex(freqs_orig, val_cplx, f_grid)
            for key, val_cplx in params_T.items()
        }
    return curvas_interp

# ========================= FUNÇÕES DE PLOTAGEM E SALVAMENTO =========================
def salvar_fig(fig, path):
    """Salva a figura atual e a fecha."""
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def plotar_stack(freqs, curvas, titulo, ylabel, out_path):
    """Plota múltiplas curvas (e.g., |Txx|) sobrepostas."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for rotulo, y_data in curvas.items():
        ax.plot(freqs, np.abs(y_data), lw=1.5, label=rotulo)
    ax.set_xlabel("Frequência (THz)")
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    ax.grid(True, ls=":")
    ax.legend(ncol=2, fontsize=9)
    salvar_fig(fig, out_path)

def plotar_diff(freqs, diffs, tol, titulo, out_path):
    """Plota as diferenças absolutas em relação a uma baseline."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for rotulo, y_data in diffs.items():
        ax.plot(freqs, y_data, lw=1.5, label=rotulo)
    ax.axhline(tol, color="k", ls="--", lw=1.2, label=f"Tolerância = {tol:g}")
    ax.set_xlabel("Frequência (THz)")
    ax.set_ylabel("|Δ| no Módulo")
    ax.set_title(titulo)
    ax.grid(True, ls=":")
    ax.legend(ncol=2, fontsize=9)
    salvar_fig(fig, out_path)

def plotar_convergencia_ressonancia(param_vals, res_data, param_name, out_dir):
    """Plota a convergência dos parâmetros da ressonância (f_res, T_max)."""
    # Ordena os dados para o plot, garantindo que as linhas se conectem corretamente
    sorted_indices = np.argsort(param_vals)
    param_vals = np.array(param_vals)[sorted_indices]
    res_data = np.array(res_data)[sorted_indices]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    f_res_xx = [d['f_res'] for d in res_data]
    T_max_xx = [d['T_max'] for d in res_data]
    f_res_yy = [d['f_res_yy'] for d in res_data]
    T_max_yy = [d['T_max_yy'] for d in res_data]

    # Gráfico 1: Frequência de Ressonância
    ax1.plot(param_vals, f_res_xx, 'o-', label='$T_{xx}$', color='C0')
    ax1.plot(param_vals, f_res_yy, 's--', label='$T_{yy}$', color='C1')
    ax1.set_ylabel("Frequência de Ressonância (THz)")
    ax1.set_title(f"Convergência dos Parâmetros de Ressonância vs. {param_name}")
    ax1.grid(True, ls=":")
    ax1.legend()

    # Gráfico 2: Amplitude Máxima
    ax2.plot(param_vals, T_max_xx, 'o-', label='$T_{xx}$', color='C0')
    ax2.plot(param_vals, T_max_yy, 's--', label='$T_{yy}$', color='C1')
    ax2.set_xlabel(f"Parâmetro de Malha ({param_name})")
    ax2.set_ylabel("Amplitude Máxima $|T|$")
    ax2.grid(True, ls=":")
    ax2.legend()
    
    # Inverter eixo x, pois menor `mesh_pilar` é mais refinado
    ax2.invert_xaxis()
    
    salvar_fig(fig, os.path.join(out_dir, "convergencia_parametros_ressonancia.png"))

# ========================= FLUXOS DE ANÁLISE =========================

def fluxo_convergencia_malha_local():
    """Executa a análise de convergência para a malha local do pilar (Passo C)."""
    if not os.path.isdir(DIR_MALHA_LOCAL):
        print(f"[AVISO] Pasta de malha local não encontrada: {DIR_MALHA_LOCAL}")
        return
    print("\n=== Passo C: Análise de Convergência da Malha Local ===")
    os.makedirs(OUT_MALHA_LOCAL, exist_ok=True)

    def label_fn(base, params):
        val = _inferir_param_varrido(base, params, PARAM_MALHA_LOCAL)
        return {PARAM_MALHA_LOCAL: val, "rotulo": f"{PARAM_MALHA_LOCAL}={val or 'N/A'}"}

    # 1. Listar todos os arquivos que correspondem aos padrões.
    arquivos = listar_arquivos_multi(DIR_MALHA_LOCAL, PAT_MALHA_LOCAL)
    
    # 2. Ler e processar cada arquivo para criar a lista de "runs".
    runs_brutos = []
    for fpath in arquivos:
        try:
            freqs, s_mats, _, params = ler_touchstone_generico(fpath)
            base = os.path.basename(fpath)
            
            run_data = {
                "path": fpath,
                "base": base,
                "freqs": freqs,
                "s_mats": s_mats,
                "params": params,
            }
            # Adiciona o valor do parâmetro varrido e um rótulo ao dicionário.
            run_data.update(label_fn(base, params))
            runs_brutos.append(run_data)
        except Exception as e:
            print(f"[AVISO] Falha ao processar o arquivo {os.path.basename(fpath)}: {e}")
            
    # 3. Filtrar runs que não têm o parâmetro de malha válido e ordenar pelo valor do parâmetro.
    #    A malha mais fina (menor valor) será a primeira da lista.
    runs = sorted(
        [r for r in runs_brutos if np.isfinite(r.get(PARAM_MALHA_LOCAL, np.nan))],
        key=lambda r: r.get(PARAM_MALHA_LOCAL, np.inf)
    )
    
    if not runs:
        print(f"[ERRO] Nenhum arquivo de simulação válido encontrado em '{DIR_MALHA_LOCAL}' com o parâmetro '{PARAM_MALHA_LOCAL}'.")
        return

    f_grid = grade_comum([r["freqs"] for r in runs], FMIN_THz, FMAX_THz, NSAMPLES)
    curvas_interp = interpolar_todos(runs, f_grid)

    # Baseline é a malha mais fina (menor valor de mesh_pilar)
    baseline_run = runs[0]
    rotulo_baseline = baseline_run["rotulo"]
    curva_baseline = curvas_interp[rotulo_baseline]
    print(f"Baseline para convergência de malha: {rotulo_baseline}")

    linhas_relatorio, diffs_Txx, diffs_Tyy = [], {}, {}
    param_vals, res_data = [], []

    for run in runs:
        rotulo = run["rotulo"]
        curva_atual = curvas_interp[rotulo]

        # Calcular métricas de erro
        diff_xx = np.abs(np.abs(curva_atual["Txx"]) - np.abs(curva_baseline["Txx"]))
        diff_yy = np.abs(np.abs(curva_atual["Tyy"]) - np.abs(curva_baseline["Tyy"]))
        e_max_xx = np.max(diff_xx)
        e_max_yy = np.max(diff_yy)
        
        if rotulo != rotulo_baseline:
            diffs_Txx[rotulo] = diff_xx
            diffs_Tyy[rotulo] = diff_yy

        # Calcular parâmetros de ressonância
        res_xx = encontrar_ressonancia(f_grid, np.abs(curva_atual["Txx"]))
        res_yy = encontrar_ressonancia(f_grid, np.abs(curva_atual["Tyy"]))

        param_val = run[PARAM_MALHA_LOCAL]
        param_vals.append(param_val)
        
        # Junta os resultados de ressonância de xx e yy para o plot
        dados_ressonancia = {
            'f_res': res_xx['f_res'], 
            'T_max': res_xx['T_max'],
            'f_res_yy': res_yy['f_res'],
            'T_max_yy': res_yy['T_max'],
        }
        res_data.append(dados_ressonancia)

        linhas_relatorio.append({
            "arquivo": run["base"],
            PARAM_MALHA_LOCAL: param_val,
            "max_err_Txx": e_max_xx,
            "max_err_Tyy": e_max_yy,
            "f_res_xx": res_xx['f_res'],
            "T_max_xx": res_xx['T_max'],
            "f_res_yy": res_yy['f_res'],
            "T_max_yy": res_yy['T_max'],
        })

    # Salvar relatórios
    if USE_PANDAS and linhas_relatorio:
        df = pd.DataFrame(linhas_relatorio)
        df = df.sort_values(by=PARAM_MALHA_LOCAL)
        df.to_csv(
            os.path.join(OUT_MALHA_LOCAL, "relatorio_convergencia_malha.csv"), index=False
        )
        print(f"\nRelatório de convergência salvo em '{OUT_MALHA_LOCAL}/relatorio_convergencia_malha.csv'")
        print(df.to_string())


    # Salvar gráficos
    if SAVE_FIGS:
        # Stacks de transmissão
        plotar_stack(f_grid, {r:d["Txx"] for r,d in curvas_interp.items()}, "|Txx| vs. Malha Local", "|Txx|", os.path.join(OUT_MALHA_LOCAL, "stack_Txx.png"))
        plotar_stack(f_grid, {r:d["Tyy"] for r,d in curvas_interp.items()}, "|Tyy| vs. Malha Local", "|Tyy|", os.path.join(OUT_MALHA_LOCAL, "stack_Tyy.png"))
        plotar_stack(f_grid, {r:d["Txy"] for r,d in curvas_interp.items()}, "|Txy| (Cross-Pol) vs. Malha Local", "|Txy|", os.path.join(OUT_MALHA_LOCAL, "stack_Txy.png"))
        plotar_stack(f_grid, {r:d["Tyx"] for r,d in curvas_interp.items()}, "|Tyx| (Cross-Pol) vs. Malha Local", "|Tyx|", os.path.join(OUT_MALHA_LOCAL, "stack_Tyx.png"))
        
        # Diferenças
        plotar_diff(f_grid, diffs_Txx, TOL_ABS, f"Erro em |Txx| vs. Baseline ({rotulo_baseline})", os.path.join(OUT_MALHA_LOCAL, "diff_Txx.png"))
        plotar_diff(f_grid, diffs_Tyy, TOL_ABS, f"Erro em |Tyy| vs. Baseline ({rotulo_baseline})", os.path.join(OUT_MALHA_LOCAL, "diff_Tyy.png"))
        
        # Convergência da Ressonância
        plotar_convergencia_ressonancia(param_vals, res_data, PARAM_MALHA_LOCAL, OUT_MALHA_LOCAL)
        print(f"Gráficos de análise salvos na pasta '{OUT_MALHA_LOCAL}'")

def main():
    """Função principal que executa os fluxos de análise."""
    plt.style.use(PLOT_STYLE)
    
    # Descomente a linha abaixo para rodar a análise de folgas
    # fluxo_folgas()
    
    # Executa a análise de convergência da malha local
    fluxo_convergencia_malha_local()

if __name__ == "__main__":
    main()