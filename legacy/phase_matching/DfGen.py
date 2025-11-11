import os
import re
import numpy as np
import pandas as pd
import skrf as rf

# Procura por "Parameters = {...}" em qualquer posição da linha
_PARAM_ANY_RE = re.compile(r"Parameters\s*=\s*\{(?P<body>.+?)\}", re.IGNORECASE)
_NUM_PORTS_RE = re.compile(r"^\[Number of Ports\]\s*(\d+)\s*$", re.IGNORECASE)

def parse_touchstone_params(path: str, max_header_lines: int = 200) -> dict:
    """
    Varre o início do arquivo (até max_header_lines ou até '[Network Data]')
    e busca uma linha que contenha 'Parameters = { ... }'. Mantém chaves originais.
    Aceita separadores ';' ou ',' entre pares K=V.
    """
    params = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, raw in enumerate(f):
                line = raw.strip()
                if i > max_header_lines:
                    break
                # Parar ao entrar claramente na seção de dados
                if line.startswith("[Network Data]"):
                    break
                # Tenta capturar o bloco dentro de { ... }
                m = _PARAM_ANY_RE.search(line)
                if not m:
                    continue
                body = m.group("body")

                # separadores: primeiro tenta ';', se quase não separar usa ','
                parts = [p.strip() for p in body.split(";") if p.strip()]
                if len(parts) <= 1:  # pode estar separado por vírgulas
                    parts = [p.strip() for p in body.split(",") if p.strip()]

                for pair in parts:
                    if "=" not in pair:
                        continue
                    k, v = pair.split("=", 1)
                    k = k.strip()                   # preserva nomes (ex.: 'L_x', 'L_y')
                    v = v.strip()
                    # remove possíveis comentários residuais depois do valor
                    v = re.split(r"\s*!|\s+#", v)[0].strip()
                    try:
                        v = float(v)
                    except Exception:
                        pass
                    params[k] = v
                # encontrou a linha de parâmetros; pode sair
                break
    except Exception:
        pass
    return params

def parse_number_of_ports_from_header(path: str) -> int | None:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = _NUM_PORTS_RE.match(line.strip())
                if m:
                    return int(m.group(1))
                if line.startswith("[Network Data]"):
                    break
    except Exception:
        pass
    return None

def find_file(fold: str) -> pd.DataFrame | None:
    """
    Lê arquivos .ts (não-recursivo) e gera DF com:
      - metadados: arquivo, caminho, id_nanopilar, frequencia_hz/ghz, nports
      - parâmetros do header (nomes originais, ex.: 'L_x', 'L_y', 'Lambda', ...)
      - Sij_real/Sij_imag conforme nports
    Garante que 'L_x' e 'L_y' existam como colunas (NaN se ausentes).
    """
    data = []
    print("Buscando...")

    if not os.path.isdir(fold):
        print("Pasta não encontrada:", fold)
        return None

    for name_arch in os.listdir(fold):
        path_ = os.path.join(fold, name_arch)
        root, ext = os.path.splitext(name_arch)
        if not os.path.isfile(path_) or ext.lower() != ".ts":
            continue

        m = re.search(r"(\d+)", root)
        id_nanopilar = int(m.group(1)) if m else -1

        print(f"Lendo arquivo: {name_arch} (ID:{id_nanopilar})...")

        # 1) parâmetros do header (robusto)
        params = parse_touchstone_params(path_)

        # 2) rede via scikit-rf
        try:
            network = rf.Network(path_)
            nports = int(network.nports)
        except Exception as e:
            print(f"  [WARN] skrf falhou em '{name_arch}': {e}")
            nports = parse_number_of_ports_from_header(path_) or 0
            if nports == 0:
                print("  [ERRO] não foi possível inferir nports. Pulando arquivo.")
                continue
            print("  [ERRO] sem leitura de S-params. Pulando.")
            continue

        # 3) linhas por frequência
        for i, f_hz in enumerate(network.f):
            row = {
                "arquivo": name_arch,
                "caminho": path_,
                "id_nanopilar": id_nanopilar,
                "frequencia_hz": float(f_hz),
                "frequencia_ghz": float(f_hz / 1e9),
                "nports": nports,
            }

            # injeta TODOS os params do header
            for k, v in params.items():
                row[k] = v

            # GARANTIR colunas L_x e L_y (mesmo que NaN)
            row["L_x"] = params.get("L_x", np.nan)
            row["L_y"] = params.get("L_y", np.nan)
            row["H"] = params.get("H", np.nan)

            # S-params
            if nports == 1:
                s11 = network.s[i, 0, 0]
                row["S11_real"] = float(np.real(s11)); row["S11_imag"] = float(np.imag(s11))

            elif nports == 2:
                s11 = network.s[i, 0, 0]; s21 = network.s[i, 1, 0]
                s12 = network.s[i, 0, 1]; s22 = network.s[i, 1, 1]
                row["S11_real"] = float(np.real(s11)); row["S11_imag"] = float(np.imag(s11))
                row["S21_real"] = float(np.real(s21)); row["S21_imag"] = float(np.imag(s21))
                row["S12_real"] = float(np.real(s12)); row["S12_imag"] = float(np.imag(s12))
                row["S22_real"] = float(np.real(s22)); row["S22_imag"] = float(np.imag(s22))

            elif nports == 4:
                for op in range(4):
                    for ip in range(4):
                        s = network.s[i, op, ip]
                        pname = f"S{op+1}{ip+1}"
                        row[f"{pname}_real"] = float(np.real(s))
                        row[f"{pname}_imag"] = float(np.imag(s))

            data.append(row)

    if not data:
        print("\nNenhum arquivo Touchstone válido foi encontrado.")
        return None

    df = pd.DataFrame(data)

    # ordena por id/frequência se existirem
    sort_cols = [c for c in ["id_nanopilar", "frequencia_hz"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ignore_index=True)

    return df

if __name__ == "__main__":
    fold_path = r"C:\Users\caval_5gfl5hy\OneDrive\Área de Trabalho\Altura_Varia"
    df_out = find_file(fold_path)
    if df_out is not None:
        print("\n--- DataFrame criado com sucesso! ---")
        print(df_out.head())
        print(df_out.info())

        out_csv = os.path.join(fold_path, "biblioteca_Bib1-27x27-perdas.csv")
        out_parquet = os.path.join(fold_path, "biblioteca_Bib1-27x27-perdas.parquet")

        try:
            df_out.to_csv(out_csv, index=False)
            print(f"\nCSV salvo em: {out_csv}")
        except Exception as e:
            print(f"\nFalha ao salvar CSV: {e}")

        try:
            df_out.to_parquet(out_parquet, index=False)
            print(f"(Opcional) Parquet salvo em: {out_parquet}")
        except Exception as e:
            print(f"(Opcional) Falha ao salvar Parquet: {e}")
