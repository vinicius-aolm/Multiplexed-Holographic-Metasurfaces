import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd
import skrf as rf
import numpy as np
import re




def find_file(fold): #Função que encontra os arquivos e armazena em um DF
    data = []
    print("Buscando...")

    if not os.path.isdir(fold):
        print("Pasta não encontrada")
        return None

    for name_arch in os.listdir(fold):
        path_ = os.path.join(fold, name_arch)

        if os.path.isfile(path_) and name_arch.lower().startswith('2_teste_nanopilar_'):
            try:
                match = re.search(r'(\d+)', os.path.splitext(name_arch)[0])
                if not match:
                    print(f"  Aviso: Não foi possível extrair o ID do arquivo '{name_arch}'. Pulando.")
                    continue
                
                id_nanopilar = int(match.group(1))
                print(f"Lendo arquivo: {name_arch} (ID:{id_nanopilar})...")
                
                network = rf.Network(path_) #Scikit-RF basicamente entra no arq e entende os dados e as formatações. Também converte valores para complexos
                
                for i, freq in enumerate(network.f): # ".f lista todas as frequencias encontradas em network"
                    data_line = {'arquivo': name_arch, 'id_nanopilar': id_nanopilar, 'frequencia_hz': freq, 'frequencia_ghz': freq / 1e9}
                    
                    # --- Lógica de Extração de Parâmetros S (Apenas Real e Imaginário) ---
                    
                    if network.nports == 2:
                        s11 = network.s[i, 0, 0]
                        s21 = network.s[i, 1, 0]
                        s12 = network.s[i, 0, 1]
                        s22 = network.s[i, 1, 1]

                        data_line['S11_real'] = s11.real
                        data_line['S11_imag'] = s11.imag
                        data_line['S21_real'] = s21.real
                        data_line['S21_imag'] = s21.imag
                        data_line['S12_real'] = s12.real
                        data_line['S12_imag'] = s12.imag
                        data_line['S22_real'] = s22.real
                        data_line['S22_imag'] = s22.imag

                    elif network.nports == 4:
                        for out_port in range(4):
                            for in_port in range(4):
                                param_name = f'S{out_port + 1}{in_port + 1}'
                                s_param_complex = network.s[i, out_port, in_port]
                                
                                data_line[f'{param_name}_real'] = s_param_complex.real
                                data_line[f'{param_name}_imag'] = s_param_complex.imag

                    elif network.nports == 1:
                        s11 = network.s[i, 0, 0]
                        data_line['S11_real'] = s11.real
                        data_line['S11_imag'] = s11.imag

                    data.append(data_line)

            except Exception as e:
                print(f"  Erro ao processar o arquivo {name_arch}: {e}")

    if not data:
        print("\nNenhum arquivo touchstone válido foi encontrado.")
        return None
        
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__": #restrição de print apenas para runs locais
    
    fold_path = r"C:\Users\humberto25043\OneDrive - ILUM ESCOLA DE CIÊNCIA\Área de Trabalho\Machine learning\Meta-atoms lib\Bibliotecas\Teste_biblioteca_1"

    # Chama a função para ler os arquivos e criar o DataFrame
    dataframe_bruto = find_file(fold_path)

    # Verifica se o DataFrame foi criado com sucesso antes de usá-lo
    if dataframe_bruto is not None:
        print("\n--- DataFrame de dados brutos criado com sucesso! ---")
        
        print("\n5 primeiras linhas do DataFrame:")
        print(dataframe_bruto.head())
        
        print("\nInformações e colunas do DataFrame:")
        dataframe_bruto.info()
        
        print("\nExemplo de filtro por id_nanopilar = 5:")
        if 5 in dataframe_bruto['id_nanopilar'].unique():
            print(dataframe_bruto[dataframe_bruto['id_nanopilar'] == 5].head())
        else:
            print("Não foram encontrados dados para o id_nanopilar = 5.")