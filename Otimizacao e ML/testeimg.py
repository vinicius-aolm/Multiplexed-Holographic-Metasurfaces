import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_final_diagram_v2():
    # Configuração da figura
    fig, ax = plt.subplots(figsize=(6, 10), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')

    # Cores (Fidelidade à referência)
    color_si_body = '#6495ED'    # Azul Cornflower
    color_si_top = '#87CEFA'     # Azul Claro
    color_sub_top = '#E0E0E0'    # Cinza Claro
    color_sub_side = '#BDBDBD'   # Cinza Médio
    color_arrow = '#8ab6d6'      # Azul da seta grande

    # ==========================================
    # 1. CÉLULA UNITÁRIA (TOPO)
    # ==========================================
    base_x, base_y = 5.0, 11.5
    
    # --- Substrato ---
    w_sub, h_sub = 2.8, 1.2
    # Face Superior
    path_sub_top = [
        (base_x, base_y - h_sub), (base_x + w_sub, base_y),
        (base_x, base_y + h_sub), (base_x - w_sub, base_y)
    ]
    ax.add_patch(patches.Polygon(path_sub_top, facecolor=color_sub_top, edgecolor='gray', lw=0.5))
    
    # Laterais
    thick = 0.25
    path_sub_left = [(base_x - w_sub, base_y), (base_x, base_y - h_sub), (base_x, base_y - h_sub - thick), (base_x - w_sub, base_y - thick)]
    path_sub_right = [(base_x, base_y - h_sub), (base_x + w_sub, base_y), (base_x + w_sub, base_y - thick), (base_x, base_y - h_sub - thick)]
    ax.add_patch(patches.Polygon(path_sub_left, facecolor=color_sub_side, edgecolor='gray', lw=0.5))
    ax.add_patch(patches.Polygon(path_sub_right, facecolor=color_sub_side, edgecolor='gray', lw=0.5))

    ax.text(base_x + w_sub + 0.1, base_y - 0.5, r'SiO$_2$', fontsize=12, color='#444')

    # --- Pilar ---
    pw = 1.2  # Largura do pilar
    ph = 3.2  # Altura visual
    
    # Base (Elipse inferior)
    ell_base = patches.Ellipse((base_x, base_y), pw, 0.5, facecolor=color_si_body)
    ax.add_patch(ell_base)
    
    # Corpo (Retângulo)
    rect = patches.Rectangle((base_x - pw/2, base_y), pw, ph, facecolor=color_si_body)
    ax.add_patch(rect)
    
    # Linhas laterais
    ax.plot([base_x - pw/2, base_x - pw/2], [base_y, base_y + ph], color='gray', lw=0.5)
    ax.plot([base_x + pw/2, base_x + pw/2], [base_y, base_y + ph], color='gray', lw=0.5)

    # Topo (Elipse Superior)
    ell_top = patches.Ellipse((base_x, base_y + ph), pw, 0.5, facecolor=color_si_top, edgecolor='gray', lw=0.5)
    ax.add_patch(ell_top)

    # Label Si
    ax.text(base_x + pw/2 + 0.2, base_y + ph/2, 'Si', fontsize=12, color='#333')

    # --- COTAS ---
    
    # H (Altura)
    ax.annotate('', xy=(base_x - pw/2 - 0.3, base_y), xytext=(base_x - pw/2 - 0.3, base_y + ph),
                arrowprops=dict(arrowstyle='<->', color='black', lw=0.8))
    ax.text(base_x - pw/2 - 0.6, base_y + ph/2, r'$H$', va='center', ha='right', fontsize=11)

    # P (Periodo)
    p_start = (base_x - w_sub, base_y - thick)
    p_end = (base_x, base_y - h_sub - thick)
    ax.annotate('', xy=(p_start[0]-0.1, p_start[1]-0.1), xytext=(p_end[0]-0.1, p_end[1]-0.1),
                arrowprops=dict(arrowstyle='<->', color='black', lw=0.8))
    ax.text(base_x - w_sub/2 - 0.5, base_y - h_sub/2 - 0.6, r'$P$', ha='right', fontsize=11)

    # Lx e Ly (NOVA LÓGICA: Cruz no topo)
    cy_top = base_y + ph
    
    # Seta Lx (Horizontal completa)
    ax.annotate('', xy=(base_x - pw/2 + 0.1, cy_top), xytext=(base_x + pw/2 - 0.1, cy_top),
                arrowprops=dict(arrowstyle='<->', color='black', lw=0.8))
    ax.text(base_x, cy_top + 0.05, r'$L_x$', ha='center', va='bottom', fontsize=10) # Texto acima da seta

    # Seta Ly (Vertical visual completa)
    # Como a elipse tem altura 0.5, o raio vertical é 0.25
    ax.annotate('', xy=(base_x, cy_top - 0.2), xytext=(base_x, cy_top + 0.2),
                arrowprops=dict(arrowstyle='<->', color='black', lw=0.8))
    ax.text(base_x + 0.1, cy_top, r'$L_y$', ha='left', va='center', fontsize=10) # Texto ao lado

    # Título
    ax.text(5, 15.5, "Processo de Casamento de Fase", ha='center', fontsize=14, fontweight='bold')

    # ==========================================
    # 2. SETA E EQUAÇÃO
    # ==========================================
    
    # Seta Curva
    arrow = patches.FancyArrowPatch((2.0, 10.0), (2.0, 5.5), 
                                    connectionstyle="arc3,rad=-0.2", 
                                    arrowstyle="Simple, tail_width=5, head_width=15, head_length=15",
                                    color=color_arrow)
    ax.add_patch(arrow)

    # Equação (Mais perto da seta)
    eq = r"$\min \left[ (\hat{E}_{xx} - e^{i\phi_{xx}}) \right.$" + "\n" + r"$ \left. + (\hat{E}_{yy} - e^{i\phi_{yy}}) \right]$"
    ax.text(3.0, 7.5, eq, fontsize=14, va='center')

    # ==========================================
    # 3. LAYOUT OTIMIZADO (BASE)
    # ==========================================
    
    # Fundo Grid
    g_x, g_y = 3.0, 0.5
    g_w = 4.0
    rect_grid = patches.Rectangle((g_x, g_y), g_w, g_w, facecolor='#f0f4fa')
    ax.add_patch(rect_grid)

    # Grid de elipses variadas
    rows, cols = 4, 4
    step = g_w / 4
    
    for i in range(rows):
        for j in range(cols):
            cx = g_x + j*step + step/2
            cy = g_y + (rows-1-i)*step + step/2
            
            # Simular variações de forma (Anisotropia)
            # Lx e Ly variam independentemente
            rx = 0.12 + 0.08 * np.abs(np.sin(i*2 + j))
            ry = 0.12 + 0.08 * np.cos(j*1.5)
            
            ell = patches.Ellipse((cx, cy), rx*2, ry*2, facecolor='#4169E1')
            ax.add_patch(ell)
            
            # Brilho centro
            ell_i = patches.Ellipse((cx, cy), rx*0.6, ry*0.6, facecolor='#87CEFA')
            ax.add_patch(ell_i)

    # Reticências
    ax.text(g_x + g_w + 0.2, g_y + g_w/2, r'$\dots$', fontsize=16, color='gray')

    # Texto Inferior
    ax.text(5, -0.5, "Layout Otimizado", ha='center', fontsize=12, fontweight='bold')

    # Salvar
    plt.tight_layout()
    plt.savefig('diagrama_final_v2.png', bbox_inches='tight', pad_inches=0.1)
    plt.savefig('diagrama_final_v2.svg', bbox_inches='tight', pad_inches=0.1)
    plt.show()

if __name__ == "__main__":
    draw_final_diagram_v2()