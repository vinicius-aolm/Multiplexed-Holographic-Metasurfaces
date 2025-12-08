import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import fft2, fftshift

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
OUTPUT_FILE = "GS_Y_Step2_Tiling_Final.gif"
N_SUPER = 60
MAX_TILES = 5
FPS = 2

# ==========================================
# 2. DADOS FÍSICOS (SIMULAÇÃO)
# ==========================================
x = np.linspace(-1, 1, N_SUPER)
X, Y = np.meshgrid(x, x)
target_disk = (np.sqrt(X**2 + Y**2) <= 0.4).astype(float)

# Otimização rápida (GS)
np.random.seed(10)
u = np.exp(1j * 2*np.pi * np.random.rand(N_SUPER, N_SUPER))
for _ in range(30):
    U_k = target_disk * np.exp(1j * np.angle(fftshift(fft2(u))))
    u = np.exp(1j * np.angle(np.fft.ifft2(np.fft.ifftshift(U_k))))

phase_supercell = np.angle(u)

# ==========================================
# 3. VISUALIZAÇÃO (CORRIGIDA)
# ==========================================
plt.style.use('default')
fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor('white')

# Layout
ax_ref   = fig.add_axes([0.05, 0.35, 0.25, 0.5]) 
ax_meta  = fig.add_axes([0.35, 0.35, 0.25, 0.5]) 
ax_ff    = fig.add_axes([0.70, 0.35, 0.25, 0.5]) 

# Setas
ax_arrow1 = fig.add_axes([0.30, 0.55, 0.05, 0.1]) 
ax_arrow2 = fig.add_axes([0.60, 0.55, 0.10, 0.1]) 

# Títulos
ax_ref.set_title(r"Unidade Fundamental", color='purple', fontsize=12, pad=20, weight='bold')
ax_meta.set_title(r"Metassuperfície (Espaço Real)", color='blue', fontsize=12, pad=20, weight='bold')
ax_ff.set_title(r"Far-Field (Espaço-k)", color='black', fontsize=12, pad=20, weight='bold')

# Subtítulos
ax_ref.text(0.5, 1.02, "Supercélula (Fase)", ha='center', transform=ax_ref.transAxes, fontsize=10)
ax_meta.text(0.5, 1.02, "Processo de Tiling", ha='center', transform=ax_meta.transAxes, fontsize=10)
ax_ff.text(0.5, 1.02, "Difração (Intensidade)", ha='center', transform=ax_ff.transAxes, fontsize=10)

for ax in [ax_ref, ax_meta, ax_ff, ax_arrow1, ax_arrow2]: ax.axis('off')

# Inicializar
im_ref = ax_ref.imshow(phase_supercell, cmap='twilight', vmin=-np.pi, vmax=np.pi)
im_meta = ax_meta.imshow(phase_supercell, cmap='twilight', vmin=-np.pi, vmax=np.pi)
im_ff = ax_ff.imshow(np.zeros((N_SUPER, N_SUPER)), cmap='inferno')

# Setas
ax_arrow1.text(0.5, 0.5, r"$\times N$", ha='center', va='center', fontsize=20, weight='bold', color='gray')
ax_arrow2.text(0.5, 0.4, r"$\longrightarrow$", ha='center', va='center', fontsize=30, weight='bold')
ax_arrow2.text(0.5, 0.9, r"$\mathbf{FFT}$", ha='center', va='bottom', fontsize=12, weight='bold')

# Legenda Inferior
lbl_step = fig.text(0.5, 0.20, "", ha='center', fontsize=16, color='#0044cc', weight='bold')

# Equação de Rodapé (CORRIGIDA: Usando \Leftrightarrow)
eq_text = r"Conceito: Repetição Periódica $(x) \Leftrightarrow$ Amostragem Discreta $(k)$"
fig.text(0.5, 0.08, eq_text, ha='center', fontsize=14, color='gray', style='italic')

# HUD
lbl_status = fig.text(0.5, 0.14, "", ha='center', fontsize=14, color='black')

def update(n_tile):
    reps = n_tile + 1
    
    # 1. Tiling
    full_meta = np.tile(phase_supercell, (reps, reps))
    
    # 2. Far-Field
    ff_field = fftshift(fft2(np.exp(1j * full_meta)))
    ff_intensity = np.abs(ff_field)**2
    ff_view = np.log10(ff_intensity + 1e-10)
    
    # Atualizar
    im_meta.set_data(full_meta)
    im_ff.set_data(ff_view)
    im_ff.set_clim(np.max(ff_view)-4, np.max(ff_view))
    
    lbl_step.set_text(f"Passo {reps}: Matriz {reps}x{reps} Supercélulas")
    
    if reps == 1:
        lbl_status.set_text("Far-Field Contínuo (Apenas Envelope)")
    else:
        lbl_status.set_text(f"Far-Field Discreto ({reps}x{reps} Pontos)")

    return [im_meta, im_ff, lbl_step, lbl_status]

print(f"Gerando GIF Etapa 2 ({MAX_TILES} frames)...")
frames_seq = [0, 1, 2, 3, 4, 4, 4]
ani = animation.FuncAnimation(fig, update, frames=frames_seq, blit=False)
ani.save(OUTPUT_FILE, writer='pillow', fps=1.5)
print(f"Sucesso! Salvo: {OUTPUT_FILE}")