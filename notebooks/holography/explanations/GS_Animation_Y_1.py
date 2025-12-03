import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import fft2, fftshift, ifft2, ifftshift

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
OUTPUT_FILE = "GS_Y_Step1_Concept_Update.gif"
N = 90
FPS = 8

# Roteiro de Tempo
FRAMES_REAL = 15       
FRAMES_FFT = 15        
FRAMES_K_MESSY = 15    
FRAMES_CONSTRAINT = 20 
FRAMES_IFFT = 15       
FRAMES_UPDATE = 15     # NOVA CENA: Mostrar a fase nova

TOTAL_FRAMES = FRAMES_REAL + FRAMES_FFT + FRAMES_K_MESSY + FRAMES_CONSTRAINT + FRAMES_IFFT + FRAMES_UPDATE

# ==========================================
# 2. DADOS FÍSICOS (SIMULAÇÃO 1 ITERAÇÃO)
# ==========================================
# Alvo (Disco)
x = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, x)
target_disk = (np.sqrt(X**2 + Y**2) <= 0.4).astype(float)

# Estado Inicial
np.random.seed(10)
phase_init = np.random.rand(N, N) * 2 * np.pi - np.pi
U_real = np.exp(1j * phase_init)

# 1. FFT
U_k = fftshift(fft2(U_real))
amp_k_messy = np.abs(U_k)
amp_k_messy /= (np.max(amp_k_messy) + 1e-10)

# 2. Restrição e iFFT (Para obter a Fase Nova)
U_k_constrained = target_disk * np.exp(1j * np.angle(U_k))
U_real_new = ifft2(ifftshift(U_k_constrained))
phase_new = np.angle(U_real_new) # Esta é a fase atualizada

# ==========================================
# 3. VISUALIZAÇÃO
# ==========================================
plt.style.use('default')
fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor('white')

# Layout
ax_real = fig.add_axes([0.1, 0.35, 0.3, 0.5]) 
ax_k    = fig.add_axes([0.6, 0.35, 0.3, 0.5]) 
ax_arrow = fig.add_axes([0.42, 0.55, 0.16, 0.1]) 

# Títulos
ax_real.set_title(r"Espaço Real $(x, y)$", fontsize=14, pad=15, weight='bold')
ax_real.text(0.5, 1.02, "Supercélula (Fase)", ha='center', transform=ax_real.transAxes, color='#cc00cc', fontsize=11)

ax_k.set_title(r"Espaço Recíproco $(k_x, k_y)$", fontsize=14, pad=15, weight='bold')
ax_k.text(0.5, 1.02, "Far-Field (Amplitude)", ha='center', transform=ax_k.transAxes, color='black', fontsize=11)

for ax in [ax_real, ax_k, ax_arrow]: ax.axis('off')

# Inicializar
im_real = ax_real.imshow(phase_init, cmap='twilight', vmin=-np.pi, vmax=np.pi)
im_k    = ax_k.imshow(np.zeros((N, N)), cmap='inferno', vmin=0, vmax=1)

# Elementos
lbl_arrow_sym = ax_arrow.text(0.5, 0.4, "", ha='center', va='center', fontsize=40, weight='bold')
lbl_arrow_math = ax_arrow.text(0.5, 0.9, "", ha='center', va='bottom', fontsize=14, weight='bold')
lbl_step = fig.text(0.5, 0.15, "", ha='center', fontsize=16, color='#0044cc', weight='bold')
lbl_eq = ax_k.text(0.5, -0.2, "", ha='center', transform=ax_k.transAxes, fontsize=14, color='red', weight='bold')

fig.text(0.5, 0.05, r"Polarização Y: Otimização no Domínio da Frequência ($\mathcal{F}$)", ha='center', fontsize=13, color='gray')

def update(frame):
    # CENA 1: INICIAL
    if frame < FRAMES_REAL:
        im_real.set_data(phase_init)
        im_k.set_visible(False)
        lbl_step.set_text("1. Fase Inicial da Supercélula (Aleatória)")
        lbl_arrow_sym.set_text("")
        lbl_arrow_math.set_text("")
        lbl_eq.set_text("")

    # CENA 2: FFT
    elif frame < (FRAMES_REAL + FRAMES_FFT):
        im_k.set_visible(False)
        lbl_arrow_sym.set_text("➡")
        lbl_arrow_math.set_text(r"$\mathbf{FFT}$")
        lbl_step.set_text("2. Transformada de Fourier (Para Espaço-k)")

    # CENA 3: K-SPACE SUJO
    elif frame < (FRAMES_REAL + FRAMES_FFT + FRAMES_K_MESSY):
        im_k.set_visible(True)
        im_k.set_data(amp_k_messy)
        lbl_arrow_sym.set_text("")
        lbl_arrow_math.set_text("")
        lbl_step.set_text("3. Distribuição Angular Atual (Irregular)")

    # CENA 4: RESTRIÇÃO
    elif frame < (FRAMES_REAL + FRAMES_FFT + FRAMES_K_MESSY + FRAMES_CONSTRAINT):
        progress = (frame - (FRAMES_REAL + FRAMES_FFT + FRAMES_K_MESSY)) / FRAMES_CONSTRAINT
        blended = (1-progress)*amp_k_messy + progress*target_disk
        im_k.set_data(blended)
        lbl_step.set_text("4. Impor Alvo: Disco Uniforme (NA)")
        lbl_eq.set_text(r"$|U_k| \leftarrow \text{Disco}$")

    # CENA 5: iFFT
    elif frame < (FRAMES_REAL + FRAMES_FFT + FRAMES_K_MESSY + FRAMES_CONSTRAINT + FRAMES_IFFT):
        im_k.set_data(target_disk)
        lbl_arrow_sym.set_text("⬅")
        lbl_arrow_math.set_text(r"$\mathbf{iFFT}$")
        lbl_eq.set_text(r"$|U_k| = \text{Disco}$")
        lbl_step.set_text("5. Retropropagação...")

    # CENA 6: ATUALIZAÇÃO DA FASE (NOVO)
    else:
        # Aqui trocamos a fase velha pela nova!
        im_real.set_data(phase_new)
        
        lbl_arrow_sym.set_text("⬅")
        lbl_arrow_math.set_text(r"$\mathbf{iFFT}$")
        lbl_step.set_text("6. Fase Atualizada (Iteração Concluída)")

    return [im_real, im_k, lbl_step, lbl_arrow_sym, lbl_arrow_math, lbl_eq]

print(f"Gerando GIF Y-Pol Corrigido ({TOTAL_FRAMES} frames)...")
ani = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=False)
ani.save(OUTPUT_FILE, writer='pillow', fps=FPS)
print(f"Sucesso! Salvo: {OUTPUT_FILE}")