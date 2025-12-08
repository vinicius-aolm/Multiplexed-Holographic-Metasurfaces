import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from PIL import Image
import os

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
INPUT_IMAGE_PATH = r"C:\Users\vinicius23011\MATLAB\Projects\TCC\Otimizacao e ML\Algoritmo GS\Images\ZJU img\2.png"
OUTPUT_FILE = "GS_Step2_Forward_Final_Spaced.gif"
SIZE = 300
FPS = 8

# Roteiro de Tempo (Frames)
FRAMES_START = 20      
FRAMES_CONSTRAINT = 25 
FRAMES_TRANSIT = 15    
FRAMES_RESULT = 35     
TOTAL_FRAMES = FRAMES_START + FRAMES_CONSTRAINT + FRAMES_TRANSIT + FRAMES_RESULT

# ==========================================
# 2. DADOS FÍSICOS
# ==========================================
WAVELENGTH = 1064e-9
Z = 380e-6
DX = 520e-9
NA = 0.65
PADDING = 2

def asm_propagate(U, z, wavelength, dx, shape, NA):
    nx, ny = U.shape
    fx = np.fft.fftfreq(nx, dx); fy = np.fft.fftfreq(ny, dx)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    mask = (FX**2 + FY**2) <= (NA/wavelength)**2
    H = np.exp(1j * 2*np.pi/wavelength * z * np.sqrt(np.maximum(0, 1 - (wavelength*FX)**2 - (wavelength*FY)**2)))
    return ifft2(ifftshift(fftshift(fft2(U)) * H * mask))

# Carregar Alvo
try:
    img = Image.open(INPUT_IMAGE_PATH).convert('L').resize((SIZE, SIZE))
    arr = np.array(img)/255.0
    target = 1.0 - arr if np.mean(arr) > 0.5 else arr
except:
    y, x = np.ogrid[:SIZE, :SIZE]
    target = ((x - SIZE//2)**2 + (y - SIZE//2)**2 <= (SIZE//4)**2).astype(float)

# Setup
padded_size = (SIZE*PADDING, SIZE*PADDING)
target_pad = np.zeros(padded_size, dtype=complex)
st = (padded_size[0]-SIZE)//2
target_pad[st:st+SIZE, st:st+SIZE] = target

# Recriar Etapa 1
np.random.seed(42)
U_start = target_pad * np.exp(1j * (np.random.rand(*padded_size)*2*np.pi - np.pi))
U_holo = asm_propagate(U_start, -Z, WAVELENGTH, DX, padded_size, NA)

sl = slice(st, st+SIZE)
holo_amp_bad = np.abs(U_holo)[sl, sl]
holo_phase = np.angle(U_holo)[sl, sl]
holo_amp_bad /= (np.max(holo_amp_bad) + 1e-10) 
holo_amp_laser = np.ones((SIZE, SIZE))

# Simulação Forward
U_holo_constrained = 1.0 * np.exp(1j * np.angle(U_holo))
U_image_result = asm_propagate(U_holo_constrained, Z, WAVELENGTH, DX, padded_size, NA)
result_amp = np.abs(U_image_result)[sl, sl]

# ==========================================
# 3. VISUALIZAÇÃO COM ESPAÇAMENTO CORRIGIDO
# ==========================================
plt.style.use('default')
fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor('white')

# Layout [Left, Bottom, Width, Height]
# Mantemos espaço em cima para os títulos (Bottom=0.40)
ax_amp   = fig.add_axes([0.05, 0.40, 0.25, 0.5]) 
ax_phase = fig.add_axes([0.32, 0.40, 0.25, 0.5]) 
ax_res   = fig.add_axes([0.70, 0.40, 0.25, 0.5]) 
ax_arrow = fig.add_axes([0.60, 0.60, 0.08, 0.1]) 

for ax in [ax_amp, ax_phase]:
    ax.set_title(r"Plano de Fourier $(k_x, k_y)$", color='black', fontsize=12, pad=35)

ax_res.set_title(r"Plano da Imagem $(x, y)$", color='blue', fontsize=12, pad=35)

ax_amp.text(0.5, 1.02, "Amplitude (Restrição)", ha='center', transform=ax_amp.transAxes, fontsize=10, weight='bold')
ax_phase.text(0.5, 1.02, "Fase (Holograma)", ha='center', transform=ax_phase.transAxes, fontsize=10, weight='bold', color='#cc00cc')
ax_res.text(0.5, 1.02, "Resultado", ha='center', transform=ax_res.transAxes, fontsize=10, weight='bold')

for ax in [ax_amp, ax_phase, ax_res, ax_arrow]: ax.axis('off')

# Inicializar
im_amp = ax_amp.imshow(holo_amp_bad, cmap='inferno', vmin=0, vmax=1)
im_phase = ax_phase.imshow(holo_phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
im_res = ax_res.imshow(np.zeros((SIZE, SIZE)), cmap='gray', vmin=0, vmax=1)

# Legenda Inferior
lbl_step = fig.text(0.5, 0.15, "", ha='center', fontsize=16, color='#0044cc', weight='bold')

# Seta e Notação
lbl_arrow_sym = ax_arrow.text(0.5, 0.4, "", ha='center', va='center', fontsize=30, weight='bold')
lbl_arrow_math = ax_arrow.text(0.5, 0.9, "", ha='center', va='bottom', fontsize=12, color='black', weight='bold')

# Equação (Embaixo)
lbl_eq = ax_amp.text(0.5, -0.20, "", ha='center', transform=ax_amp.transAxes, fontsize=14, color='red', weight='bold')

# FÓRMULA DE RODAPÉ
formula_latex = r"Propagação: $U(x, y) = \mathbf{FT}^{-1} \{ U(k_x, k_y) \cdot H(k_x, k_y) \}$"
fig.text(0.5, 0.05, formula_latex, ha='center', fontsize=13, color='#444444')

def update(frame):
    # CENA 1: INICIAL
    if frame < FRAMES_START:
        im_amp.set_data(holo_amp_bad)
        im_amp.set_cmap('inferno')
        im_res.set_visible(False)
        lbl_step.set_text("1. Amplitude Irregular (Calculada)")
        lbl_arrow_sym.set_text("")
        lbl_arrow_math.set_text("")
        lbl_eq.set_text("")
        
    # CENA 2: FADE (RESTRIÇÃO)
    elif frame < (FRAMES_START + FRAMES_CONSTRAINT):
        progress = (frame - FRAMES_START) / FRAMES_CONSTRAINT
        blended = (1-progress)*holo_amp_bad + progress*holo_amp_laser
        im_amp.set_data(blended)
        if progress > 0.8: im_amp.set_cmap('gray')
        
        lbl_step.set_text("2. Impor Fonte: Laser Uniforme")
        lbl_eq.set_text(r"$|U| \leftarrow A_{laser}$")
        
    # CENA 3: PROPAGAÇÃO
    elif frame < (FRAMES_START + FRAMES_CONSTRAINT + FRAMES_TRANSIT):
        im_amp.set_data(holo_amp_laser); im_amp.set_cmap('gray')
        im_res.set_visible(False)
        
        lbl_arrow_sym.set_text("➡")
        lbl_arrow_math.set_text(r"$\mathbf{FT}^{-1}$ (ASM)") 
        
        lbl_step.set_text("3. Propagação Forward (iFT)")
        lbl_eq.set_text(r"$|U| = A_{laser}$")
        
    # CENA 4: RESULTADO
    else:
        im_res.set_visible(True)
        im_res.set_data(result_amp)
        lbl_arrow_sym.set_text("➡")
        lbl_arrow_math.set_text(r"$\mathbf{FT}^{-1}$ (ASM)")
        
        lbl_step.set_text("4. Resultado: Imagem com Ruído")

    return [im_amp, im_phase, im_res, lbl_step, lbl_arrow_sym, lbl_arrow_math, lbl_eq]

print(f"Gerando GIF Final Corrigido ({TOTAL_FRAMES} frames)...")
ani = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=False)
ani.save(OUTPUT_FILE, writer='pillow', fps=FPS)
print(f"Sucesso! Salvo: {OUTPUT_FILE}")