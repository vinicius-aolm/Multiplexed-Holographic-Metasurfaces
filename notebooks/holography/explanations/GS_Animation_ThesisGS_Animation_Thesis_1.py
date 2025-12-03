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
OUTPUT_FILE = "GS_Step1_Final_Layout.gif"
SIZE = 300
FPS = 8

# Roteiro de Tempo
FRAMES_TARGET = 20     # 1. Mostrar Alvo
FRAMES_PHASE = 20      # 2. Mostrar Fase
FRAMES_TRANSIT = 15    # 3. Seta
FRAMES_RESULT = 30     # 4. Resultado no Fourier

TOTAL_FRAMES = FRAMES_TARGET + FRAMES_PHASE + FRAMES_TRANSIT + FRAMES_RESULT

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

# Fase Inicial (Aleatória)
np.random.seed(42)
initial_phase = np.random.rand(*padded_size) * 2 * np.pi - np.pi

# Campo Complexo na Imagem
U_image_start = target_pad * np.exp(1j * initial_phase)

# Retropropagar (Imagem -> Fourier)
U_holo = asm_propagate(U_image_start, -Z, WAVELENGTH, DX, padded_size, NA)

# Preparar Visualização
sl = slice(st, st+SIZE)
img_amp_target = target
img_phase_rand = initial_phase[sl, sl]
holo_amp_result = np.abs(U_holo)[sl, sl]

# Normalização Visual (para evitar tela preta)
holo_amp_result /= (np.max(holo_amp_result) + 1e-10)

# ==========================================
# 3. VISUALIZAÇÃO 3 PAINÉIS (LAYOUT REVERSO)
# ==========================================
plt.style.use('default')
fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor('white')

# Layout [Left, Bottom, Width, Height]
# Esquerda: Fourier (1 painel)
ax_holo_amp = fig.add_axes([0.05, 0.40, 0.25, 0.5]) 

# Direita: Imagem (2 painéis)
ax_img_phase = fig.add_axes([0.42, 0.40, 0.25, 0.5]) 
ax_img_amp   = fig.add_axes([0.70, 0.40, 0.25, 0.5]) 

# Seta no meio (apontando para esquerda)
ax_arrow = fig.add_axes([0.31, 0.60, 0.08, 0.1]) 

# Títulos Principais (Domínios)
ax_holo_amp.set_title(r"Plano de Fourier $(k_x, k_y)$", color='black', fontsize=12, pad=35)
# Título compartilhado ou individual para a direita? Individuais ficam mais alinhados.
ax_img_phase.set_title(r"Plano da Imagem $(x, y)$", color='blue', fontsize=12, pad=35)
ax_img_amp.set_title(r"Plano da Imagem $(x, y)$", color='blue', fontsize=12, pad=35)

# Subtítulos (O que é cada coisa)
ax_holo_amp.text(0.5, 1.02, "Amplitude (Resultado)", ha='center', transform=ax_holo_amp.transAxes, fontsize=10, weight='bold')
ax_img_phase.text(0.5, 1.02, "Fase (Estimativa)", ha='center', transform=ax_img_phase.transAxes, fontsize=10, weight='bold', color='#cc00cc')
ax_img_amp.text(0.5, 1.02, "Amplitude (Alvo)", ha='center', transform=ax_img_amp.transAxes, fontsize=10, weight='bold')

for ax in [ax_holo_amp, ax_img_phase, ax_img_amp, ax_arrow]: ax.axis('off')

# Inicializar
im_holo = ax_holo_amp.imshow(np.zeros((SIZE, SIZE)), cmap='inferno', vmin=0, vmax=1)
im_phase = ax_img_phase.imshow(np.zeros((SIZE, SIZE)), cmap='twilight', vmin=-np.pi, vmax=np.pi)
im_amp = ax_img_amp.imshow(np.zeros((SIZE, SIZE)), cmap='gray', vmin=0, vmax=1)

# Legenda Inferior
lbl_step = fig.text(0.5, 0.15, "", ha='center', fontsize=16, color='#0044cc', weight='bold')

# Seta e Notação
lbl_arrow_sym = ax_arrow.text(0.5, 0.4, "", ha='center', va='center', fontsize=30, weight='bold')
lbl_arrow_math = ax_arrow.text(0.5, 0.9, "", ha='center', va='bottom', fontsize=12, color='black', weight='bold')

# Equação (Target) - Embaixo do painel da direita
lbl_eq = ax_img_amp.text(0.5, -0.20, "", ha='center', transform=ax_img_amp.transAxes, fontsize=14, color='red', weight='bold')

# FÓRMULA DE RODAPÉ (FT normal para ida ao Fourier)
formula_latex = r"Retropropagação: $U(k_x, k_y) = \mathbf{FT} \{ U(x, y) \cdot H^{-1} \}$"
fig.text(0.5, 0.05, formula_latex, ha='center', fontsize=13, color='#444444')

def update(frame):
    # CENA 1: ALVO (DIREITA)
    if frame < FRAMES_TARGET:
        im_holo.set_visible(False)
        im_phase.set_visible(False)
        im_amp.set_visible(True)
        im_amp.set_data(img_amp_target)
        
        lbl_step.set_text("1. Definir Amplitude do Alvo")
        lbl_eq.set_text(r"$|U| = A_{alvo}$")
        lbl_arrow_sym.set_text("")
        lbl_arrow_math.set_text("")

    # CENA 2: FASE ALEATÓRIA (CENTRO)
    elif frame < (FRAMES_TARGET + FRAMES_PHASE):
        im_phase.set_visible(True)
        im_phase.set_data(img_phase_rand)
        
        lbl_step.set_text("2. Adicionar Fase Inicial (Aleatória)")
        # Mostra que estamos construindo o campo
        lbl_eq.set_text(r"$U = A_{alvo} \cdot e^{i \phi_{rand}}$")

    # CENA 3: RETROPROPAGAÇÃO (SETA)
    elif frame < (FRAMES_TARGET + FRAMES_PHASE + FRAMES_TRANSIT):
        im_holo.set_visible(False)
        lbl_arrow_sym.set_text("⬅") # Seta para esquerda
        lbl_arrow_math.set_text(r"$\mathbf{FT}$ (ASM)") # Transformada direta (Espaço -> Freq)
        
        lbl_step.set_text("3. Retropropagação para Fourier")

    # CENA 4: RESULTADO (ESQUERDA)
    else:
        im_holo.set_visible(True)
        im_holo.set_data(holo_amp_result)
        lbl_arrow_sym.set_text("⬅")
        lbl_arrow_math.set_text(r"$\mathbf{FT}$ (ASM)")
        
        lbl_step.set_text("4. Amplitude Resultante (Irregular)")
        lbl_eq.set_text("") # Limpa equação da direita para focar na esquerda

    return [im_holo, im_phase, im_amp, lbl_step, lbl_arrow_sym, lbl_arrow_math, lbl_eq]

print(f"Gerando GIF Etapa 1 Final ({TOTAL_FRAMES} frames)...")
ani = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=False)
ani.save(OUTPUT_FILE, writer='pillow', fps=FPS)
print(f"Sucesso! Salvo: {OUTPUT_FILE}")