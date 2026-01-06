import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import fft2, ifft2
from PIL import Image
import os

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
INPUT_IMAGE_PATH = r"C:\Users\vinicius23011\MATLAB\Projects\TCC\Otimizacao e ML\Algoritmo GS\Images\ZJU img\2.png"
OUTPUT_FILE = "GS_Step3_Exact_Physics.gif"
SIZE = 90  # Tamanho usado no TCC
PADDING = 2
FPS = 10
TOTAL_ITERS = 50 

# ==========================================
# 2. FUNÇÕES EXATAS DO SEU TCC
# ==========================================
def apply_zero_padding(image, padding_factor=2):
    original_size = image.shape
    padded_size = (int(image.shape[0] * padding_factor), int(image.shape[1] * padding_factor))
    padded_image = np.zeros(padded_size, dtype=complex)
    start_row = (padded_size[0] - original_size[0]) // 2
    start_col = (padded_size[1] - original_size[1]) // 2
    padded_image[start_row:start_row+original_size[0], start_col:start_col+original_size[1]] = image
    return padded_image, original_size

def extract_center(image, original_size):
    nx, ny = original_size
    start_row = (image.shape[0] - nx) // 2
    start_col = (image.shape[1] - ny) // 2
    return image[start_row:start_row+nx, start_col:start_col+ny]

def create_low_pass_filter(shape, wavelength, dx, NA):
    nx, ny = shape
    fx = np.fft.fftfreq(nx, dx)
    fy = np.fft.fftfreq(ny, dx)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    f_cutoff = NA / wavelength
    freq_radius = np.sqrt(FX**2 + FY**2)
    return (freq_radius <= f_cutoff).astype(np.float64)

def angular_spectrum_propagation(U, wavelength, z, dx, filter_mask=None):
    k = 2 * np.pi / wavelength
    nx, ny = U.shape
    fx = np.fft.fftfreq(nx, dx)
    fy = np.fft.fftfreq(ny, dx)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    
    root_term = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
    root_term[root_term < 0] = 0
    H = np.exp(1j * k * z * np.sqrt(root_term))
    
    if filter_mask is not None:
        H = H * filter_mask
        
    U_freq = fft2(U)
    U_prop_freq = U_freq * H
    return ifft2(U_prop_freq)

def calculate_metrics(target_center, recon_center):
    # Garante que ambos são reais e normalizados para métrica justa
    t = np.abs(target_center)
    r = np.abs(recon_center)
    r = r / (np.max(r) + 1e-10) # Normalizar reconstrução
    
    # Pearson
    pearson = np.corrcoef(t.flatten(), r.flatten())[0, 1]
    # RMSE
    rmse = np.sqrt(np.mean((t - r)**2))
    return pearson, rmse

# ==========================================
# 3. SIMULAÇÃO REAL
# ==========================================
WAVELENGTH = 1064e-9
Z = 380e-6
DX = 520e-9
NA = 0.65

# Carregar e Preparar Imagem
try:
    img = Image.open(INPUT_IMAGE_PATH).convert('L').resize((SIZE, SIZE), Image.LANCZOS)
    target_original = np.array(img, dtype=float) / 255.0
    # Binarizar levemente para ajudar contraste visual
    target_original = (target_original > 0.5).astype(float)
except:
    target_original = np.zeros((SIZE, SIZE))
    target_original[20:70, 20:70] = 1.0

# Setup Inicial
target_padded, original_size = apply_zero_padding(target_original, PADDING)
nx_pad, ny_pad = target_padded.shape
filter_mask = create_low_pass_filter((nx_pad, ny_pad), WAVELENGTH, DX, NA)

print("Rodando Algoritmo GS...")

# Estado Inicial (Fase Aleatória)
np.random.seed(42)
phase = np.random.rand(nx_pad, ny_pad) * 2 * np.pi
U = target_padded * np.exp(1j * phase)

history = []

for k in range(TOTAL_ITERS):
    # 1. Forward (Hologram -> Image)
    # Nota: No seu código GS, o passo 1 propaga U para Image.
    U_image = angular_spectrum_propagation(U, WAVELENGTH, Z, DX, filter_mask)
    
    # Capturar estado para visualização
    # Importante: O que vemos no "Plano de Fourier" é a fase de U antes de propagar
    current_holo_phase = np.angle(U)
    
    # Extrair e medir (Plano da Imagem)
    amplitude_image = np.abs(U_image)
    recon_center = extract_center(amplitude_image, original_size)
    
    pearson, rmse_val = calculate_metrics(target_original, recon_center)
    
    # Salvar para o GIF (usando o recorte central para melhor visualização)
    # Para a fase do holograma, também mostramos o centro ou tudo? 
    # O centro costuma ter a informação principal.
    phase_view = extract_center(current_holo_phase, original_size)
    
    history.append({
        'iter': k + 1,
        'phase': phase_view,
        'recon': recon_center,
        'pearson': pearson,
        'rmse': rmse_val
    })
    
    # --- Lógica GS ---
    # 2. Impor Alvo na Imagem (Amplitude Constraint)
    phase_image = np.angle(U_image)
    U_image_updated = target_padded * np.exp(1j * phase_image)
    
    # 3. Backward (Image -> Hologram)
    U_back = angular_spectrum_propagation(U_image_updated, WAVELENGTH, -Z, DX, filter_mask)
    
    # 4. Impor Fonte no Holograma (Source Constraint)
    phase_hologram = np.angle(U_back)
    U = np.exp(1j * phase_hologram) # Amplitude = 1

# ==========================================
# 4. VISUALIZAÇÃO (LAYOUT ETAPA 2)
# ==========================================
plt.style.use('default')
fig = plt.figure(figsize=(14, 7))
fig.patch.set_facecolor('white')

# Layout [Left, Bottom, Width, Height]
ax_amp   = fig.add_axes([0.05, 0.40, 0.25, 0.5]) 
ax_phase = fig.add_axes([0.32, 0.40, 0.25, 0.5]) 
ax_res   = fig.add_axes([0.70, 0.40, 0.25, 0.5]) 
ax_arrow = fig.add_axes([0.60, 0.60, 0.08, 0.1]) 

# Títulos (Idênticos Etapa 2)
for ax in [ax_amp, ax_phase]:
    ax.set_title(r"Plano de Fourier $(k_x, k_y)$", color='black', fontsize=12, pad=35)
ax_res.set_title(r"Plano da Imagem $(x, y)$", color='blue', fontsize=12, pad=35)

ax_amp.text(0.5, 1.02, "Amplitude (Laser)", ha='center', transform=ax_amp.transAxes, fontsize=10, weight='bold')
ax_phase.text(0.5, 1.02, "Fase (Evolução)", ha='center', transform=ax_phase.transAxes, fontsize=10, weight='bold', color='#cc00cc')
ax_res.text(0.5, 1.02, "Reconstrução (Iterativa)", ha='center', transform=ax_res.transAxes, fontsize=10, weight='bold')

for ax in [ax_amp, ax_phase, ax_res, ax_arrow]: ax.axis('off')

# Inicializar
im_amp = ax_amp.imshow(np.ones((SIZE, SIZE)), cmap='gray', vmin=0, vmax=1)
im_phase = ax_phase.imshow(history[0]['phase'], cmap='twilight', vmin=-np.pi, vmax=np.pi)
im_res = ax_res.imshow(history[0]['recon'], cmap='gray', vmin=0, vmax=1)

# SETAS ESTÁTICAS
ax_arrow.text(0.5, 0.5, r"$\rightleftarrows$", ha='center', va='center', fontsize=50, weight='bold')
ax_arrow.text(0.5, 1.1, r"$\mathbf{FT}$", ha='center', va='bottom', fontsize=12, weight='bold') # Forward
ax_arrow.text(0.5, -0.1, r"$\mathbf{FT}^{-1}$", ha='center', va='top', fontsize=12, weight='bold') # Backward

# HUD
txt_iter = fig.text(0.5, 0.20, "", ha='center', fontsize=18, color='black', weight='bold')
txt_metrics = fig.text(0.5, 0.14, "", ha='center', fontsize=14, color='#444444', weight='bold')

# Rodapé
fig.text(0.5, 0.05, "Ciclo Iterativo: Otimização via Gerchberg-Saxton (Código TCC)", ha='center', fontsize=12, color='gray')

def update(frame):
    data = history[frame]
    
    # Atualizar imagens
    im_phase.set_data(data['phase'])
    
    # Normalizar reconstrução para visualização (brilho consistente)
    recon_vis = data['recon'] / (np.max(data['recon']) + 1e-10)
    im_res.set_data(recon_vis)
    
    # Atualizar Métricas
    txt_iter.set_text(f"Iteração: {data['iter']}")
    
    # Cor dinâmica (Verde se bom)
    p_val = data['pearson']
    color = 'green' if p_val > 0.9 else 'black'
    if p_val < 0: color = 'red' # Alerta se algo estiver muito errado
    
    txt_metrics.set_text(f"Pearson: {p_val:.4f}  |  RMSE: {data['rmse']:.4f}")
    txt_metrics.set_color(color)

    return [im_phase, im_res, txt_iter, txt_metrics]

print(f"Gerando GIF Physics-Accurate ({TOTAL_ITERS} frames)...")
ani = animation.FuncAnimation(fig, update, frames=TOTAL_ITERS, blit=False)
ani.save(OUTPUT_FILE, writer='pillow', fps=FPS)
print(f"Sucesso! Salvo: {OUTPUT_FILE}")