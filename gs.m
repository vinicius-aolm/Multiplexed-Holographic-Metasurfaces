% Author : Musa AYDIN  |  Revisão: Vinícius A. O. L. Moura (2025-08-12)
% e-mail : maydin@fsm.edu.tr
% Fatih Sultan Mehmet Vakif University – Dept. of Computer Engineering
% -------------------------------------------------------------------------
%        GERCHBERG–SAXTON (GS) – PIPELINE COM ASM/NA + CAMADA SEMIÓTICA
% -------------------------------------------------------------------------
% Novidades nesta versão:
%   • Camada "fotografia/semiótica": saliency + bordas + regra-dos-terços
%     → gera um mapa de pesos W que prioriza o que importa na reconstrução.
%   • Alvo ponderado TargetW substitui Target no passo (D) do GS.
%   • Métrica extra: correlação de bordas edgeR, além de Pearson e erro.
%   • Parada antecipada opcional por Pearson/edgeR.
%   • (Opcional) "DOF semiótico": blur controlado no fundo (bokeh).
%   • (Opcional) campo de orientação θ da cena, para orientar meta-átomos.
%   • Stubs p/ mapeamento CST: salvamos W, θ e fase final quantizada.
% -------------------------------------------------------------------------

%% 0) Limpeza, parâmetros e alvo .........................................
clear;  close all;  clc;  tic

% ---------------- Parâmetros numéricos (malha) --------------------------
N     = 256;            % resolução (pixels) dos dois planos
sigma = 2;              % largura do feixe gaussiano (malha x,y)
iter  = 500;            % iterações GS (ajuste conforme necessidade)

% ---------------- Parâmetros físicos (ASM) ------------------------------
lambda = 633e-9;        % comprimento de onda [m]
z      = 0.20;          % distância de propagação [m] (SLM -> tela)
Lx     = 8e-3;          % largura física do campo no plano de entrada [m]
Ly     = 8e-3;          % altura física do campo no plano de entrada [m]
dx     = Lx / N;        % pitch x [m]
dy     = Ly / N;        % pitch y [m]
NA     = 0.25;          % abertura numérica efetiva (ajuste)

% ---------------- Camada semiótica/fotografia (flags) -------------------
USE_SEMIOTICA = true;   % ativa pesos W (saliency+bordas+terços)
USE_DOF       = false;  % "bokeh" semiótico no fundo (blur ponderado)
PEARSON_STOP  = 0.99;   % critério de parada (correlação)
EDGE_STOP     = 0.85;   % critério de parada (correlação de bordas)

% ----------------- Escolha do alvo (teste ou imagem) --------------------
use_synthetic  = false;               % true: alvo sintético p/ debug
synthetic_type = 'checker';           % 'checker' | 'disk' | 'letter'

if use_synthetic
    switch synthetic_type
        case 'checker'
            Target = checkerboard(N/8) > 0.5;
        case 'disk'
            [xx,yy] = meshgrid(linspace(-1,1,N));
            Target = sqrt(xx.^2 + yy.^2) <= 0.5;
        case 'letter'
            Target = zeros(N,N);
            Target(round(N*0.2):round(N*0.8), round(N*0.45):round(N*0.55)) = 1; % barra
            Target(round(N*0.2):round(N*0.35), round(N*0.3):round(N*0.7))  = 1; % topo
            Target(round(N*0.5):round(N*0.65), round(N*0.3):round(N*0.7))  = 1; % meio
        otherwise
            Target = checkerboard(N/8) > 0.5;
    end
    Target = im2double(Target);
else
    % --- imagem-alvo real ------------------------------------------------
    Target = imread('espaco.jpeg');                      % coloque sua imagem aqui
    if size(Target,3) > 1, Target = rgb2gray(Target); end
    Target = im2double(Target);
end
Target = imresize(Target,[N N]);                         % força N×N
T0     = mat2gray(Target);                               % alvo base [0,1]

figure; imshow(T0,[]); title('TARGET – amplitude desejada');

%% 0.1) Camada semiótica/fotográfica (gera W e TargetW) ..................
if USE_SEMIOTICA
    % --- Saliency espectral (Residual Spectrum) -------------------------
    F   = fft2(T0);  logAmp = log(abs(F)+eps);  phaseF = angle(F);
    avg = imfilter(logAmp, fspecial('average', 9), 'replicate');
    SR  = logAmp - avg;
    sal = abs(ifft2(exp(SR + 1i*phaseF))).^2;
    sal = mat2gray(imgaussfilt(sal, 2));

    % --- Bordas (Canny) -------------------------------------------------
    E   = mat2gray(edge(T0,'Canny')); 
    E   = imgaussfilt(E, 0.8);  % suaviza serrilhado

    % --- Regra dos terços ----------------------------------------------
    [u,v] = meshgrid(linspace(0,1,N), linspace(0,1,N));
    s     = 0.06;
    H     = exp(-((u-1/3).^2+(v-1/3).^2)/(2*s^2)) + ...
            exp(-((u-2/3).^2+(v-1/3).^2)/(2*s^2)) + ...
            exp(-((u-1/3).^2+(v-2/3).^2)/(2*s^2)) + ...
            exp(-((u-2/3).^2+(v-2/3).^2)/(2*s^2));
    H     = mat2gray(H);

    % --- Peso combinado -------------------------------------------------
    alpha=0.6; beta=0.3; gamma=0.1;      % saliency/bordas/terços
    W    = mat2gray(alpha*sal + beta*E + gamma*H);

    % --- Alvo com reforço perceptual -----------------------------------
    Tclahe   = adapthisteq(T0,'ClipLimit',0.01,'NumTiles',[8 8]);
    enh      = mat2gray(Tclahe + 0.4*E);          % reforça contornos
    wStrength= 0.35;                               % mistura alvo base + reforço
    TargetW  = mat2gray( (1-wStrength).*T0 + wStrength.*((1+0.7*W).*enh) );

    if USE_DOF
        % "Bokeh": mais blur onde W é baixo (fundo)
        sigmaBlur = 1.0;
        Bg       = imgaussfilt(TargetW, sigmaBlur*(1 - mat2gray(W)));
        TargetW  = mat2gray( 0.7*TargetW + 0.3*Bg );
    end

    figure; 
    subplot(2,2,1); imshow(sal,[]); title('Saliency')
    subplot(2,2,2); imshow(E,[]);   title('Bordas')
    subplot(2,2,3); imshow(W,[]);   title('Peso W (composição)')
    subplot(2,2,4); imshow(TargetW,[]); title('TargetW (alvo ponderado)')
else
    W       = ones(N);               % pesos uniformes
    E       = mat2gray(edge(T0,'Canny'));
    TargetW = T0;
end

%% 1) Plano de entrada – feixe gaussiano .................................
x = linspace(-10,10,N);
y = linspace(-10,10,N);
[X,Y] = meshgrid(x,y);
input_amplitude = exp(-(X.^2 + Y.^2)/(2*sigma^2));   % |I_in|

figure; imagesc(input_amplitude); axis image off
title('Input amplitude (|I_{in}|) – restrição no plano de entrada');

%% 2) Inicialização: chute de fase ......................................
A = fftshift( ifft2( fftshift(TargetW) ) );  % chute orientado pelo alvo ponderado
figure;
subplot(1,2,1), imagesc(abs(A)),   axis image off, title('|A| (amp)')
subplot(1,2,2), imagesc(angle(A)), axis image off, title('∠A (fase)')
sgtitle('Campo A após IFFT do alvo ponderado (chute inicial)');

%% 3) Passo (1) – impõe amplitude gaussiana: B ...........................
B = input_amplitude .* exp(1i*angle(A));
figure;
subplot(1,2,1), imagesc(input_amplitude), axis image off, title('|I_{in}|')
subplot(1,2,2), imagesc(angle(B)),        axis image off, title('∠B = ∠A')
sgtitle('B = |I_{in}|·e^{i∠A} (restrição no plano de entrada)');

%% 4) Passo (2) – propaga ao plano de saída (ASM): C .....................
C = propagateASM(B, dx, dy, lambda, z, NA);
figure;
subplot(1,2,1), imagesc(abs(C)),   axis image off, title('|C| (amp)')
subplot(1,2,2), imagesc(angle(C)), axis image off, title('∠C (fase)')
sgtitle('C = ASM(B) – campo no plano de saída (a z metros)');

%% 5) Passo (3) – impõe amplitude do alvo ponderado: D ...................
D = TargetW .* exp(1i*angle(C));   % <<< aqui entra a intenção fotográfica
figure;
subplot(1,2,1), imagesc(TargetW),   axis image off, title('|TargetW|')
subplot(1,2,2), imagesc(angle(D)),  axis image off, title('∠D = ∠C')
sgtitle('D = |TargetW|·e^{i∠C} (restrição no plano de saída)');

%% 6) Passo (4) – volta ao plano de entrada (ASM -z): novo A .............
A = propagateASM(D, dx, dy, lambda, -z, NA);
figure;
subplot(1,2,1), imagesc(abs(A)),   axis image off, title('|A| após 1 ciclo')
subplot(1,2,2), imagesc(angle(A)), axis image off, title('∠A após 1 ciclo')
sgtitle('Novo A – pronto p/ próximo ciclo');

%% 7) LOOP automático (restante das iterações) ...........................
errorF  = zeros(iter,1);
pearson = zeros(iter,1);
edgeR   = zeros(iter,1);
epss    = 1e-12;

for k = 1:iter
    % (1) entrada
    B = input_amplitude .* exp(1i*angle(A));
    % (2) ida
    C = propagateASM(B, dx, dy, lambda, z, NA);
    % (3) saída (alvo com pesos)
    D = TargetW .* exp(1i*angle(C));
    % (4) volta
    A = propagateASM(D, dx, dy, lambda, -z, NA);

    % ---- métricas ------------------------------------------------------
    rec = abs(C(:));  tgt = TargetW(:);
    errorF(k) = norm(rec - tgt, 'fro');

    rn = (rec - mean(rec)) / (std(rec)+epss);
    tn = (tgt - mean(tgt)) / (std(tgt)+epss);
    pearson(k) = mean(rn .* tn);

    Ce = mat2gray(edge(mat2gray(abs(C)),'Canny'));
    edgeR(k) = corr(Ce(:), E(:));

    % parada antecipada (opcional)
    if pearson(k) >= PEARSON_STOP && edgeR(k) >= EDGE_STOP
        errorF = errorF(1:k); pearson = pearson(1:k); edgeR = edgeR(1:k);
        break
    end
end

% ---- plots de convergência ----
figure;
yyaxis left;  plot(1:length(errorF), errorF, 'LineWidth',1.2); ylabel('Erro (Frobenius)');
yyaxis right; plot(1:length(pearson), pearson, 'LineWidth',1.2); hold on
plot(1:length(edgeR), edgeR, '--', 'LineWidth',1.1); ylabel('r (Pearson/Edge)')
xlabel('Iteração'); grid on; title('Convergência: erro × Pearson × borda');
legend('Erro','Pearson','EdgeCorr','Location','best')

% ---- reconstrução final ----
figure; imagesc(abs(C)); axis image off
title(sprintf('Reconstrução final (|C|) após %d iterações',length(pearson)));

% ---- salva métricas em CSV ----
T = table((1:length(errorF)).', errorF(:), pearson(:), edgeR(:), ...
          'VariableNames',{'iter','errorF','pearson','edgeR'});
writetable(T, 'gs_metrics.csv');

%% 8) Exportação do mapa de fase (plano de entrada) ......................
phase_in = mod(angle(A), 2*pi);      % [0, 2π)
Lq       = 256;                      % níveis (8 bits)
phase_q  = uint8(round( phase_in/(2*pi) * (Lq-1) ));

imwrite(phase_q, 'phase_map_8bit.tif');
save('phase_map.mat','phase_in','dx','dy','lambda','z','NA','W');
writematrix(phase_q, 'phase_map_levels.csv');

%% 9) (Opcional) Campo de orientação θ (para orientar meta-átomos) .......
[Gx,Gy] = imgradientxy(T0);
[~,theta_deg] = imgradient(Gx,Gy);         % -180..180 (graus)
theta_deg = imgaussfilt(theta_deg, 1.0);   % suaviza
writematrix(theta_deg, 'orientation_theta_deg.csv');

toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            FUNÇÕES AUXILIARES                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Uz = propagateASM(U0, dx, dy, lambda, z, NA)
% propagateASM  Propaga um campo U0 por distância z usando o
%               Método do Espectro Angular (ASM), com filtro por NA.
% Entradas:
%   U0     : campo complexo no plano z=0 (matriz NxN)
%   dx, dy : passo espacial [m]
%   lambda : comprimento de onda [m]
%   z      : distância de propagação [m] (pode ser negativa)
%   NA     : abertura numérica (aplica filtro passa-baixas circular)
%
% Saída:
%   Uz     : campo complexo propagado a z [m]

    [Ny, Nx] = size(U0);
    k  = 2*pi/lambda;

    % Grades de frequência espacial (centradas)
    fx = (-Nx/2:Nx/2-1) / (Nx*dx);
    fy = (-Ny/2:Ny/2-1) / (Ny*dy);
    [FX, FY] = meshgrid(fx, fy);
    f2 = FX.^2 + FY.^2;

    % Termo kz (suprime evanescentes)
    arg = max(0, 1 - (lambda^2)*f2);
    kz  = k * sqrt(arg);

    % Transfer function + filtro NA
    H = exp(1i * z .* kz);
    fcut = (NA / lambda)^2;          % raio de corte em f^2
    pass = (f2 <= fcut);
    H = H .* pass;

    % FFT centrada -> aplica H -> IFFT centrada
    U0f = fftshift( fft2( ifftshift(U0) ) );
    Uzf = U0f .* H;
    Uz  = fftshift( ifft2( ifftshift(Uzf) ) );
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  STUB: casamento por pixel (para CST)                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Como usar depois:
%  1) Carregue a biblioteca de meta-átomos (CSV): cada linha tem
%     [Lx, Ly, h, tx_real, tx_imag, ty_real, ty_imag].
%  2) Passe phix, phiy (fase alvo por pixel), pesos W (importância),
%     e receba índices das geometrias ideais para cada pixel.
%
% Exemplo de chamada:
%   lib = readmatrix('meta_library.csv');                 % sua biblioteca
%   [idx] = pick_meta_atoms(phase_in_x, phase_in_y, lib, W);
%   % depois gere o GDSII a partir de lib(idx,1:3) (Lx,Ly,h) por pixel.

function idx = pick_meta_atoms(phix, phiy, lib, W)
% phix, phiy : mapas de fase alvo (rad) para x e y (NxN)
% lib        : matriz [Lx, Ly, h, txr, txi, tyr, tyi]
% W          : pesos [0..1] (NxN) – importância semiótica
% idx        : índice do meta-átomo escolhido por pixel (NxN)

    [Ny,Nx] = size(phix);
    nLib = size(lib,1);
    tx   = lib(:,4) + 1i*lib(:,5);
    ty   = lib(:,6) + 1i*lib(:,7);

    % alvos unitários
    Tx = exp(1i*phix);  Ty = exp(1i*phiy);

    idx = zeros(Ny,Nx);
    for j = 1:Ny
        for i = 1:Nx
            w  = 0.5 + 0.5*W(j,i);     % mais peso onde W alto
            % custo: aproxima fase x e y simultaneamente
            err = w*abs(tx - Tx(j,i)).^2 + (1-w)*abs(ty - Ty(j,i)).^2;
            [~,best] = min(err);
            idx(j,i) = best;
        end
    end
end
