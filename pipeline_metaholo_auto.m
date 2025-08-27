% ========================================================================
% pipeline_metaholo_auto.m — Auto-biblioteca + GS/ASM (com Pearson/Edge)
% ========================================================================
% Pastas esperadas:
%   library_raw/aXXX_bYYY_hZZZ[_pxPPP][_pyQQQ]/  ← 8 ASCII do CST
% Saídas em library_processed/:
%   meta_library.csv, cst_logs.csv,
%   phase_map.mat/.tif, gs_metrics.csv,
%   chosen_indices.mat, quick_report.txt
% ========================================================================
clear; clc; close all;

%% =========================== CONFIGURAÇÃO ================================
ROOT            = pwd;
RAW_DIR         = fullfile(ROOT,'library_raw');
PROC_DIR        = fullfile(ROOT,'library_processed');
TARGETS_DIR     = fullfile(ROOT,'targets');

if ~exist(RAW_DIR,'dir'),  mkdir(RAW_DIR);  end
if ~exist(PROC_DIR,'dir'), mkdir(PROC_DIR); end
if ~exist(TARGETS_DIR,'dir'), mkdir(TARGETS_DIR); end

% Flags
DO_IMPORT_ALL   = true;     % varre library_raw e atualiza biblioteca
DO_RUN_GS       = true;     % roda GS/ASM (sem semiótica)
DO_PICK         = true;     % casa fase com biblioteca
SHOW_GS_STEPS   = true;     % figuras A/B/C/D + alvo/entrada e final
SAVE_FIGS       = true;     % salvar figuras em PNG no PROC_DIR

% Física
lambda_nm   = 1064;                 % nm (coerente com CST)
lambda      = lambda_nm * 1e-9;     % m
NA          = 0.25;

% Período (fallback) — será sobrescrito se pasta trouxer px/py
Lx_nm_fallback = 520;
Ly_nm_fallback = 520;

% Ordem dos modos (CST Floquet)
MODE1_IS_TE = true;

% Sanidade S-params
max_cross   = 0.03;                 % |S21 cross-pol| tolerado
tol_power   = 0.05;                 % |S21|^2 + |S11|^2 ≈ 1 ± 5%

% GS/ASM
N           = 256;
iter        = 300;
z           = 0.20;                 % m
Lx_area_m   = 8e-3;  Ly_area_m = 8e-3;
dx = Lx_area_m/N;  dy = Ly_area_m/N;
TARGET_IMG  = fullfile(TARGETS_DIR,'espaco.jpeg');   % opcional

% Saídas
LIB_CSV     = fullfile(PROC_DIR,'meta_library.csv');
LOG_CSV     = fullfile(PROC_DIR,'cst_logs.csv');
REPORT_TXT  = fullfile(PROC_DIR,'quick_report.txt');

%% =============================== [A] IMPORT ==============================
if DO_IMPORT_ALL
    fprintf('\n[A] Lendo subpastas em %s ...\n', RAW_DIR);

    % === tabela base (existente) ===
    if isfile(LIB_CSV)
        T = readtable(LIB_CSV);
    else
        T = table('Size',[0 13], ...
            'VariableTypes', {'double','double','double','double','double', ...
                              'double','double','double','double','double','double','double','double'}, ...
            'VariableNames', {'Lx_nm','Ly_nm','h_nm','a_nm','b_nm', ...
                              'Re_Exx','Im_Exx','Re_Eyy','Im_Eyy','Abs_Exx','Abs_Eyy','Phi_x_deg','Phi_y_deg'});
    end
    % Garante tipos numéricos
    T = convertvars(T, T.Properties.VariableNames, 'double');

    % Política de duplicata: 'skip' (padrão) ou 'update'
    DEDUP_MODE      = 'update';   % 'skip' ou 'update'
    FORCE_REIMPORT  = false;      % true: ignora markers processed.ok

    subs = dir(RAW_DIR); subs = subs([subs.isdir]); subs = subs(~ismember({subs.name},{'.','..'}));
    logs = {};
    n_added = 0; n_skipped = 0; n_updated = 0; n_missing = 0;

    for k = 1:numel(subs)
        sub    = subs(k).name;
        subdir = fullfile(RAW_DIR,sub);

        % ---- marker para evitar reprocessamento desnecessário ----
        marker = fullfile(subdir,'processed.ok');
        if ~FORCE_REIMPORT && isfile(marker)
            n_skipped = n_skipped + 1;
            logs(end+1,:) = {sub, 'SKIP(marker)'}; %#ok<SAGROW>
            fprintf('  · %-30s  (marcado como processado) \n', sub);
            continue
        end

        % Extrair a,b,h e (opcionais) px/py do nome da pasta
        a_nm = NaN; b_nm = NaN; h_nm = NaN; Lx_nm = Lx_nm_fallback; Ly_nm = Ly_nm_fallback;
        tok = regexp(sub,'a(\d+)[^\d]+b(\d+)[^\d]+h(\d+)','tokens','once');
        if ~isempty(tok), a_nm=str2double(tok{1}); b_nm=str2double(tok{2}); h_nm=str2double(tok{3}); end
        tpx = regexp(sub,'px(\d+)','tokens','once'); if ~isempty(tpx), Lx_nm = str2double(tpx{1}); end
        tpy = regexp(sub,'py(\d+)','tokens','once'); if ~isempty(tpy), Ly_nm = str2double(tpy{1}); end

        % Verifica arquivos necessários
        need = { ...
          'SZmax(1),Zmin(1).txt','SZmax(2),Zmin(2).txt', ...
          'SZmax(1),Zmin(2).txt','SZmax(2),Zmin(1).txt', ...
          'SZmin(1),Zmin(1).txt','SZmin(2),Zmin(2).txt', ...
          'SZmin(1),Zmin(2).txt','SZmin(2),Zmin(1).txt' };
        ok=true; miss={};
        for q=1:numel(need)
            if ~isfile(fullfile(subdir,need{q})), ok=false; miss{end+1}=need{q}; end %#ok<AGROW>
        end
        if ~ok
            n_missing = n_missing + 1;
            logs(end+1,:) = {sub, sprintf('FALTANDO: %s', strjoin(miss,', '))}; %#ok<SAGROW>
            fprintf('  - %-30s faltam %d arquivo(s)\n', sub, numel(miss));
            continue
        end

        % Lê complexos
        getC  = @(fn) read_cst_scalar_complex(fullfile(subdir,fn));
        S21_11 = getC('SZmax(1),Zmin(1).txt');  % 1->1
        S21_22 = getC('SZmax(2),Zmin(2).txt');  % 2->2
        S21_12 = getC('SZmax(1),Zmin(2).txt');  % 2->1 (cross)
        S21_21 = getC('SZmax(2),Zmin(1).txt');  % 1->2 (cross)
        S11_11 = getC('SZmin(1),Zmin(1).txt');  % reflexões p/ power check
        S22_22 = getC('SZmin(2),Zmin(2).txt');

        if MODE1_IS_TE, Exx=S21_11; Eyy=S21_22; else, Exx=S21_22; Eyy=S21_11; end

        P_TE = abs(S21_11)^2 + abs(S11_11)^2;
        P_TM = abs(S21_22)^2 + abs(S22_22)^2;
        cross_max = max(abs(S21_12),abs(S21_21));

        warn = '';
        if abs(P_TE-1)>tol_power || abs(P_TM-1)>tol_power, warn=[warn sprintf('[POWER TE=%.3f TM=%.3f]',P_TE,P_TM)]; end
        if cross_max>max_cross, warn=[warn sprintf('[XPOL=%.3f]',cross_max)]; end

        % ---- CHAVE geométrica (dedup) ----
        % Arredondar para inteiro em nm (evita problemas de float)
        keyLx = round(Lx_nm); keyLy = round(Ly_nm);
        keyH  = round(h_nm);  keyA  = round(a_nm);  keyB = round(b_nm);

        isDup = T.Lx_nm==keyLx & T.Ly_nm==keyLy & T.h_nm==keyH & ...
                T.a_nm==keyA & T.b_nm==keyB;

        % Linha candidata (numérica)
        r = [ keyLx, keyLy, keyH, keyA, keyB, ...
              real(Exx), imag(Exx), real(Eyy), imag(Eyy), ...
              abs(Exx), abs(Eyy), angle(Exx)*180/pi, angle(Eyy)*180/pi ];

        if any(isDup)
            switch lower(DEDUP_MODE)
                case 'skip'
                    n_skipped = n_skipped + 1;
                    logs(end+1,:) = {sub, 'SKIP(duplicata)'}; %#ok<SAGROW>
                    fprintf('  · %-30s duplicata → SKIP\n', sub);

                case 'update'
                    ndup = nnz(isDup);
                    T{isDup, :} = repmat(r, ndup, 1);
                    n_updated = n_updated + ndup;
                    logs(end+1,:) = {sub, 'UPDATE(duplicata)'}; %#ok<SAGROW>
                    fprintf('  * %-30s duplicata → UPDATE (%d)\n', sub, ndup);
            end
        else
            % adiciona
            T = [T; array2table(r, 'VariableNames', T.Properties.VariableNames)]; %#ok<AGROW>
            n_added = n_added + 1;
            fprintf('  + %-30s Exx∠=%6.1f° |Exx|=%.2f  Eyy∠=%6.1f° |Eyy|=%.2f  %s\n', ...
                sub, angle(Exx)*180/pi, abs(Exx), angle(Eyy)*180/pi, abs(Eyy), warn);
        end

        % grava marker (para acelerar próximas rodadas)
        fid = fopen(marker,'w');
        if fid>0
            fprintf(fid,'ok %s\n', datestr(now));
            fclose(fid);
        end

        logs(end+1,:) = {sub, warn}; %#ok<SAGROW>
    end

    % === salvar CSV único (deduplicado) ===
    % (garante unicidade da chave — se vieram duplicados fora do fluxo)
    [~, ia] = unique( strcat(string(T.Lx_nm),'_',string(T.Ly_nm),'_', ...
                             string(T.h_nm),'_',string(T.a_nm),'_',string(T.b_nm)) , 'stable');
    T = T(ia,:);
    writetable(T, LIB_CSV);

    % log detalhado
    if ~isempty(logs)
        Tlog = cell2table(logs, 'VariableNames',{'subfolder','notes'});
        writetable(Tlog, LOG_CSV);
    end

    % resumo no relatório
    fid = fopen(REPORT_TXT,'a');
    if fid>0
        fprintf(fid,'\n[A] Import summary %s\n', datestr(now));
        fprintf(fid,'  RAW_DIR: %s\n', RAW_DIR);
        fprintf(fid,'  ADD=%d  SKIP=%d  UPDATE=%d  MISSING=%d | total lib: %d\n', ...
            n_added, n_skipped, n_updated, n_missing, height(T));
        fclose(fid);
    end

    fprintf('  => ADD=%d  SKIP=%d  UPDATE=%d  MISSING=%d | total na lib: %d\n', ...
        n_added, n_skipped, n_updated, n_missing, height(T));
end

%% =============================== [B] GS ==================================
if DO_RUN_GS
    fprintf('\n[B] GS/ASM (sem semiótica)  N=%d  iter=%d  λ=%.1f nm\n', N, iter, lambda_nm);

    % Alvo (amplitude desejada)
    if isfile(TARGET_IMG)
        Timg = imread(TARGET_IMG); if size(Timg,3)>1, Timg=rgb2gray(Timg); end
        T0 = im2double(imresize(Timg,[N N]));
    else
        T0 = im2double(imresize(checkerboard(N/8)>0.5,[N N]));
    end
    Target = mat2gray(T0);

    % Inicialização
    xg = linspace(-10,10,N); [X,Y] = meshgrid(xg,xg);
    sigma = 2;
    input_amp = exp(-(X.^2+Y.^2)/(2*sigma^2));  % |I_in|
    A = fftshift(ifft2(fftshift(Target)));      % chute inicial

    errorF = zeros(iter,1); pearson = zeros(iter,1); edgeR = zeros(iter,1); epss=1e-12;

    if SHOW_GS_STEPS
        fig1 = figure; imshow(Target,[]); title('TARGET – amplitude desejada');
        fig2 = figure; imshow(input_amp,[]); title('|I_{in}| (amplitude de entrada)');
    end

    for k = 1:iter
        % (1) entrada
        B = input_amp .* exp(1i*angle(A));
        % (2) ida
        C = propagateASM_local(B, dx, dy, lambda,  z, NA);
        % (3) saída
        D = Target    .* exp(1i*angle(C));
        % (4) volta
        A = propagateASM_local(D, dx, dy, lambda, -z, NA);

        rec = abs(C(:));  tgt = Target(:);
        errorF(k) = norm(rec - tgt, 'fro');

        rn = (rec-mean(rec))/(std(rec)+epss);
        tn = (tgt-mean(tgt))/(std(tgt)+epss);
        pearson(k) = mean(rn.*tn);

        Ce = mat2gray(edge(mat2gray(abs(C)),'Canny'));
        E  = mat2gray(edge(Target,'Canny'));
        edgeR(k) = corr(Ce(:), E(:));

        if k==1 && SHOW_GS_STEPS
            fig3 = figure('Name','Passos do 1º ciclo');
            subplot(2,3,1); imshow(abs(A),[]); title('|A| (antes)')
            subplot(2,3,2); imshow(angle(A),[]); title('∠A (antes)')
            subplot(2,3,3); imshow(angle(B),[]); title('∠B')
            subplot(2,3,4); imshow(abs(C),[]);   title('|C|')
            subplot(2,3,5); imshow(angle(C),[]); title('∠C')
            subplot(2,3,6); imshow(angle(D),[]); title('∠D')
        end
    end

    % Curvas de convergência
    fconv = figure('Name','Convergência');
    yyaxis left;  plot(errorF,'LineWidth',1.2); ylabel('Erro (Frobenius)');
    yyaxis right; plot(pearson,'LineWidth',1.2); hold on
    plot(edgeR,'--','LineWidth',1.1); ylabel('r (Pearson / Edge)')
    xlabel('Iteração'); grid on; legend('Erro','Pearson','EdgeCorr','Location','best')
    title('Convergência: erro × Pearson × borda');

    % Resultado final
    ffinal = figure('Name','Reconstrução final'); imshow(abs(C),[]); title(sprintf('|C| após %d iterações', numel(pearson)));

    % Salvas
    Ttbl = table((1:length(errorF)).', errorF(:), pearson(:), edgeR(:), ...
                 'VariableNames',{'iter','errorF','pearson','edgeR'});
    writetable(Ttbl, fullfile(PROC_DIR,'gs_metrics.csv'));

    phase_in = mod(angle(A), 2*pi);
    Lq = 256; phase_q = uint8(round( phase_in/(2*pi) * (Lq-1) ));
    imwrite(phase_q, fullfile(PROC_DIR,'phase_map_8bit.tif'));
    save(fullfile(PROC_DIR,'phase_map.mat'),'phase_in','dx','dy','lambda','z','NA');

    if SAVE_FIGS
        saveas(fconv, fullfile(PROC_DIR,'convergencia.png'));
        saveas(ffinal, fullfile(PROC_DIR,'recon_final.png'));
        try, saveas(fig1, fullfile(PROC_DIR,'target.png')); end %#ok<*TRYNC>
        try, saveas(fig2, fullfile(PROC_DIR,'input_amp.png')); end
        try, saveas(fig3, fullfile(PROC_DIR,'steps_cycle1.png')); end
    end

    % Por ora, mesmo mapa de fase para x e y (pode separar depois)
    phix = phase_in; phiy = phase_in;
end

%% =============================== [C] PICK ================================
if DO_PICK
    fprintf('\n[C] Casando fase com biblioteca...\n');
    if ~exist('phix','var'), S=load(fullfile(PROC_DIR,'phase_map.mat')); phix=S.phase_in; phiy=S.phase_in; end
    if ~isfile(LIB_CSV), error('Falta biblioteca %s — rode [A].', LIB_CSV); end

    lib = readmatrix(LIB_CSV);
    if size(lib,2) < 13, error('Formato inesperado do meta_library.csv'); end
    tx = lib(:,6) + 1i*lib(:,7);
    ty = lib(:,8) + 1i*lib(:,9);

    idx = pick_meta_atoms_local(phix, phiy, tx, ty);
    save(fullfile(PROC_DIR,'chosen_indices.mat'),'idx','lib');

    % Visual
    nlib = size(lib,1);
    if nlib==1
        figure; imagesc(idx); axis image off; colormap(gca,lines(1)); colorbar;
        title('Índice do meta-átomo escolhido (biblioteca com 1 elemento)');
        fprintf('  [AVISO] Biblioteca tem 1 meta-átomo; todos os pixels usam o mesmo índice.\n');
    else
        uidx = unique(idx(:));
        cmap = parula(max(uidx));
        figure; imagesc(idx); axis image off; colormap(gca,cmap); colorbar;
        title(sprintf('Índice do meta-átomo (|lib|=%d, únicos=%d)', nlib, numel(uidx)));
    end

    % Relatório rápido da biblioteca
    Exx = tx; Eyy = ty;
    cover_x = coverage_deg(angle(Exx));  cover_y = coverage_deg(angle(Eyy));
    medEx = median(abs(Exx)); medEy = median(abs(Eyy));
    msg = sprintf(['Biblioteca: %d meta-atomos | Cobertura fase (Exx)=%.0f%%  (Eyy)=%.0f%% | ' ...
                   'Mediana |Exx|=%.2f  |Eyy|=%.2f\n'], nlib, cover_x, cover_y, medEx, medEy);
    fprintf(['  => ' msg]);
    fid=fopen(REPORT_TXT,'a'); if fid>0, fprintf(fid,'%s',msg); fclose(fid); end
end

fprintf('\n✅ Concluído.\n');

% ============================ FUNÇÕES LOCAIS ==============================
function c = read_cst_scalar_complex(fname)
    % Lê arquivo ASCII de curva do CST e retorna o último ponto como complexo.
    raw = fileread(fname); raw = strrep(raw, ',', '.');
    L = regexp(raw,'\r\n|\n','split'); data = [];
    for i = 1:numel(L)
        t = strtrim(L{i});
        if isempty(t), continue; end
        if startsWith(t,{'#','!','%','Freq','frequency'},'IgnoreCase',true), continue; end
        nums = textscan(t, '%f');
        if ~isempty(nums{1}), data = [data; nums{1}(:)']; end %#ok<AGROW>
    end
    if isempty(data), error('Sem dados numéricos em %s', fname); end
    v = data(end,:);
    % Heurística: [Mag,PhaseDeg] OU [Re,Im] no final
    if numel(v)==2
        c = v(1) + 1i*v(2);
    elseif numel(v)>=3 && abs(v(end-1))<=2 && abs(v(end))<=2
        % última dupla parece |Re|,|Im| (<=2 ~ coerente para S)
        c = v(end-1) + 1i*v(end);
    else
        Mag = v(end-1); Ph = v(end);
        if Mag>=0 && abs(Ph)<=360
            c = Mag.*exp(1i*deg2rad(Ph));
        else
            c = v(end-1) + 1i*v(end);
        end
    end
end

function Uz = propagateASM_local(U0, dx, dy, lambda, z, NA)
    % ASM com supressão de evanescentes e corte por NA
    [Ny, Nx] = size(U0); k = 2*pi/lambda;
    fx = (-Nx/2:Nx/2-1) / (Nx*dx); fy = (-Ny/2:Ny/2-1) / (Ny*dy);
    [FX, FY] = meshgrid(fx, fy); f2 = FX.^2 + FY.^2;
    kz  = k * sqrt(max(0, 1 - (lambda^2)*f2));
    H   = exp(1i*z.*kz) .* (f2 <= (NA/lambda)^2);
    U0f = fftshift( fft2( ifftshift(U0) ) );
    Uzf = U0f .* H;
    Uz  = fftshift( ifft2( ifftshift(Uzf) ) );
end

function idx = pick_meta_atoms_local(phix, phiy, tx, ty)
    % Casamento por menor erro quadrático entre alvos unitários e respostas tx/ty
    [Ny,Nx] = size(phix);  Tx = exp(1i*phix);  Ty = exp(1i*phiy);
    idx = zeros(Ny,Nx);
    for j = 1:Ny
        for i = 1:Nx
            err = abs(tx - Tx(j,i)).^2 + abs(ty - Ty(j,i)).^2;
            [~,best] = min(err);
            idx(j,i) = best;
        end
    end
end

function covp = coverage_deg(phi)
    % cobertura aproximada em graus: % de bins (18 bins de 20°) ocupados
    edges = linspace(-pi,pi,19);
    h = histcounts(phi,edges);
    covp = 100*nnz(h)/numel(h);
end
