%% ======================================================================
%  PSD_feature.m
%  Driving load(Current) → PSD (Welch, Hann) + tau-band energy features
%
%  - 입력(엑셀): 1열=시간[s], 2열=전류[A]
%  - PSD: pwelch(I0, hann, overlap, nfft, fs, 'onesided')  → Pxx [A^2/Hz]
%
%  (FEATURES)
%    PSD_mean_A2Hz        = mean(Pxx)  (DC 제외 여부 설정)
%    PSD_std_A2Hz         = std(Pxx)
%    PSD_int_A2           = ∫ Pxx df   (A^2)  ~ 전류변동 분산 스케일(DC 제외 시)
%    E_fast_A2            = ∫_{f_fast} Pxx df
%    E_slow_A2            = ∫_{f_slow} Pxx df
%    E_slow_fast_ratio    = E_slow / E_fast
%
%  (PSD-weighted mean/std in frequency domain)
%    f_mean_Hz            = sum(f*P) / sum(P)
%    f_std_Hz             = sqrt( sum((f-f_mean)^2 * P) / sum(P) )
%    tau_mean_s           = 1/(2*pi*f_mean_Hz)
%
%  (NEW) band-wise PSD-weighted mean frequencies
%    fast_mean_Hz, slow_mean_Hz
%    tau_fast_mean_s, tau_slow_mean_s
%
%  Plot:
%    - FAST band shading: green
%    - SLOW band shading: yellow
%    - f_mean_Hz line: red dashed
%    - fast_mean_Hz line: blue dashed
%    - slow_mean_Hz line: orange dashed
%    - (REQUEST) band edge lines removed
%    - (REQUEST) info textbox removed
%
%  저장:
%    - psd_stat_tbl.csv
%    - psd_stat_tbl.mat
%% ======================================================================
clear; clc; close all;

%% 0) 분석할 파일 목록 -----------------------------------------------------
driving_files = {
    "G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\us06_0725.xlsx"
    "G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\udds_0725.xlsx"
    "G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\hwfet_0725.xlsx"
    "G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\WLTP_0725.xlsx"
    "G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\BSL_CITY1_0725.xlsx"
    "G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\BSL_CITY2_0726.xlsx"
    "G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\BSL_HW1_0725.xlsx"
    "G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\BSL_HW2_0725.xlsx"
};

%% 저장경로 ---------------------------------------------------------------
save_dir = "G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\PSD";
if ~exist(save_dir,'dir'); mkdir(save_dir); end

%% 사용자 설정 ------------------------------------------------------------
t_max    = 600;          % [s] []면 전체, 지금은 600초 고정
xlim_psd = [];           % 예: [1e-4 0.2] / 자동이면 []
xlog_psd = true;         % x축 로그 on/off
ylog_psd = false;        % y축 로그 on/off
use_dB   = false;        % true면 10*log10(PSD)로 표시(dB/Hz)
exclude_dc_for_stats = true;

% Welch PSD 설정
welch_win_sec = 600;     % [s]
welch_ovlp    = 0.5;     % overlap 비율 (0~1)

% ===== (옵션) nfft 강제 =====
force_nfft = true;
nfft_fixed = 4096;       % 4096 추천 (저주파 bin 촘촘). 더 촘촘=8192/16384.

% DataTip(클릭) 옵션: false면 켬
turn_off_datacursor = false;

%% ===== tau-대역 에너지 feature 설정 (band는 그대로 유지) =====
tau_fast_range = [1 20];       % [s]
tau_slow_range = [20 400];     % [s]

% tau [s] -> f [Hz] : f = 1/(2*pi*tau)
f_fast = sort(1./(2*pi*tau_fast_range));   % [Hz]
f_slow = sort(1./(2*pi*tau_slow_range));   % [Hz]

fprintf("=== tau-band 설정 ===\n");
fprintf("fast tau=[%.2g %.2g] s -> f=[%.6g %.6g] Hz\n", tau_fast_range(1), tau_fast_range(2), f_fast(1), f_fast(2));
fprintf("slow tau=[%.2g %.2g] s -> f=[%.6g %.6g] Hz\n\n", tau_slow_range(1), tau_slow_range(2), f_slow(1), f_slow(2));

%% 결과 테이블 ------------------------------------------------------------
nLoad = numel(driving_files);

% (REQUEST) tau_low_s / tau_high_s 제거
psd_stat_tbl = table('Size',[nLoad 16], ...
    'VariableTypes', {'string','double','double', ...
                      'double','double','double','double','double','double', ...
                      'double','double','double', ...
                      'double','double','double','double'}, ...
    'VariableNames', {'Load','fs_Hz','dt_s', ...
                      'PSD_mean_A2Hz','PSD_std_A2Hz','PSD_int_A2','E_fast_A2','E_slow_A2','E_slow_fast_ratio', ...
                      'f_mean_Hz','f_std_Hz','tau_mean_s', ...
                      'fast_mean_Hz','slow_mean_Hz','tau_fast_mean_s','tau_slow_mean_s'});

all_psd_stat = struct;

%% 메인 루프 --------------------------------------------------------------
for fileIdx = 1:nLoad

    %% (A) 데이터 읽기
    filename = driving_files{fileIdx};
    data = readtable(filename, 'VariableNamingRule','preserve');
    t_vec = data{:,1};
    I_vec = data{:,2};

    [~, nm, ~] = fileparts(char(filename));

    % ==================================================================
    % (NEW) load_label 표준화: 반드시 {"US06","UDDS","HWFET","WLTP","CITY1","CITY2","HW1","HW2"} 중 하나로 저장
    % ==================================================================
    load_label = standardizeLoadName(nm);  % <- 아래 helper function 사용
    % ==================================================================

    % (NEW) Load 문자열 클린업: 따옴표/공백 제거 (CSV/엑셀에서 "US06" 형태로 들어오는 케이스 방지)
    load_label = erase(load_label, '"');    % 큰따옴표 제거
    load_label = erase(load_label, '''');   % 작은따옴표 제거(혹시)
    load_label = strtrim(load_label);       % 앞뒤 공백 제거

    mask = ~(isnan(t_vec) | isnan(I_vec));
    t_vec = t_vec(mask);
    I_vec = I_vec(mask);

    % crop
    if ~isempty(t_max)
        t0 = t_vec(1);
        maskT = (t_vec >= t0) & (t_vec <= t0 + t_max);
        t_vec = t_vec(maskT);
        I_vec = I_vec(maskT);
    end

    if numel(t_vec) < 10
        warning('데이터가 너무 짧습니다: %s', filename);
        continue;
    end

    %% (B) 샘플링 정보
    dt = median(diff(t_vec));
    fs = 1/dt;

    %% (C) DC 제거
    I0 = I_vec - mean(I_vec);

    %% (D) Welch PSD
    N = numel(I0);

    nWin = max(16, round(welch_win_sec * fs));
    nWin = min(nWin, N);
    win = hann(nWin,'periodic');
    noverlap = round(welch_ovlp * nWin);

    if force_nfft
        nfft = nfft_fixed;
    else
        nfft = max(256, 2^nextpow2(nWin));
    end

    [Pxx, f] = pwelch(I0, win, noverlap, nfft, fs, 'onesided'); % A^2/Hz

    % --- stats용 벡터 (DC 제외 여부 반영) ---
    if exclude_dc_for_stats
        mStat = (f > 0);
    else
        mStat = true(size(f));
    end
    f_stat = f(mStat);
    P_stat = Pxx(mStat);

    %% (E) bin 평균 mean/std
    psd_mean = mean(P_stat);
    psd_std  = std(P_stat);

    %% (F) 분산 스케일 적분 + band energy
    PSD_int = trapz(f_stat, P_stat); % A^2

    m_fast = (f_stat >= f_fast(1)) & (f_stat <= f_fast(2));
    m_slow = (f_stat >= f_slow(1)) & (f_stat <= f_slow(2));

    if nnz(m_fast) < 2
        E_fast = NaN;
    else
        E_fast = trapz(f_stat(m_fast), P_stat(m_fast));
    end

    if nnz(m_slow) < 2
        E_slow = NaN;
    else
        E_slow = trapz(f_stat(m_slow), P_stat(m_slow));
    end

    ratio_slow_fast = E_slow / E_fast;

    %% (G) PSD-weighted mean/std in frequency (전체 대역)
    wSum = sum(P_stat);
    if wSum <= 0 || ~isfinite(wSum)
        f_mean = NaN; f_std = NaN;
    else
        f_mean = sum(f_stat .* P_stat) / wSum;
        f_var  = sum(((f_stat - f_mean).^2) .* P_stat) / wSum;
        f_std  = sqrt(max(f_var, 0));
    end

    if isfinite(f_mean) && f_mean > 0
        tau_mean = 1/(2*pi*f_mean);
    else
        tau_mean = NaN;
    end

    %% (G-2) band별 PSD-weighted mean frequency (fast/slow)
    fast_mean = NaN; slow_mean = NaN;
    tau_fast_mean = NaN; tau_slow_mean = NaN;

    if nnz(m_fast) >= 2
        wf = sum(P_stat(m_fast));
        if wf > 0 && isfinite(wf)
            fast_mean = sum(f_stat(m_fast).*P_stat(m_fast)) / wf;
            if fast_mean > 0
                tau_fast_mean = 1/(2*pi*fast_mean);
            end
        end
    end

    if nnz(m_slow) >= 2
        ws = sum(P_stat(m_slow));
        if ws > 0 && isfinite(ws)
            slow_mean = sum(f_stat(m_slow).*P_stat(m_slow)) / ws;
            if slow_mean > 0
                tau_slow_mean = 1/(2*pi*slow_mean);
            end
        end
    end

    %% (H) 테이블 채우기
    psd_stat_tbl.Load(fileIdx)              = load_label;
    psd_stat_tbl.fs_Hz(fileIdx)             = fs;
    psd_stat_tbl.dt_s(fileIdx)              = dt;
    psd_stat_tbl.PSD_mean_A2Hz(fileIdx)     = psd_mean;
    psd_stat_tbl.PSD_std_A2Hz(fileIdx)      = psd_std;
    psd_stat_tbl.PSD_int_A2(fileIdx)        = PSD_int;
    psd_stat_tbl.E_fast_A2(fileIdx)         = E_fast;
    psd_stat_tbl.E_slow_A2(fileIdx)         = E_slow;
    psd_stat_tbl.E_slow_fast_ratio(fileIdx) = ratio_slow_fast;

    psd_stat_tbl.f_mean_Hz(fileIdx)         = f_mean;
    psd_stat_tbl.f_std_Hz(fileIdx)          = f_std;
    psd_stat_tbl.tau_mean_s(fileIdx)        = tau_mean;

    psd_stat_tbl.fast_mean_Hz(fileIdx)      = fast_mean;
    psd_stat_tbl.slow_mean_Hz(fileIdx)      = slow_mean;
    psd_stat_tbl.tau_fast_mean_s(fileIdx)   = tau_fast_mean;
    psd_stat_tbl.tau_slow_mean_s(fileIdx)   = tau_slow_mean;

    all_psd_stat.(sprintf('file%d',fileIdx)) = struct( ...
        'Load',load_label,'fs',fs,'dt',dt,'nfft',nfft,'nWin',nWin, ...
        'psd_mean',psd_mean,'psd_std',psd_std,'PSD_int',PSD_int, ...
        'E_fast',E_fast,'E_slow',E_slow,'ratio',ratio_slow_fast, ...
        'f_mean',f_mean,'f_std',f_std,'tau_mean',tau_mean, ...
        'fast_mean',fast_mean,'slow_mean',slow_mean, ...
        'tau_fast_mean',tau_fast_mean,'tau_slow_mean',tau_slow_mean, ...
        'tau_fast_range',tau_fast_range,'tau_slow_range',tau_slow_range,'f_fast',f_fast,'f_slow',f_slow);

    %% 콘솔 출력 (요청: Hz + tau 둘 다)
    df = fs/nfft;
    fmin_pos = df; % 첫 양의 bin
    tau_max_effective = 1/(2*pi*fmin_pos);

    fprintf('--- %s ---\n', load_label);
    fprintf('fs=%.6g Hz, dt=%.6g s, nWin=%d, nfft=%d -> df=%.9g Hz, fmin=%.9g Hz, tau_max≈%.1f s\n', ...
        fs, dt, nWin, nfft, df, fmin_pos, tau_max_effective);
    fprintf('PSD(mean)=%.6g A^2/Hz, PSD(std)=%.6g A^2/Hz, PSD(int)=%.6g A^2\n', psd_mean, psd_std, PSD_int);
    fprintf('E_fast=%.6g A^2, E_slow=%.6g A^2, ratio(Eslow/Efast)=%.6g\n', E_fast, E_slow, ratio_slow_fast);
    fprintf('Total weighted mean f=%.6g Hz (tau=%.3g s), std f=%.6g Hz\n', f_mean, tau_mean, f_std);
    fprintf('FAST mean f=%.6g Hz (tau=%.3g s), SLOW mean f=%.6g Hz (tau=%.3g s)\n\n', ...
        fast_mean, tau_fast_mean, slow_mean, tau_slow_mean);

    %% (I) Plot 준비
    if use_dB
        yplot = 10*log10(Pxx + realmin);
        ylab  = 'PSD (dB/Hz)';
    else
        yplot = Pxx;
        ylab  = 'PSD (A^2/Hz)';
    end

    mPlot = true(size(f));
    if exclude_dc_for_stats
        mPlot = (f > 0);
    end

    %% (J) Figure 1개/파일
    fig_title = sprintf('[%d] %s', fileIdx, filename);
    fig = figure('Name', fig_title, 'Position', [100 100 980 560], 'Color','w');

    if turn_off_datacursor
        datacursormode(fig, 'off');
    else
        datacursormode(fig, 'on'); % 클릭하면 DataTip 뜨게
    end

    % time domain
    subplot(2,1,1);
    plot(t_vec, I_vec, 'LineWidth', 1.1); grid on;
    xlabel('Time (s)'); ylabel('Current (A)');
    title(sprintf('%s Current (0~%ds)', load_label, round(t_max)), 'Interpreter','none');

    % PSD
    ax = subplot(2,1,2);
    hP = plot(f(mPlot), yplot(mPlot), 'LineWidth', 1.2); grid on; hold on;
    xlabel('Frequency (Hz)'); ylabel(ylab);

    if xlog_psd, set(ax,'XScale','log'); end
    if ylog_psd, set(ax,'YScale','log'); end
    if ~isempty(xlim_psd), xlim(ax, xlim_psd); end

    yl = ylim(ax);
    xL = xlim(ax);

    % --- band shading (FAST=green, SLOW=yellow) ---
    % clip to axes limits (band 자체는 유지)
    f_fast_clip = [max(f_fast(1), xL(1)), min(f_fast(2), xL(2))];
    f_slow_clip = [max(f_slow(1), xL(1)), min(f_slow(2), xL(2))];

    hFastPatch = gobjects(1);
    hSlowPatch = gobjects(1);

    if f_fast_clip(2) > f_fast_clip(1)
        hFastPatch = patch(ax, ...
            [f_fast_clip(1) f_fast_clip(2) f_fast_clip(2) f_fast_clip(1)], ...
            [yl(1) yl(1) yl(2) yl(2)], ...
            [0 1 0], 'FaceAlpha', 0.10, 'EdgeColor', 'none'); % green
    end

    if f_slow_clip(2) > f_slow_clip(1)
        hSlowPatch = patch(ax, ...
            [f_slow_clip(1) f_slow_clip(2) f_slow_clip(2) f_slow_clip(1)], ...
            [yl(1) yl(1) yl(2) yl(2)], ...
            [1 1 0], 'FaceAlpha', 0.10, 'EdgeColor', 'none'); % yellow
    end

    % (REQUEST) band edge lines 제거: 그리는 코드 없음

    % --- mean lines ---
    hMeanLine = gobjects(1);
    if isfinite(f_mean) && f_mean > 0
        hMeanLine = plot(ax, [f_mean f_mean], yl, 'r--', 'LineWidth', 1.4);
    end

    hFastMeanLine = gobjects(1);
    if isfinite(fast_mean) && fast_mean > 0
        hFastMeanLine = plot(ax, [fast_mean fast_mean], yl, '--', 'LineWidth', 1.4);
        hFastMeanLine.Color = [0 0.4470 0.7410];  % blue
    end

    hSlowMeanLine = gobjects(1);
    if isfinite(slow_mean) && slow_mean > 0
        hSlowMeanLine = plot(ax, [slow_mean slow_mean], yl, '--', 'LineWidth', 1.4);
        hSlowMeanLine.Color = [0.8500 0.3250 0.0980]; % orange
    end

    ylim(ax, yl);

    title(ax, sprintf('PSD (Welch,Hann) | int=%.3g A^2 | ratio(Eslow/Efast)=%.3g | mean=%.3g std=%.3g (A^2/Hz)', ...
        PSD_int, ratio_slow_fast, psd_mean, psd_std), 'Interpreter','none');

    % legend (의미있는 라벨만)
    legH = [hP];
    legL = ["PSD (Welch)"];

    if isgraphics(hFastPatch)
        legH(end+1) = hFastPatch; %#ok<AGROW>
        legL(end+1) = sprintf("FAST band (tau=%g~%gs)", tau_fast_range(1), tau_fast_range(2));
    end
    if isgraphics(hSlowPatch)
        legH(end+1) = hSlowPatch; %#ok<AGROW>
        legL(end+1) = sprintf("SLOW band (tau=%g~%gs)", tau_slow_range(1), tau_slow_range(2));
    end
    if isgraphics(hMeanLine)
        legH(end+1) = hMeanLine; %#ok<AGROW>
        legL(end+1) = "Total mean f (red --)";
    end
    if isgraphics(hFastMeanLine)
        legH(end+1) = hFastMeanLine; %#ok<AGROW>
        legL(end+1) = "FAST mean f (blue --)";
    end
    if isgraphics(hSlowMeanLine)
        legH(end+1) = hSlowMeanLine; %#ok<AGROW>
        legL(end+1) = "SLOW mean f (orange --)";
    end

    legend(ax, legH, legL, 'Location','northeast');
end

%% 요약 출력 --------------------------------------------------------------
fprintf('\n=============== PSD 요약 테이블 ===============\n');
disp(psd_stat_tbl);

% fprintf("PSD Load unique = %s\n", strjoin(unique(psd_stat_tbl.Load), ", "));

%% 저장 -------------------------------------------------------------------
writetable(psd_stat_tbl, fullfile(save_dir, "psd_stat_tbl.csv"));
save(fullfile(save_dir, "psd_stat_tbl.mat"), ...
    "psd_stat_tbl", "all_psd_stat", ...
    "tau_fast_range", "tau_slow_range", "f_fast", "f_slow", ...
    "welch_win_sec", "welch_ovlp", "force_nfft", "nfft_fixed", ...
    "exclude_dc_for_stats", "t_max", "use_dB", "xlog_psd", "ylog_psd");

fprintf("[done] saved psd_stat_tbl(.csv/.mat) to: %s\n", save_dir);

%% ========================= helper function =============================
function L = standardizeLoadName(nm)
% nm: file basename without extension (예: "BSL_CITY1_0725", "us06_0725")
% return: {"US06","UDDS","HWFET","WLTP","CITY1","CITY2","HW1","HW2"} 중 하나

s = upper(string(nm));

% 흔한 구분자/접두어 정리
s = erase(s, "BSL_");
s = erase(s, "BSL");
s = strrep(s, "-", "_");
s = strrep(s, "__", "_");

% 핵심 키워드로 매핑
if contains(s, "US06")
    L = "US06";
elseif contains(s, "UDDS")
    L = "UDDS";
elseif contains(s, "HWFET")
    L = "HWFET";
elseif contains(s, "WLTP")
    L = "WLTP";
elseif contains(s, "CITY1")
    L = "CITY1";
elseif contains(s, "CITY2")
    L = "CITY2";
elseif contains(s, "HW1")
    L = "HW1";
elseif contains(s, "HW2")
    L = "HW2";
else
    warning("표준 load 매핑 실패: nm=%s → LoadStd=UNKNOWN", nm);
    L = "UNKNOWN";
end
end