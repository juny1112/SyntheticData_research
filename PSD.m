%% ======================================================================
%  Driving load(Current) → PSD (Welch, Hann) → 주요 주파수 & τ 요약 + 리포팅
%  - 입력(엑셀): 1열=시간[s], 2열=전류[A]
%  - τ = 1/(2πf) [s]
%
%  (INTEGRATED)
%   1) 0~t_max 구간 crop 옵션 ([]면 전체)
%   2) Welch window를 '초' 기준으로 고정 (비교 목적에 타당)
%   3) PSD mean/std 출력 + figure title에 표시
%   4) 콘솔 요약 + summary_tbl(long) + Tau_wide_tbl(wide)
%   5) PSD 플랏: xlim 제한 가능 + x축 log 스케일 토글
%   6) (NEW) psd_stat_tbl: Load | mean | std ( + fs, dt ) 테이블 생성
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

% 저장경로
save_dir = "G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\PSD";
if ~exist(save_dir,'dir'); mkdir(save_dir); end

%% 사용자 설정 ------------------------------------------------------------
t_max    = 600;          % [s] (비교 목적이면 고정 추천)  []면 전체 사용
peak_num = 0;            % 피크 개수 (0이면 피크/τ 테이블은 비어있음)

% PSD plot 옵션
xlim_psd = [];           % 예: [1e-4 0.2]  / 자동이면 []
xlog_psd = true;         % x축 로그 스케일 on/off
ylog_psd = false;        % y축 로그 스케일(원하면 true)
use_dB   = false;        % true면 10*log10(PSD)로 표시(dB/Hz)

% Welch PSD 설정(비교용으로 '초' 기준 고정 추천)
welch_win_sec = 600;      % [s]
welch_ovlp    = 0.5;     % overlap 비율 (0~1)
nfft_mode     = "byWin"; % "byWin"=2^nextpow2(nWin) (추천) / "byData"=2^nextpow2(min(N,2^16))

% 피크 탐색 prominence
prom_ratio = 0.02;       % 0.01~0.05 추천

% DC 제외(권장)
exclude_dc_for_stats = true;
exclude_dc_for_peaks = true;

%% 결과 저장용 ------------------------------------------------------------
all_peak_tbl  = struct;   % 파일별 피크 테이블
all_psd_stat  = struct;   % 파일별 PSD mean/std, fs/dt

summary_tbl = table();    % long 형태 (피크 있을 때만 채워짐)

nLoad = numel(driving_files);
Tau_mat = nan(peak_num, nLoad);
load_labels = strings(nLoad,1);

% (NEW) Load | mean | std 테이블
psd_stat_tbl = table('Size',[nLoad 5], ...
    'VariableTypes', {'string','double','double','double','double'}, ...
    'VariableNames', {'Load','fs_Hz','dt_s','PSD_mean_A2Hz','PSD_std_A2Hz'});

%% 메인 루프 --------------------------------------------------------------
for fileIdx = 1:nLoad

    %% (A) 데이터 읽기 ---------------------------------------------------
    filename = driving_files{fileIdx};
    data = readtable(filename, 'VariableNamingRule','preserve');
    t_vec = data{:,1};
    I_vec = data{:,2};

    [~, nm, ~] = fileparts(char(filename));
    load_label = string(nm);
    load_labels(fileIdx) = load_label;

    % NaN 제거
    mask = ~(isnan(t_vec) | isnan(I_vec));
    t_vec = t_vec(mask);
    I_vec = I_vec(mask);

    %% (B) 구간 crop (0~t_max) -------------------------------------------
    if ~isempty(t_max)
        t0 = t_vec(1);
        maskT = (t_vec >= t0) & (t_vec <= t0 + t_max);
        t_vec = t_vec(maskT);
        I_vec = I_vec(maskT);
    end

    if numel(t_vec) < 10
        warning('데이터가 너무 짧습니다: %s', filename);
        all_peak_tbl.(sprintf('file%d',fileIdx)) = table();
        continue;
    end

    %% (C) 샘플링 정보 ---------------------------------------------------
    dt = median(diff(t_vec));
    fs = 1/dt;

    % DC 제거
    I0 = I_vec - mean(I_vec);

    %% (D) Welch PSD -----------------------------------------------------
    N = numel(I0);

    nWin = max(16, round(welch_win_sec * fs));
    nWin = min(nWin, N);
    win = hann(nWin,'periodic');
    noverlap = round(welch_ovlp * nWin);

    switch nfft_mode
        case "byWin"
            nfft = max(256, 2^nextpow2(nWin));
        case "byData"
            nfft = 2^nextpow2(min(N, 2^16));
        otherwise
            error('nfft_mode must be "byWin" or "byData"');
    end

    [Pxx, f] = pwelch(I0, win, noverlap, nfft, fs, 'onesided'); % A^2/Hz

    % PSD mean/std (기본: f>0 bins 기준)
    if exclude_dc_for_stats
        P_stat = Pxx(f>0);
    else
        P_stat = Pxx;
    end

    psd_mean = mean(P_stat);
    psd_std  = std(P_stat);

    % (NEW) psd_stat_tbl 채우기
    psd_stat_tbl.Load(fileIdx) = load_label;
    psd_stat_tbl.fs_Hz(fileIdx) = fs;
    psd_stat_tbl.dt_s(fileIdx) = dt;
    psd_stat_tbl.PSD_mean_A2Hz(fileIdx) = psd_mean;
    psd_stat_tbl.PSD_std_A2Hz(fileIdx)  = psd_std;

    all_psd_stat.(sprintf('file%d',fileIdx)) = struct( ...
        'fs',fs,'dt',dt,'psd_mean',psd_mean,'psd_std',psd_std);

    % 표시용 y
    if use_dB
        yplot = 10*log10(Pxx + realmin);
        ylab  = 'PSD (dB/Hz)';
    else
        yplot = Pxx;
        ylab  = 'PSD (A^2/Hz)';
    end

    %% (E) 피크 탐색 -----------------------------------------------------
    % peak_num=0이면 피크 탐색/리포팅을 스킵 (경고 방지)
    if peak_num <= 0
        peak_tbl = table();
        all_peak_tbl.(sprintf('file%d',fileIdx)) = peak_tbl;

        % (F) 시각화는 계속 수행 (PSD/mean/std만 보고 싶을 수 있으니)
        fig_title = sprintf('[%d] %s', fileIdx, filename);
        figure('Name', fig_title, 'Position', [100 100 920 540]);

        subplot(2,1,1);
        plot(t_vec, I_vec, 'LineWidth', 1.1); grid on;
        xlabel('Time (s)'); ylabel('Current (A)');
        title(sprintf('%s Current', load_label), 'Interpreter','none');

        subplot(2,1,2);
        mPlot = true(size(f));
        if exclude_dc_for_stats || exclude_dc_for_peaks
            mPlot = (f>0);
        end
        plot(f(mPlot), yplot(mPlot), 'LineWidth', 1.2); grid on;
        xlabel('Frequency (Hz)'); ylabel(ylab);
        if xlog_psd, set(gca,'XScale','log'); end
        if ylog_psd, set(gca,'YScale','log'); end
        if ~isempty(xlim_psd), xlim(xlim_psd); end
        title(sprintf('PSD (Welch, Hann) | mean=%.3g, std=%.3g', psd_mean, psd_std));
        legend('PSD (Welch)','Location','best');

        continue;
    end

    if exclude_dc_for_peaks
        f_pk = f(f>0);
        P_pk = Pxx(f>0);
    else
        f_pk = f;
        P_pk = Pxx;
    end

    if all(~isfinite(P_pk)) || max(P_pk)<=0
        pks = []; locs = [];
    else
        [pks, locs] = findpeaks(P_pk, f_pk, ...
            'MinPeakProminence', prom_ratio*max(P_pk), ...
            'SortStr','descend');
    end

    nShow = min(peak_num, numel(pks));

    if nShow == 0
        peak_tbl = table();
        warning('피크를 찾지 못했습니다: %s', filename);
        locs_col = []; pks_col = []; tau_col = []; psd_db_col = [];
    else
        idx = 1:nShow;
        locs_col = locs(idx); locs_col = locs_col(:);
        pks_col  = pks(idx);  pks_col  = pks_col(:);
        tau_col  = 1./(2*pi*locs_col);
        psd_db_col = 10*log10(pks_col + realmin);

        peak_tbl = table(locs_col, tau_col, pks_col, psd_db_col, ...
            'VariableNames', {'Freq_Hz','Tau_s','PSD_A2Hz','PSD_dBHz'});
    end

    all_peak_tbl.(sprintf('file%d',fileIdx)) = peak_tbl;

    % summary_tbl(long) 누적
    if ~isempty(peak_tbl)
        tmp = peak_tbl;
        tmp.Load = repmat(load_label, height(tmp), 1);
        tmp.PeakRank = (1:height(tmp))';
        tmp = movevars(tmp, {'Load','PeakRank'}, 'Before', 1);
        summary_tbl = [summary_tbl; tmp]; %#ok<AGROW>
    end

    % Tau wide 채우기
    if nShow > 0
        Tau_mat(1:nShow, fileIdx) = tau_col;
    end

    %% (F) 시각화 --------------------------------------------------------
    fig_title = sprintf('[%d] %s', fileIdx, filename);
    figure('Name', fig_title, 'Position', [100 100 920 540]);

    % 시간 영역
    subplot(2,1,1);
    plot(t_vec, I_vec, 'LineWidth', 1.1); grid on;
    xlabel('Time (s)'); ylabel('Current (A)');
    title(sprintf('%s Current', load_label), 'Interpreter','none');

    % PSD
    subplot(2,1,2);
    mPlot = true(size(f));
    if exclude_dc_for_stats || exclude_dc_for_peaks
        mPlot = (f>0);
    end

    plot(f(mPlot), yplot(mPlot), 'LineWidth', 1.2); hold on; grid on;
    xlabel('Frequency (Hz)'); ylabel(ylab);

    if xlog_psd, set(gca,'XScale','log'); end
    if ylog_psd, set(gca,'YScale','log'); end
    if ~isempty(xlim_psd), xlim(xlim_psd); end

    if nShow > 0
        if use_dB
            stem(locs_col, 10*log10(pks_col + realmin), 'r','filled','LineWidth',1.1);
        else
            stem(locs_col, pks_col, 'r','filled','LineWidth',1.1);
        end
        legend('PSD (Welch)','Detected Peaks','Location','best');
    else
        legend('PSD (Welch)','Location','best');
    end

    title(sprintf('PSD (Welch, Hann) | mean=%.3g, std=%.3g', psd_mean, psd_std));

end

%% (G) 콘솔 요약 ---------------------------------------------------------
fprintf('\n=============== 고유 주파수 & τ 요약 (PSD 기반) ===============\n');

for fileIdx = 1:nLoad
    fprintf('\n--- %s ---\n', driving_files{fileIdx});

    if isfield(all_psd_stat, sprintf('file%d',fileIdx))
        st = all_psd_stat.(sprintf('file%d',fileIdx));
        fprintf('fs = %.6g Hz | dt = %.6g s\n', st.fs, st.dt);
        fprintf('PSD mean = %.6g (A^2/Hz) | PSD std = %.6g (A^2/Hz)\n', st.psd_mean, st.psd_std);
    end

    disp(all_peak_tbl.(sprintf('file%d', fileIdx)));
end

%% (H) 테이블 생성 --------------------------------------------------------
if ~isempty(summary_tbl)
    summary_tbl = sortrows(summary_tbl, {'Load','PeakRank'});
end

vnames = matlab.lang.makeValidName(cellstr(load_labels));
vnames = matlab.lang.makeUniqueStrings(vnames);

Tau_wide_tbl = array2table(Tau_mat, 'VariableNames', vnames);
Tau_wide_tbl.PeakRank = (1:peak_num)';
Tau_wide_tbl = movevars(Tau_wide_tbl, 'PeakRank', 'Before', 1);

%% 확인용 출력 ------------------------------------------------------------
disp(" ");
disp("==== psd_stat_tbl (Load | mean | std) ====");
disp(psd_stat_tbl);

disp(" ");
disp("==== summary_tbl (PSD long) 미리보기 ====");
if isempty(summary_tbl)
    disp("(empty: peak_num=0 이거나 피크가 없어서 long 테이블이 생성되지 않았습니다.)");
else
    disp(head(summary_tbl, 20));
end

disp(" ");
disp("==== Tau_wide_tbl (PeakRank x Loads) ====");
disp(Tau_wide_tbl);

% 파일 저장
% writetable(psd_stat_tbl, fullfile(save_dir, "psd_stat_tbl.csv"));

% (옵션) workspace 재사용용 MAT 저장
save(fullfile(save_dir, "psd_stat_tbl.mat"), "psd_stat_tbl");

fprintf("[done] saved psd_stat_tbl to: %s\n", save_dir);
