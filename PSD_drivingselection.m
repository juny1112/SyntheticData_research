%% ======================================================================
%  PSD_feature_selected12_full.m
%  selected_additional_12.csv 에서 선택된 부하들의 current_profile(csv)을 읽어
%  - Current profile subplot
%  - PSD subplot
%  - PSD statistics table 저장
%
%  원본 전체 길이(full-length) 기준으로 분석
%% ======================================================================
clear; clc; close all;

%% 0) 경로 설정 -----------------------------------------------------------
selected_csv = "G:\공유 드라이브\Battery Software Group (2025)\Members\김주은\주행부하\selected_additional_12.csv";
current_profile_dir = "\\BSL\Shared_Drive\SamsungSTF\Processed_Data\KOTI\current_profile";
save_dir = "G:\공유 드라이브\Battery Software Group (2025)\Members\김주은\주행부하\PSD_selected12_full";

if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

%% 1) selected 목록 읽기 --------------------------------------------------
sel_tbl = readtable(selected_csv, 'TextType', 'string', 'VariableNamingRule', 'preserve');

if ~ismember("file_name", string(sel_tbl.Properties.VariableNames))
    error("'file_name' column not found in selected_additional_12.csv");
end

file_names = string(sel_tbl.file_name);
nLoad = numel(file_names);

driving_files = strings(nLoad, 1);
load_labels = strings(nLoad, 1);

for i = 1:nLoad
    driving_files(i) = fullfile(current_profile_dir, file_names(i));

    if ismember("raw_base_name", string(sel_tbl.Properties.VariableNames))
        load_labels(i) = string(sel_tbl.raw_base_name(i));
    else
        load_labels(i) = erase(file_names(i), ".csv");
    end
end

%% 2) 사용자 설정 ---------------------------------------------------------
t_max    = [];          % []면 전체 길이 사용
xlim_psd = [];          % 예: [1e-4 0.2]
xlog_psd = true;
ylog_psd = false;
use_dB   = false;
exclude_dc_for_stats = true;

% Welch PSD 설정
welch_win_sec = 600;     % window length만 600 s
welch_ovlp    = 0.5;

% nfft 강제
force_nfft = true;
nfft_fixed = 4096;

%% tau-band 설정 ----------------------------------------------------------
tau_fast_range = [1 20];
tau_slow_range = [20 400];

f_fast = sort(1./(2*pi*tau_fast_range));
f_slow = sort(1./(2*pi*tau_slow_range));

fprintf("=== tau-band 설정 ===\n");
fprintf("fast tau=[%.2g %.2g] s -> f=[%.6g %.6g] Hz\n", tau_fast_range(1), tau_fast_range(2), f_fast(1), f_fast(2));
fprintf("slow tau=[%.2g %.2g] s -> f=[%.6g %.6g] Hz\n\n", tau_slow_range(1), tau_slow_range(2), f_slow(1), f_slow(2));

%% 3) 결과 저장용 ---------------------------------------------------------
psd_rows = struct([]);

nRows = 4;
nCols = 4;

fig_cur = figure('Name', 'Selected Current Profiles (full-length)', ...
    'Position', [100 50 1400 1000], 'Color', 'w');
tl_cur = tiledlayout(fig_cur, nRows, nCols, 'TileSpacing', 'compact', 'Padding', 'compact');

fig_psd = figure('Name', 'Selected PSD (full-length)', ...
    'Position', [160 80 1400 1000], 'Color', 'w');
tl_psd = tiledlayout(fig_psd, nRows, nCols, 'TileSpacing', 'compact', 'Padding', 'compact');

%% 4) 메인 루프 -----------------------------------------------------------
for fileIdx = 1:nLoad

    filename = driving_files(fileIdx);
    load_label = strtrim(load_labels(fileIdx));

    if ~isfile(filename)
        warning("파일이 존재하지 않습니다: %s", filename);
        continue;
    end

    %% (A) 데이터 읽기
    data = readtable(filename, 'VariableNamingRule', 'preserve');
    varNames = string(data.Properties.VariableNames);

    if all(ismember(["time", "scaled_current"], varNames))
        t_vec = data.("time");
        I_vec = data.("scaled_current");
    else
        t_vec = data{:,1};
        I_vec = data{:,2};
    end

    t_vec = double(t_vec);
    I_vec = double(I_vec);

    mask = ~(isnan(t_vec) | isnan(I_vec) | ~isfinite(t_vec) | ~isfinite(I_vec));
    t_vec = t_vec(mask);
    I_vec = I_vec(mask);

    % full-length 사용: crop 없음
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
    win = hann(nWin, 'periodic');
    noverlap = round(welch_ovlp * nWin);

    if force_nfft
        nfft = nfft_fixed;
    else
        nfft = max(256, 2^nextpow2(nWin));
    end

    [Pxx, f] = pwelch(I0, win, noverlap, nfft, fs, 'onesided');

    if exclude_dc_for_stats
        mStat = (f > 0);
    else
        mStat = true(size(f));
    end
    f_stat = f(mStat);
    P_stat = Pxx(mStat);

    %% (E) 통계량
    psd_mean = mean(P_stat);
    psd_std  = std(P_stat);
    PSD_int  = trapz(f_stat, P_stat);

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

    wSum = sum(P_stat);
    if wSum <= 0 || ~isfinite(wSum)
        f_mean = NaN;
        f_std  = NaN;
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

    f_geo_mean = NaN; tau_geo_mean = NaN; f_geo_std_dec = NaN;
    m_log = (f_stat > 0) & isfinite(f_stat) & isfinite(P_stat) & (P_stat >= 0);
    f_log = f_stat(m_log);
    P_log = P_stat(m_log);

    wLog = sum(P_log);
    if numel(f_log) >= 2 && wLog > 0 && isfinite(wLog)
        u_log = log10(f_log);
        mu_log10f = sum(u_log .* P_log) / wLog;
        var_log10f = sum(((u_log - mu_log10f).^2) .* P_log) / wLog;
        f_geo_std_dec = sqrt(max(var_log10f, 0));

        f_geo_mean = 10^(mu_log10f);
        if isfinite(f_geo_mean) && f_geo_mean > 0
            tau_geo_mean = 1/(2*pi*f_geo_mean);
        end
    end

    f_fast_geo_mean = NaN; f_slow_geo_mean = NaN;
    tau_fast_geo_mean = NaN; tau_slow_geo_mean = NaN;

    if nnz(m_fast) >= 2
        f_fast_use = f_stat(m_fast);
        P_fast_use = P_stat(m_fast);
        m_ok = (f_fast_use > 0) & isfinite(f_fast_use) & isfinite(P_fast_use) & (P_fast_use >= 0);
        f_fast_use = f_fast_use(m_ok);
        P_fast_use = P_fast_use(m_ok);

        wFastLog = sum(P_fast_use);
        if numel(f_fast_use) >= 2 && wFastLog > 0 && isfinite(wFastLog)
            mu_fast_log10f = sum(log10(f_fast_use) .* P_fast_use) / wFastLog;
            f_fast_geo_mean = 10^(mu_fast_log10f);
            if isfinite(f_fast_geo_mean) && f_fast_geo_mean > 0
                tau_fast_geo_mean = 1/(2*pi*f_fast_geo_mean);
            end
        end
    end

    if nnz(m_slow) >= 2
        f_slow_use = f_stat(m_slow);
        P_slow_use = P_stat(m_slow);
        m_ok = (f_slow_use > 0) & isfinite(f_slow_use) & isfinite(P_slow_use) & (P_slow_use >= 0);
        f_slow_use = f_slow_use(m_ok);
        P_slow_use = P_slow_use(m_ok);

        wSlowLog = sum(P_slow_use);
        if numel(f_slow_use) >= 2 && wSlowLog > 0 && isfinite(wSlowLog)
            mu_slow_log10f = sum(log10(f_slow_use) .* P_slow_use) / wSlowLog;
            f_slow_geo_mean = 10^(mu_slow_log10f);
            if isfinite(f_slow_geo_mean) && f_slow_geo_mean > 0
                tau_slow_geo_mean = 1/(2*pi*f_slow_geo_mean);
            end
        end
    end

    %% (F) 구조체 저장
    psd_rows(fileIdx).Load = load_label;
    psd_rows(fileIdx).FileName = file_names(fileIdx);
    psd_rows(fileIdx).duration_s = t_vec(end) - t_vec(1);
    psd_rows(fileIdx).fs_Hz = fs;
    psd_rows(fileIdx).dt_s = dt;
    psd_rows(fileIdx).PSD_mean_A2Hz = psd_mean;
    psd_rows(fileIdx).PSD_std_A2Hz = psd_std;
    psd_rows(fileIdx).PSD_int_A2 = PSD_int;
    psd_rows(fileIdx).E_fast_A2 = E_fast;
    psd_rows(fileIdx).E_slow_A2 = E_slow;
    psd_rows(fileIdx).E_slow_fast_ratio = ratio_slow_fast;
    psd_rows(fileIdx).f_mean_Hz = f_mean;
    psd_rows(fileIdx).f_std_Hz = f_std;
    psd_rows(fileIdx).tau_mean_s = tau_mean;
    psd_rows(fileIdx).f_geo_mean_Hz = f_geo_mean;
    psd_rows(fileIdx).tau_geo_mean_s = tau_geo_mean;
    psd_rows(fileIdx).f_geo_std_dec = f_geo_std_dec;
    psd_rows(fileIdx).f_fast_geo_mean_Hz = f_fast_geo_mean;
    psd_rows(fileIdx).tau_fast_geo_mean_s = tau_fast_geo_mean;
    psd_rows(fileIdx).f_slow_geo_mean_Hz = f_slow_geo_mean;
    psd_rows(fileIdx).tau_slow_geo_mean_s = tau_slow_geo_mean;

    %% (G) plot 값 준비
    if use_dB
        yplot = 10*log10(Pxx + realmin);
        ylab = 'PSD (dB/Hz)';
    else
        yplot = Pxx;
        ylab = 'PSD (A^2/Hz)';
    end

    mPlot = true(size(f));
    if exclude_dc_for_stats
        mPlot = (f > 0);
    end

    %% (H) Current subplot
    figure(fig_cur);
    ax1 = nexttile(tl_cur, fileIdx);
    plot(ax1, t_vec, I_vec, 'LineWidth', 1.0);
    grid(ax1, 'on');
    xlabel(ax1, 'Time (s)');
    ylabel(ax1, 'Current (A)');
    title(ax1, sprintf('%d) %s', fileIdx, load_label), 'Interpreter', 'none', 'FontSize', 9);

    %% (I) PSD subplot
    figure(fig_psd);
    ax2 = nexttile(tl_psd, fileIdx);
    plot(ax2, f(mPlot), yplot(mPlot), 'LineWidth', 1.1);
    hold(ax2, 'on');
    grid(ax2, 'on');

    if xlog_psd
        set(ax2, 'XScale', 'log');
    end
    if ylog_psd
        set(ax2, 'YScale', 'log');
    end
    if ~isempty(xlim_psd)
        xlim(ax2, xlim_psd);
    end

    xlabel(ax2, 'Frequency (Hz)');
    ylabel(ax2, ylab);
    title(ax2, sprintf('%d) %s', fileIdx, load_label), 'Interpreter', 'none', 'FontSize', 9);

    yl = ylim(ax2);
    xL = xlim(ax2);

    f_fast_clip = [max(f_fast(1), xL(1)), min(f_fast(2), xL(2))];
    f_slow_clip = [max(f_slow(1), xL(1)), min(f_slow(2), xL(2))];

    if f_fast_clip(2) > f_fast_clip(1)
        patch(ax2, ...
            [f_fast_clip(1) f_fast_clip(2) f_fast_clip(2) f_fast_clip(1)], ...
            [yl(1) yl(1) yl(2) yl(2)], ...
            [0 1 0], 'FaceAlpha', 0.08, 'EdgeColor', 'none');
    end

    if f_slow_clip(2) > f_slow_clip(1)
        patch(ax2, ...
            [f_slow_clip(1) f_slow_clip(2) f_slow_clip(2) f_slow_clip(1)], ...
            [yl(1) yl(1) yl(2) yl(2)], ...
            [1 1 0], 'FaceAlpha', 0.08, 'EdgeColor', 'none');
    end

    if isfinite(f_geo_mean) && f_geo_mean > 0
        plot(ax2, [f_geo_mean f_geo_mean], yl, 'k:', 'LineWidth', 1.3);
    end
    if isfinite(f_fast_geo_mean) && f_fast_geo_mean > 0
        h = plot(ax2, [f_fast_geo_mean f_fast_geo_mean], yl, ':', 'LineWidth', 1.3);
        h.Color = [0 0.4470 0.7410];
    end
    if isfinite(f_slow_geo_mean) && f_slow_geo_mean > 0
        h = plot(ax2, [f_slow_geo_mean f_slow_geo_mean], yl, ':', 'LineWidth', 1.3);
        h.Color = [0.8500 0.3250 0.0980];
    end

    ylim(ax2, yl);

    text(ax2, 0.03, 0.95, ...
        sprintf('ratio=%.3g', ratio_slow_fast), ...
        'Units', 'normalized', ...
        'VerticalAlignment', 'top', ...
        'FontSize', 8, ...
        'BackgroundColor', 'w', ...
        'Margin', 1);
end

%% 5) 전체 제목 -----------------------------------------------------------
title(tl_cur, 'Selected Current Profiles (full-length)', 'FontWeight', 'bold');
title(tl_psd, 'Selected PSD (Welch, Hann, full-length input)', 'FontWeight', 'bold');

%% 6) 결과 테이블 저장 ----------------------------------------------------
if ~isempty(psd_rows)
    psd_stat_tbl = struct2table(psd_rows);

    writetable(psd_stat_tbl, fullfile(save_dir, "selected_psd_stat_tbl_full.csv"));
    save(fullfile(save_dir, "selected_psd_stat_tbl_full.mat"), ...
        "psd_stat_tbl", ...
        "tau_fast_range", "tau_slow_range", "f_fast", "f_slow", ...
        "welch_win_sec", "welch_ovlp", "force_nfft", "nfft_fixed", ...
        "exclude_dc_for_stats", "t_max", "use_dB", "xlog_psd", "ylog_psd");

    exportgraphics(fig_cur, fullfile(save_dir, "selected_current_profiles_full.png"), 'Resolution', 200);
    exportgraphics(fig_psd, fullfile(save_dir, "selected_psd_full.png"), 'Resolution', 200);

    fprintf("[done] saved to: %s\n", save_dir);
else
    warning("No valid PSD rows were generated.");
end