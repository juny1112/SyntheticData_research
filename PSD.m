%% ======================================================================
%  드라이빙 부하(Current) 파일  →  PSD(Welch, Hann)  →  주요 주파수 & τ 요약
%  - 입력(엑셀): 1열=시간[s], 2열=전류[A]
%  - τ = 1/(2πf) [s]
%
%  (NEW)
%   1) PSD 플랏 xlim 조절 가능 (예: [0,0.2])
%   2) 콘솔 요약을 "테이블"로도 생성 (summary_tbl)
%   3) Tau_s만: (행=피크 10개) x (열=주행부하 8종) 테이블 생성 (Tau_wide_tbl)
%   4) Figure(시간영역) title을 "<주행부하이름> Current" 로 표시
% ======================================================================
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

%% 사용자 설정 ------------------------------------------------------------
t_max    = 600;          % [s] 0~600초 구간만 분석
peak_num = 10;           % 피크 10개
xlim_psd = [0 0.2];      % PSD 플랏 x축 범위. 자동이면 [] 로.
% xlim_psd = [];

% Welch PSD 설정
welch_win_sec = 20;      % [s] 윈도 길이 (추천: 10~50s 정도)
welch_ovlp    = 0.5;     % overlap 비율 (0~1)

% 표시 옵션
use_dB = false;          % true면 10*log10(PSD)로 표시 (dB/Hz)

%% 결과 저장용 ------------------------------------------------------------
all_peak_tbl = struct;   % 파일별 피크 테이블 저장(PSD 기반)

summary_tbl = table();   % long 형태: Load, PeakRank, Freq_Hz, Tau_s, PSD_A2Hz, PSD_dBHz(optional)

nLoad = numel(driving_files);
Tau_mat = nan(peak_num, nLoad);     % Tau_s wide (10 x 8)
load_labels = strings(nLoad,1);

%% 메인 루프 --------------------------------------------------------------
for fileIdx = 1:nLoad

    %% (A) 데이터 읽기 ---------------------------------------------------
    filename = driving_files{fileIdx};
    data = readtable(filename, 'VariableNamingRule','preserve');
    t_vec = data{:,1};   % 시간
    I_vec = data{:,2};   % 전류
    file_label = filename;

    [~, nm, ~] = fileparts(char(filename));
    load_label = string(nm);
    load_labels(fileIdx) = load_label;

    % NaN 제거
    mask  = ~(isnan(t_vec) | isnan(I_vec));
    t_vec = t_vec(mask);
    I_vec = I_vec(mask);

    %% (B) 0~t_max 구간만 사용 ------------------------------------------
    t0 = t_vec(1);
    mask600 = (t_vec >= t0) & (t_vec <= t0 + t_max);
    t_vec = t_vec(mask600);
    I_vec = I_vec(mask600);

    if numel(t_vec) < 10
        warning('600초 구간 데이터가 너무 짧습니다: %s', file_label);
        all_peak_tbl.(sprintf('file%d',fileIdx)) = table();
        continue;
    end

    %% (C) 샘플링 정보 ---------------------------------------------------
    dt = median(diff(t_vec));
    fs = 1/dt;

    % (권장) 평균 제거(DC 제거)
    I0 = I_vec - mean(I_vec);

    %% (D) Welch PSD -----------------------------------------------------
    % 윈도 길이(샘플) 지정 (너무 길거나 짧으면 자동 보정)
    nWin = max(16, round(welch_win_sec * fs));
    nWin = min(nWin, numel(I0));  % 데이터 길이 초과 방지

    win = hann(nWin, 'periodic');
    noverlap = round(welch_ovlp * nWin);

    % nfft는 기본(자동) 써도 되지만, 분해능 조절하고 싶으면 2^nextpow2(nWin) 등 사용
    nfft = max(256, 2^nextpow2(nWin));

    % Pxx: A^2/Hz, f: Hz
    [Pxx, f] = pwelch(I0, win, noverlap, nfft, fs, 'onesided');

    % 표시용 y (선형 또는 dB)
    if use_dB
        yplot = 10*log10(Pxx + realmin);
        ylab  = 'PSD (dB/Hz)';
    else
        yplot = Pxx;
        ylab  = 'PSD (A^2/Hz)';
    end

    %% (E) 피크 탐색 (PSD 기준) ------------------------------------------
    % PSD는 dynamic range가 크니 prominence를 살짝 완만하게(예: max의 1~5%)
    if all(~isfinite(Pxx)) || max(Pxx)<=0
        pks = []; locs = [];
    else
        [pks, locs] = findpeaks(Pxx, f, ...
            'MinPeakProminence', 0.02*max(Pxx), ...  % 필요시 0.01~0.05 조절
            'SortStr','descend');
    end

    nShow = min(peak_num, numel(pks));

    if nShow == 0
        peak_tbl = table();
        warning('피크를 찾지 못했습니다: %s', file_label);
        locs_col = []; pks_col = []; tau_col = []; psd_db_col = [];
    else
        idx      = 1:nShow;
        locs_col = locs(idx); locs_col = locs_col(:);     % Hz
        pks_col  = pks(idx);  pks_col  = pks_col(:);      % A^2/Hz
        tau_col  = 1./(2*pi*locs_col);                    % s
        psd_db_col = 10*log10(pks_col + realmin);         % dB/Hz (참고용)

        peak_tbl = table(locs_col, tau_col, pks_col, psd_db_col, ...
            'VariableNames', {'Freq_Hz','Tau_s','PSD_A2Hz','PSD_dBHz'});
    end

    all_peak_tbl.(sprintf('file%d',fileIdx)) = peak_tbl;

    % (NEW-1) 요약 테이블(long) 누적
    if ~isempty(peak_tbl)
        tmp = peak_tbl;
        tmp.Load = repmat(load_label, height(tmp), 1);
        tmp.PeakRank = (1:height(tmp))';
        tmp = movevars(tmp, {'Load','PeakRank'}, 'Before', 1);
        summary_tbl = [summary_tbl; tmp]; %#ok<AGROW>
    end

    % (NEW-2) Tau wide 채우기
    if nShow > 0
        Tau_mat(1:nShow, fileIdx) = tau_col;
    end

    %% (F) 시각화 --------------------------------------------------------
    fig_title = sprintf('[%d] %s', fileIdx, file_label);
    figure('Name', fig_title, 'Position', [100 100 900 520]);

    % 시간 영역
    subplot(2,1,1);
    plot(t_vec, I_vec,'LineWidth',1.1); grid on;
    xlabel('Time (s)'); ylabel('Current (A)');
    title(sprintf('%s Current', load_label), 'Interpreter','none');

    % 주파수 영역(PSD)
    subplot(2,1,2);
    plot(f, yplot,'LineWidth',1.2); hold on;

    if nShow > 0
        if use_dB
            stem(locs_col, 10*log10(pks_col + realmin), 'r','filled','LineWidth',1.2);
        else
            stem(locs_col, pks_col, 'r','filled','LineWidth',1.2);
        end
        legend('PSD (Welch)','Detected Peaks','Location','best');
    else
        legend('PSD (Welch)','Location','best');
    end

    grid on; xlabel('Frequency (Hz)'); ylabel(ylab);
    title('PSD (Welch, Hann window)');

    if ~isempty(xlim_psd)
        xlim(xlim_psd);
    end
end

%% (G) 콘솔 요약 ---------------------------------------------------------
fprintf('\n=============== 고유 주파수 & τ 요약 (PSD 기반) ===============\n');
for fileIdx = 1:nLoad
    fprintf('\n--- %s ---\n', driving_files{fileIdx});
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
disp("==== summary_tbl (PSD long) 미리보기 ====");
disp(head(summary_tbl, 20));

disp(" ");
disp("==== Tau_wide_tbl (10 x Loads) ====");
disp(Tau_wide_tbl);
