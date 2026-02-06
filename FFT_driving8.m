%% ======================================================================
%  드라이빙 부하(Current) 파일  →  FFT(Hann 창)  →  주요 주파수 & τ 요약
%  Var1 = 시간[s], Var2 = 전류[A]  (현재는 1열=시간, 2열=전류로 읽음)
%  τ  = 1 / (2πf)  [s] 도 함께 출력
%
%  (NEW)
%   1) FFT 플랏 xlim 조절 가능 (예: [0,0.2])
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

%% (옵션) Pulse -----------------------------------------------------------
t_end = 300;    % [sec]
dt    = 0.1;    % [sec]
t_p0  = 10;     % 펄스 시작 [sec]
t_p1  = 20;     % 펄스 종료 [sec]
pulse.t = (0:dt:t_end)';            % 시간 벡터 (세로 벡터)
pulse.I = zeros(size(pulse.t));     % 전류 벡터
pulse.I(pulse.t>=t_p0 & pulse.t<=t_p1) = 1;  % 10~20초 구간에 I=1[A] 펄스
% driving_files = {pulse};
% driving_files = [ driving_files; {pulse} ];

%% 사용자 설정 ------------------------------------------------------------
t_max    = 600;          % [s] 0~600초 구간만 분석
peak_num = 10;           % 피크 10개

xlim_fft = [0 0.2];      % FFT 플랏 x축 범위. 자동이면 [] 로.
% xlim_fft = [];

%% 결과 저장용 ------------------------------------------------------------
all_peak_tbl = struct;   % 파일별 피크·τ 저장(기존 그대로)

summary_tbl = table();   % 콘솔 요약을 테이블로도 쌓기 (long 형태)

nLoad = numel(driving_files);
Tau_mat = nan(peak_num, nLoad);     % Tau_s만 wide 형태로 모으기 (10 x 8)
load_labels = strings(nLoad,1);     % 열 이름용

%% 메인 루프 --------------------------------------------------------------
for fileIdx = 1:numel(driving_files)

    %% (A) 데이터 읽기 ---------------------------------------------------
    item = driving_files{fileIdx};
    if isstruct(item)
        t_vec = item.t(:);
        I_vec = item.I(:);
        file_label = sprintf('Pulse');
        load_label = "Pulse";
    else
        filename = item;
        data = readtable(filename, 'VariableNamingRule','preserve');
        t_vec = data{:,1};   % 1열 = 시간
        I_vec = data{:,2};   % 2열 = 전류
        file_label = filename;

        % 열 이름용 라벨: 파일명만(경로 제거)
        [~, nm, ~] = fileparts(char(filename));
        load_label = string(nm);
    end
    load_labels(fileIdx) = load_label;

    % NaN 제거
    mask  = ~(isnan(t_vec) | isnan(I_vec));
    t_vec = t_vec(mask);
    I_vec = I_vec(mask);

    %% (B) 0~600초 구간만 사용 ------------------------------------------
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

    %% (D) FFT -----------------------------------------------------------
    N       = numel(I_vec);
    win     = hann(N,'periodic');
    I_fft   = fft((I_vec - mean(I_vec)) .* win);

    halfIdx = 1:floor(N/2);
    f       = (halfIdx-1) * fs / N;                 % [Hz]
    mag     = abs(I_fft(halfIdx)) / (sum(win)/2);   % 진폭 보정

    %% (E) 피크 탐색 -----------------------------------------------------
    [pks, locs] = findpeaks(mag, f, ...
        'MinPeakProminence', 0.05*max(mag), ...
        'SortStr','descend');

    nShow = min(peak_num, numel(pks));

    if nShow == 0
        peak_tbl = table();
        warning('피크를 찾지 못했습니다: %s', file_label);
        locs_col = []; pks_col = []; tau_col = []; magdb_col = [];
    else
        idx       = 1:nShow;
        locs_col  = locs(idx);   locs_col  = locs_col(:);   % Hz
        pks_col   = pks(idx);    pks_col   = pks_col(:);
        tau_col   = 1./(2*pi*locs_col);                     % s
        magdb_col = 20*log10(pks_col);

        peak_tbl  = table(locs_col, tau_col, pks_col, magdb_col, ...
            'VariableNames', {'Freq_Hz','Tau_s','Amplitude','Mag_dB'});
    end

    all_peak_tbl.(sprintf('file%d',fileIdx)) = peak_tbl;

    % (NEW-1) 콘솔 요약용 테이블(long) 누적
    if ~isempty(peak_tbl)
        tmp = peak_tbl;
        tmp.Load = repmat(load_label, height(tmp), 1);
        tmp.PeakRank = (1:height(tmp))';
        tmp = movevars(tmp, {'Load','PeakRank'}, 'Before', 1);
        summary_tbl = [summary_tbl; tmp]; %#ok<AGROW>
    end

    % (NEW-2) Tau_s wide 매트릭스 채우기 (10 x nLoad)
    if nShow > 0
        Tau_mat(1:nShow, fileIdx) = tau_col;
    end

    %% (F) 시각화 --------------------------------------------------------
    fig_title = sprintf('[%d] %s', fileIdx, file_label);
    figure('Name', fig_title, 'Position', [100 100 850 500]);

    % 시간 영역
    subplot(2,1,1);
    plot(t_vec, I_vec,'LineWidth',1.1); grid on;
    xlabel('Time (s)'); ylabel('Current (A)');
    title(sprintf('%s Current', load_label), 'Interpreter','none');  % <-- 변경

    % 주파수 영역
    subplot(2,1,2);
    plot(f, mag,'LineWidth',1.2); hold on;
    if nShow > 0
        stem(locs_col, pks_col,'r','filled','LineWidth',1.3);
        legend('Spectrum','Detected Peaks');
    else
        legend('Spectrum');
    end
    grid on; xlabel('Frequency (Hz)'); ylabel('|I(f)|');
    title('FFT Magnitude Spectrum (Hann window)');

    % FFT 플랏 xlim 조절
    if ~isempty(xlim_fft)
        xlim(xlim_fft);
    end
end

%% (G) 콘솔 요약 ---------------------------------------------------------
fprintf('\n=============== 고유 주파수 & τ 요약 ===============\n');
for fileIdx = 1:numel(driving_files)
    item = driving_files{fileIdx};
    if isstruct(item)
        fprintf('\n--- Pulse ---\n');
    else
        fprintf('\n--- %s ---\n', item);
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

%% 확인용 출력(원하면 주석) ----------------------------------------------
disp(" ");
disp("==== summary_tbl (long) 미리보기 ====");
disp(head(summary_tbl, 20));

disp(" ");
disp("==== Tau_wide_tbl (10 x Loads) ====");
disp(Tau_wide_tbl);
