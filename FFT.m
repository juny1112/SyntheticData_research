%% ======================================================================
%  드라이빙 부하(Current) 파일  →  FFT(Hann 창)  →  주요 주파수 & τ 요약
%  Var1 = 시간[s], Var2 = 전류[A]
%  τ  = 1 / (2πf)  [s] 도 함께 출력
% ======================================================================
clear; clc; close all;

% 0) 분석할 파일 목록 -----------------------------------------------------
% Driving
 driving_files = {
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\udds_unit_time_scaled_current.xlsx'
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\us06_unit_time_scaled_current.xlsx'
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_CITY1_time_scaled_current.xlsx'
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_HW1_time_scaled_current.xlsx'
};

% Pulse
t_end = 300;    % [sec]
dt    = 0.1;    % [sec]
t_p0  = 10;     % 펄스 시작 [sec]
t_p1  = 20;     % 펄스 종료 [sec]
pulse.t = (0:dt:t_end)';            % 시간 벡터 (세로 벡터)
pulse.I = zeros(size(pulse.t));     % 전류 벡터
pulse.I(pulse.t>=t_p0 & pulse.t<=t_p1) = 1;  % 10~20초 구간에 I=1[A] 펄스
% driving_files = {pulse};

% 같이 처리시 
driving_files = [ driving_files; {pulse} ];

all_peak_tbl = struct;            % 파일별 피크·τ 저장

for fileIdx = 1:numel(driving_files)
    %% (A) 데이터 읽기 ---------------------------------------------------
    item = driving_files{fileIdx};
    if isstruct(item)
        % (B) struct 형태로 들어온 경우: Pulse 데이터
        t_vec = item.t(:);   % 세로 벡터로 보장
        I_vec = item.I(:);
        file_label = sprintf('Pulse'); 
    else
        % (A) 문자열(파일 경로)로 들어온 경우: 엑셀 파일 읽기
        filename = item;
        data     = readtable(filename);
        t_vec = data.Var1;
        I_vec = data.Var2;
        file_label = filename;  
    end

    % NaN 제거
    mask  = ~(isnan(t_vec) | isnan(I_vec));
    t_vec = t_vec(mask);
    I_vec = I_vec(mask);

    %% (B) 샘플링 정보 ---------------------------------------------------
    dt = median(diff(t_vec));                  % 평균 샘플링 간격 [s]
    fs = 1/dt;                                 % 샘플링 주파수 [Hz]

    %% (C) FFT -----------------------------------------------------------
    N       = numel(I_vec);
    win     = hann(N,'periodic');                   % 창 곱해서 leakage 제거(Hann창: 양 끝단 0)
    I_fft   = fft((I_vec - mean(I_vec)) .* win);    % DC 성분 제거
    %I_fft   = fft(I_vec); 이렇게만 하면 0Hz 성분 커짐

    halfIdx = 1:floor(N/2);                         % 양의 주파수(+fs/2)만 취하기 위해 인덱스 절반만 가져감(floor: 가까운 정수로 내림)
    f       = (halfIdx-1) * fs / N; % [Hz]          % fs/N = FFT 분해능
    mag     = abs(I_fft(halfIdx)) / (sum(win)/2);   % 창에 의해 줄어든 진폭 보정

    %% (D) 피크 탐색(최대 5개) -----------------------------------------
    [pks, locs] = findpeaks(mag, f, ...
                            'MinPeakProminence', 0.05*max(mag), ... % 피크의 두드러짐을 최대값의 0.05이상으로 제한
                            'SortStr','descend'); % pks, locs 내림차순 정렬(진폭 순)
    peak_num = 10; % 반환하고싶은 피크 수
    nShow = min(peak_num, numel(pks)); % 찾은 피크 수와 5 중 더 작은 값 반환

    if nShow == 0
        peak_tbl = table();
        warning('피크를 찾지 못했습니다: %s', file_label);
        locs_col = []; pks_col = []; tau_col = [];
    else
        idx       = 1:nShow;
        % 열벡터로 변환
        locs_col  = locs(idx);      locs_col  = locs_col(:);      % f(주파수) [Hz]
        pks_col   = pks(idx);       pks_col   = pks_col(:);       % 진폭
        % 주파수 -> 시상수
        tau_col   = 1./(2*pi*locs_col);                          % τ [s]
        % 진폭(A) -> 데시벨 단위 - 굳이 필요는 없음
        magdb_col = 20*log10(pks_col);                           % dB

        peak_tbl  = table(locs_col, tau_col, pks_col, magdb_col, ...
            'VariableNames', {'Freq_Hz','Tau_s','Amplitude','Mag_dB'});
    end
    all_peak_tbl.(sprintf('file%d',fileIdx)) = peak_tbl;

    %% (E) 시각화 --------------------------------------------------------
    fig_title = sprintf('[%d] %s', fileIdx, file_label);
    figure('Name', fig_title, 'Position', [100 100 850 500]);

    % 시간 영역
    subplot(2,1,1);
    plot(t_vec, I_vec,'LineWidth',1.1); grid on;
    xlabel('Time (s)'); ylabel('Current (A)');
    title('Time-domain Current');

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
end

%% (F) 콘솔 요약 ---------------------------------------------------------
fprintf('\n=============== 고유 주파수 & τ 요약 ===============\n');
for fileIdx = 1:numel(driving_files)
    % 파일 라벨(Pulse 혹은 파일 경로) 표시
    item = driving_files{fileIdx};
    if isstruct(item)
        fprintf('\n--- Pulse ---\n');
    else
        fprintf('\n--- %s ---\n', item);
    end
    disp(all_peak_tbl.(sprintf('file%d', fileIdx)));
end
