%% ======================================================================
%  드라이빙 부하(Current) 파일  →  (주행 부하 + 인공 sin)  →  FFT & τ 요약
%  ● 인공 사인파: τ_add = 60 s (f_add = 1/60 Hz) 를 전류에 합산
%  ● 엑셀 헤더가 없고 첫·둘째 열이 시간[s], 전류[A] 라고 가정
% ======================================================================
clear; clc; close all;

% 0) 분석할 파일 목록 -----------------------------------------------------
driving_files = {
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\udds_unit_time_scaled_current.xlsx'
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\us06_unit_time_scaled_current.xlsx'
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_CITY1_time_scaled_current.xlsx'
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_HW1_time_scaled_current.xlsx'
};

% ── 인공 사인 성분 설정 ───────────────────────────────────────────────
tau_add = 377;               % [s]  ← τ₂ 와 동일
f_add   = 1 / tau_add;      % [Hz]
A_add   = 30;                % [A]  ← 원하는 진폭(조절 가능)

all_peak_tbl = struct;      % 결과 저장

for fileIdx = 1:numel(driving_files)
    %% (A) 데이터 읽기 ---------------------------------------------------
    filename = driving_files{fileIdx};
    data     = readtable(filename);          

    t_vec = data.Var1;                       % 시간 [s]
    I_vec = data.Var2;                       % 전류 [A]

    % NaN 제거
    mask  = ~(isnan(t_vec) | isnan(I_vec));
    t_vec = t_vec(mask);
    I_vec = I_vec(mask);

    %% (B) 인공 사인파 합산 --------------------------------------------
    I_add = A_add * sin(2*pi*f_add * t_vec); % 사인파
    I_vec = I_vec + I_add;                   % 기존 부하 + τ₍add₎ 사인

    %% (C) 샘플링 정보 ---------------------------------------------------
    dt = median(diff(t_vec));
    fs = 1/dt;

    %% (D) FFT -----------------------------------------------------------
    N       = numel(I_vec);
    win     = hann(N,'periodic');
    I_fft   = fft((I_vec - mean(I_vec)) .* win);

    halfIdx = 1:floor(N/2);
    f       = (halfIdx-1) * fs / N;
    mag     = abs(I_fft(halfIdx)) / (sum(win)/2);

    %% (E) 피크 탐색 -----------------------------------------------------
    [pks, locs] = findpeaks(mag, f, ...
                            'MinPeakProminence', 0.05*max(mag), ...
                            'SortStr','descend');
    peak_num = 10;
    nShow  = min(peak_num, numel(pks));

    if nShow == 0
        peak_tbl   = table();
        locs_col   = []; pks_col = []; tau_col = [];
    else
        idx       = 1:nShow;
        locs_col  = locs(idx);  locs_col  = locs_col(:);
        pks_col   = pks(idx);   pks_col   = pks_col(:);
        tau_col   = 1./(2*pi*locs_col);
        %tau_col = 1./locs_col;
        magdb_col = 20*log10(pks_col);

        peak_tbl  = table(locs_col, tau_col, pks_col, magdb_col, ...
            'VariableNames', {'Freq_Hz','Tau_s','Amplitude','Mag_dB'});
    end
    all_peak_tbl.(sprintf('file%d',fileIdx)) = peak_tbl;

    %% (F) 시각화 --------------------------------------------------------
    figure('Name', sprintf('[%d] %s',fileIdx,filename), ...
           'Position',[100 100 850 500]);

    % (F-1) 시간 영역 파형
    subplot(2,1,1);
    plot(t_vec, I_vec,'LineWidth',1.1); grid on;
    xlabel('Time (s)'); ylabel('Current (A)');
    title(sprintf('Time-domain Current  (+ %.1f A · sin(2π·t/tau s))', A_add));

    % (F-2) 주파수 영역
    subplot(2,1,2);
    plot(f, mag,'LineWidth',1.2); hold on;
    if nShow > 0
        stem(locs_col, pks_col,'r','filled','LineWidth',1.3);
        legend('Spectrum','Detected Peaks');
    else
        legend('Spectrum');
    end
    % 인공 성분 표시용 선 (f_add 위치)
    xline(f_add,'--k','τ_{add}','LabelOrientation','horizontal',...
          'LabelVerticalAlignment','bottom');
    grid on; xlabel('Frequency (Hz)'); ylabel('|I(f)|');
    title('FFT Magnitude Spectrum (Hann window)');
end

%% (G) 콘솔 요약 ---------------------------------------------------------
fprintf('\n======= 고유 주파수 & τ 요약 (sin τ_{add}=60 s 포함) =======\n');
for fileIdx = 1:numel(driving_files)
    fprintf('\n--- %s ---\n', driving_files{fileIdx});
    disp(all_peak_tbl.(sprintf('file%d',fileIdx)));
end
