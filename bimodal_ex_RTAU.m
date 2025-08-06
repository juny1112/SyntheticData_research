clc;clear;close all
% ── 파라미터 생성 ─────────────────────────────────────────────────────
n     = 40;                     % RC 쌍 수
n_mode1  = 10;                     % 모드1에 10개 할당
tau_peak = [6 60];                 % 두 피크 중심 [s]
sigma    = [10 20];                 % 표준편차 [s]
tau_rng  = [0.1 150];                 % τ 범위 [s]
R0       = 0.001;                      % 직렬저항 (필요하면 값 지정)

[X, R_i, tau_i] = bimodal_Rtau(n, n_mode1, tau_peak, sigma, tau_rng, R0);

% ── 확인: 모드별 합계 ---------------------------------------------------
R_i = X(2:n+1);
fprintf('모드1 합 = %.6f Ω\n', sum(R_i(1:n_mode1)));
fprintf('모드2 합 = %.6f Ω\n', sum(R_i(n_mode1+1:end)));

figure
stem(tau_i, R_i*1e3, 'filled', 'LineWidth',1.4, 'MarkerSize',4); % mΩ로 표시
grid on; xlabel('\tau  (s)'); ylabel('R  (m\Omega)');
title('n-RC tau vs R (1 mΩ per mode)');