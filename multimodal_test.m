clc; clear; close all

% ─ 입력 데이터(50 % SOC 기준) ────────────────────────
n        = 50;                  % 총 RC 쌍
n_mode12 = [15 15];              % mode 할당 n 개수
tau_pk   = [0.4 3 88];           % τ 중심 [s]
sigma = [0.07 0.18 0.2];         % 표준편차
areaRel  = [1e-3 5e-2 1];      % 면적 비
R_total  = 0.002;                % 총 저항
R_mode_sum   = R_total * areaRel / sum(areaRel);  % 각 mode 별 저항 [Ω]
tau_rng  = [0.1 400];           % 배치 범위
R0       = 1e-3;                 % R0


% ─ 분포 생성 ────────────────────────────────────────
[X,R_i,tau_i] = multimodal_Rtau(n, n_mode12, ...
                              tau_pk, sigma, R_mode_sum, ...
                              tau_rng, R0);

% ─ 확인용 플롯 ──────────────────────────────────────
figure;
semilogx(tau_i, R_i*1e3, 'o-')
xlabel('\tau [s]'); ylabel('R_i [m\Omega]');
title('n-RC Resistance Distribution (Trimodal, 50 % SOC)');
grid on

