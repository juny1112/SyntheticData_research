clear; clc; close all;

% ---------- 예시 파라미터 ----------
n       = 50;
tau_min = 0.1;         % s
tau_max = 400;          % s

% θ = log10(τ) 기준 (데케이드)
% tau_mode = [0.4; 3; 88];        % s
% sigma10  = [0.210; 0.200; 0.190];  % decades
tau_mode = [3; 88];        % s
sigma10  = [0.200; 0.190];  % decades


[mu10, tau_mean, tau_std, tau_median, mu_n, sigma_n] = ...
    DRT_mu_sigma(tau_mode, sigma10);

T = table(tau_mode, sigma10, mu10, tau_median, tau_mean, tau_std, ...
          mu_n, sigma_n, ...
          'VariableNames', ...
          {'tau_mode','sigma10','mu10','tau_median','tau_mean','tau_std','mu_n','sigma_n'});
disp(T)

A_tot = 2.0e-3;         % 총 저항 [Ω]
% w     = [1e-3 5e-2 1];% 모드 면적 비 (합이 1 아니어도 OK)
% % w     = [1e-4 2.5e-3 1] % 참고문헌
w     = [10e-2 1]; 


% ---------- 함수 호출 ----------
[theta, r_mode, r_theta, dth, R, tau, g_tau, A_modes, w_used] = ...
    DRT_Rtau(n, tau_min, tau_max, mu10, sigma10, A_tot, w);

% ---------- 검증(선택) ----------
sumR = sum(R)                          % ≈ A_tot 여야 함
rel_err = abs(sumR - A_tot)/A_tot      % 상대 오차
[mu10 tau_mean tau_std]        % 변환된 파라미터들

% ==== 한 번만 설정 (세션 전체 기본값을 LaTeX로) ====
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

% ---------- 1) r(θ) vs θ ----------
figure;
plot(theta, r_theta, 'k-', 'LineWidth', 1.6); grid on
xlabel('$\theta = \log_{10}(\tau)$');
ylabel('$r(\theta)\;[\Omega/\mathrm{decade}]$');
title('$r(\theta)\ \mathrm{vs}\ \theta$');


% ---------- 2) g(τ) vs τ ----------
figure;
semilogx(tau, g_tau, 'LineWidth', 1.6); grid on
xlabel('$\tau\ (\mathrm{s})$');
ylabel('$g(\tau)\;[\Omega/\mathrm{s}]$');
title('$g(\tau)\ \mathrm{vs}\ \tau$');


% ---------- 3) R vs τ ----------
% R은 bin별 가지 저항이므로, 이산성 강조를 원하면 stairs를 사용:
figure;
semilogx(tau, R, 'LineWidth', 1.6); grid on
xlabel('$\tau\ (\mathrm{s})$');
ylabel('$R_i\ [\Omega]$');
title('$R\ \mathrm{vs}\ \tau$');

