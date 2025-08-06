clc; clear; close all;

% ======================================================================
% Example: 40-point τ grid, τ ∈ [0.1, 200], two modes each 1 mΩ at τ≈6 s & τ≈60 s
% ======================================================================
n       = 40;
tau_min = 0.1;   tau_max = 200;
mu1     = log(6);     sigma1 = 0.8;    % mode 1 centered at θ = ln(6)
mu2     = log(60);    sigma2 = 0.4;    % mode 2 centered at θ = ln(60)
mode1   = 0.001;      mode2  = 0.001;  % each mode’s total area = 1 mΩ

% ----------------------------------------------------------------------
% 1) Call Bimodal_R to get θ, τ, density vectors r1, r2, r_tot and lumped R
% ----------------------------------------------------------------------
[theta, tau, r1, r2, r_tot, R] = Bimodal_R( ...
    n, tau_min, tau_max, mu1, sigma1, mu2, sigma2, mode1, mode2);

% 2) 첫 번째 모드(r1)와 두 번째 모드(r2)의 최고점 인덱스 구하기
[~, idx1] = max(r1);   % idx1 == 21 (≈ τ ≈ 6 s)
[~, idx2] = max(r2);   % idx2 == 32 (≈ τ ≈ 60 s)

tau1_true = tau(idx1);
R1_true   = R(idx1);
tau2_true = tau(idx2);
R2_true   = R(idx2);

% 3) 이 값들을 2RC 참값으로 설정
R0_true = 1e-3;  % (예: 직렬 이온 저항 등)
X_true_2RC = [ R0_true;
               R1_true;  % 약 1e-3 Ω
               R2_true;  % 약 1e-3 Ω
               tau1_true; % ≈ 6 s
               tau2_true; % ≈ 60 s
             ];

% ----------------------------------------------------------------------
% 2) Plot the density functions r1(θ), r2(θ), r_tot(θ)
% ----------------------------------------------------------------------
figure('Name','Bimodal');
plot(theta, r1,    'r-o', 'LineWidth',1.5, 'MarkerSize',5, 'DisplayName','Mode 1 (1 mΩ)');
hold on;
plot(theta, r2,    'g-s', 'LineWidth',1.5, 'MarkerSize',5, 'DisplayName','Mode 2 (1 mΩ)');
plot(theta, r_tot, 'k-^','LineWidth',2, 'MarkerSize',6, 'DisplayName','Total Density (2 mΩ)');
xlabel('\theta = ln(\tau)', 'FontSize',12);
ylabel('\gamma', 'FontSize',12);
title('Bimodal', 'FontSize',14);
grid on;
legend('Location','northwest','FontSize',11);
xlim([log(tau_min), log(tau_max)]);

% ----------------------------------------------------------------------
% 3) Show τ grid and corresponding R_i values
% ----------------------------------------------------------------------
figure('Name','\tau vs R');
stem(tau, R, 'filled','MarkerSize',4);
xlabel('\tau[s]', 'FontSize',12);
ylabel('R[Ω]',  'FontSize',12);
title('\tau vs R', 'FontSize',14);
grid on;
set(gca, 'XScale', 'log');   % log-scale x-axis to reflect τ spacing
xlim([tau_min, tau_max]);
