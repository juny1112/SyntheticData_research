clc; clear; close all;
% bimodal 예시: 40-점 τ 그리드, τ∈[0.1, 130], 모드1=1mΩ at τ≈6s, 모드2=1mΩ at τ≈60s
n       = 40;
tau_min = 0.1;  tau_max = 200;
mu1     = log(6);    sigma1 = 0.8;
mu2     = log(60);   sigma2 = 0.4;
mode1   = 0.001;     mode2  = 0.001;

[theta, tau, R1, R2, R_tot] = Bimodal_R(n, tau_min, tau_max, mu1, sigma1, mu2, sigma2, mode1, mode2);

% 결과를 그려보기
figure('Name','Bimodal R Distribution');
plot(theta, R1, 'r-o', 'LineWidth',1.5, 'MarkerSize',5, 'DisplayName','Mode 1 (1 mΩ)'); hold on;
plot(theta, R2, 'g-s', 'LineWidth',1.5, 'MarkerSize',5, 'DisplayName','Mode 2 (1 mΩ)');
plot(theta, R_tot, 'k-^','LineWidth',2, 'MarkerSize',6, 'DisplayName','Total (2 mΩ)');
xlabel('\theta = ln(\tau)', 'FontSize',12);
ylabel('\gamma', 'FontSize',12);
title('Bimodal Resistance Distribution', 'FontSize',14);
grid on; legend('Location','northwest','FontSize',11);
xlim([log(tau_min) log(tau_max)]);