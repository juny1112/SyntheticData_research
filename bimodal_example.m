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

% ----------------------------------------------------------------------
% 2) Plot the density functions r1(θ), r2(θ), r_tot(θ)
% ----------------------------------------------------------------------
figure('Name','Bimodal','Color','w');
plot(theta, r1,    'r-o', 'LineWidth',1.5, 'MarkerSize',5, 'DisplayName','Mode 1 (1 mΩ)');
hold on;
plot(theta, r2,    'g-s', 'LineWidth',1.5, 'MarkerSize',5, 'DisplayName','Mode 2 (1 mΩ)');
plot(theta, r_tot, 'k-^','LineWidth',2, 'MarkerSize',6, 'DisplayName','Total Density (2 mΩ)');
xlabel('\theta = ln(\tau)', 'FontSize',12);
ylabel('r(\theta)\;[Ω\!/\,\text{rad}]', 'FontSize',12);
title('Bimodal Resistance Density vs \theta', 'FontSize',14);
grid on;
legend('Location','northwest','FontSize',11);
xlim([log(tau_min), log(tau_max)]);

% ----------------------------------------------------------------------
% 3) Plot the lumped branch resistances R_i = r_tot_i · Δθ_i
% ----------------------------------------------------------------------
figure('Name','Bimodal Lumped R Distribution','Color','w');
plot(theta, R, 'b-o', 'LineWidth',1.5, 'MarkerSize',5, 'DisplayName','R_i = r_{tot,i}·Δθ_i');
xlabel('\theta = ln(\tau)', 'FontSize',12);
ylabel('R_i[Ω]', 'FontSize',12);
title('Lumped Branch Resistances vs \theta', 'FontSize',14);
grid on;
xlim([log(tau_min), log(tau_max)]);

% ----------------------------------------------------------------------
% 4) (Optional) Show τ grid and corresponding R_i values
% ----------------------------------------------------------------------
figure('Name','\tau vs Lumped R','Color','w');
stem(tau, R, 'filled','MarkerSize',4);
xlabel('\tau\;[s]', 'FontSize',12);
ylabel('R_i[Ω]',  'FontSize',12);
title('Lumped Branch Resistances vs \tau', 'FontSize',14);
grid on;
set(gca, 'XScale', 'log');   % log-scale x-axis to reflect τ spacing
xlim([tau_min, tau_max]);
