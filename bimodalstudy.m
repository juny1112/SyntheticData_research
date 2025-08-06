%% ===================================================================
%  – bimodal R(τ) 분포를 출력합니다.
%  – 각 모드의 면적(∑ R·Δ(lnτ))을 0.001 Ω(=1 mΩ)으로 정규화
%% ===================================================================
clc; clear; close all;

%% 1) 그리드 및 저항분포 설정
n    = 40;                     % RC 개수
tau_min  = 0.1; tau_max = 200;      % τ 범위 0.1 ~ 130 s
tau      = logspace(log10(tau_min), log10(tau_max), n).';  % τ 그리드 설정

% µ= ln(τ_peak), σ는 폭 조절
mu1    = log(6);       sigma1 = 0.8;    % 모드1: τ≈6 s 주변
mu2    = log(60);      sigma2 = 0.4;    % 모드2: τ≈60 s 주변

% 각 mode 면적 설정
mode1 = 0.001; %[Ω]
mode2 = 0.001; %[Ω]

%% 2) γ 함수 계산 (연속 로그정규 PDF 식)
gamma1 = 1./(sigma1 * sqrt(2*pi)) .* exp( - (log(tau) - mu1).^2 ./ (2*sigma1^2) );
gamma2 = 1./(sigma2 * sqrt(2*pi)) .* exp( - (log(tau) - mu2).^2 ./ (2*sigma2^2) );

%% 3) Δ(ln τ) = Δθ 계산 (중앙차분 방식)
theta = log(tau);
dtheta   = zeros(n,1);
dtheta(1)    = theta(2)      - theta(1);
dtheta(end)  = theta(end)    - theta(end-1);
for i = 2:n-1
    dtheta(i) = (theta(i+1) - theta(i-1))/2;
end

%% 4) 각 모드 면적 정규화 (=1 mΩ)
%   Σ[γ(i) * Δθ(i)] = 1 이 되도록 먼저 나눈 뒤, 0.001 Ω 곱함
gamma1 = gamma1 ./ sum( gamma1 .* dtheta );   % Σ[γ₁·Δlnτ] = 1
gamma2 = gamma2 ./ sum( gamma2 .* dtheta );   % Σ[γ₂·Δlnτ] = 1

R1     = gamma1 * mode1;  % (n×1), Σ[R1·Δlnτ] = 0.001 Ω
R2     = gamma2 * mode2;  % (n×1), Σ[R2·Δlnτ] = 0.001 Ω
R_tot  = R1 + R2;         % (n×1), Σ[R_tot·Δlnτ] = 0.002 Ω

% % 면적 확인 출력 (옵션)
% S1 = sum( R1    .* dtheta );
% S2 = sum( R2    .* dtheta );
% St = sum( R_tot .* dtheta );
% fprintf('Mode1 합 = %.4g Ω, Mode2 합 = %.4g Ω, Total 합 = %.4g Ω\n', S1, S2, St);

%% 5) 그래프 그리기
figure('Name','Bimodal');
plot(theta, R1, 'r-o', 'LineWidth',1.5, 'MarkerSize',5, 'DisplayName','Mode 1 (1 mΩ)'); hold on;
plot(theta, R2, 'g-s', 'LineWidth',1.5, 'MarkerSize',5, 'DisplayName','Mode 2 (1 mΩ)');
plot(theta, R_tot, 'k-^','LineWidth',2, 'MarkerSize',6, 'DisplayName','Total (2 mΩ)');
xlabel('\theta = ln(\tau)', 'FontSize',12);
ylabel('\gamma', 'FontSize',12);
title('Bimodal Resistance Distribution', 'FontSize',14);
grid on; legend('Location','northwest','FontSize',11);
xlim([log(tau_min) log(tau_max)]);

figure('Name','\tau vs R','Color','w');
stem(tau, R, 'filled','MarkerSize',4);
xlabel('\tau\;[s]', 'FontSize',12);
ylabel('R_i[Ω]',  'FontSize',12);
title('Lumped Branch Resistances vs \tau', 'FontSize',14);
grid on;
set(gca, 'XScale', 'log');   % log-scale x-axis to reflect τ spacing
xlim([tau_min, tau_max]);

