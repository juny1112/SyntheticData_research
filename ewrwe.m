clear; clc; close all

%% -----------------------------------------------------------
% 0) 설정
% -----------------------------------------------------------
n        = 40;
tau_min  = 0.1;     tau_max = 200;
mu1      = log(6);  sigma1  = 0.8;      % mode-1 중심·폭
mu2      = log(60); sigma2  = 0.4;      % mode-2 중심·폭
mode1    = 0.001;   mode2   = 0.001;    % 면적 = 1 mΩ씩
R0       = 2e-3;                       % 직렬 저항(예) 2 mΩ

% -----------------------------------------------------------
% 1) bimodal 저항 분포 생성
% -----------------------------------------------------------
[theta, tau, R1, R2, R_tot, R] = ...
    Bimodal_R(n, tau_min, tau_max, ...
              mu1, sigma1, mu2, sigma2, mode1, mode2);
%   → tau (n×1), R_tot (n×1) 획득

% -----------------------------------------------------------
% 2) n-RC 파라미터 벡터 X 구성
%    X = [R0 ; R1 … Rn ; tau1 … taun]
% -----------------------------------------------------------
X = [ R0 ; R ; tau ];   % 길이 = 1 + 2n

% -----------------------------------------------------------
% 3) 전류·시간 벡터 예시 & n-RC 시뮬
% -----------------------------------------------------------
% Pulse data
t_end = 50; %[sec]
dt = 0.1; %[sec]
t_p0 = 10;
t_p1 = 20;
t_vec = (0:dt:t_end)'; % 0초 부터 시작
I_vec = zeros(size(t_vec)); 
idx_pulse = (t_vec >= t_p0) & (t_vec <= t_p1); % 펄스 구간(10초~20초) 인덱싱
I_vec(idx_pulse) = 1;

% Plot current (확인용)
figure;
plot(t_vec, I_vec, 'r-');
xlabel('Time(sec)');
ylabel('Current(A)');
title('pulse data');
grid on;

% 1RC model -> 전압 데이터 생성
V_est = RC_model_n(X, t_vec, I_vec, n); % <-- 바로 사용

% Plot voltage (확인용)
figure;
plot(t_vec, V_est, 'b-');
xlabel('Time (sec)');
ylabel('Voltage (V)');
title('1RC Model Voltage');
grid on;

%% Plot I,V
figure;
yyaxis left;
plot(t_vec, I_vec, 'r-');
ylabel('Current (C)');
ax = gca; 
ax.YColor = 'r'; 
%ylim([-0.1, 1.1]);

yyaxis right;
plot(t_vec, V_est, 'b-'); 
ylabel('Voltage (V)');
ax.YColor = 'b';
%ylim([-0.2, 2.2]);

xlabel('Time (sec)');
title('pulse data\_1RC model');
grid on;
legend('Current (I)', 'Voltage (V)');

