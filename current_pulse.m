clear; clc; close all;

%% I(t)_pulse 생성
% parameters
t_end = 50; %[sec]
dt = 0.1; %[sec]
t_p0 = 10;
t_p1 = 20;
X = [1 1 1]; %임의로 [R0, R1, tau1] 설정

% t vector
t_vec = 0:dt:t_end; % 0초 부터 시작

% I vector 
I_vec = zeros(size(t_vec)); 
idx_pulse = (t_vec >= t_p0) & (t_vec <= t_p1); % 펄스 구간(10초~20초) 인덱싱
I_vec(idx_pulse) = 1;

% output [t_vec, I_vec]

% Plot current (확인용)
figure;
plot(t_vec, I_vec, 'r-');
xlabel('Time(sec)');
ylabel('Current(A)');
title('pulse data');
grid on;


%% 1RC model -> 전압 데이터 생성
V_est = RC_model_1(X, t_vec, I_vec);
save('pulse_data.mat', 't_vec', 'I_vec', 'V_est') %t_vec, I_vec, V_est 저장

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




