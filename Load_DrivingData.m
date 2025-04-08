clear; clc; close all;

%% load driving data
filename = 'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE\udds_unit_time_scaled_current.xlsx';

data = readtable(filename);

% parameters
t_vec = data.time; %[sec]
I_vec = data.scaled_current; %[Ah]
X = [0.001 0.001 10]; % 임의로 [R0[ohm], R1[ohm], tau1[sec]] 설정

% Plot current (확인용)
figure;
plot(t_vec, I_vec, 'r-');
xlabel('Time(sec)');
ylabel('Current(C)');
title('Current Profile of UDDS');
grid on;


%% 1RC model -> 전압 데이터 생성
V_est = RC_model_1(X, t_vec, I_vec);
save('UDDS_data.mat', 't_vec', 'I_vec', 'V_est') %t_vec, I_vec, V_est 저장

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

yyaxis right;
plot(t_vec, V_est, 'b-'); 
ylabel('Voltage (V)');
ax.YColor = 'b';

xlabel('Time (sec)');
title('Current Profile and 1RC Model Voltage');
grid on;
legend('Current (I)', 'Voltage (V)');


