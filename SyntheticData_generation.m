clear; clc; close all;

filename_input = 'pulse_data.mat';
filename_output = 'noise.mat';
% filename_input = 'UDDS_data.mat';
% filename_output = 'noise_UDDS.mat';

load(filename_input); % V_est 데이터 로드

rng(9) %seed

%% MarkovNoise 함수 호출
original = V_est;
epsilon_percent_span = 5; % ±5%
initial_state = 51; % 임의로 지정. 처음 상태를 noise 0% 에서 시작(상태 개수 N=101개)
sigma = 0.005; % 그래프 및 전이행렬 보고 판단

[noisy, states, final_state, P, epsilon_vector, eps_values] = ...
    MarkovNoise(original, epsilon_percent_span, initial_state, sigma);

save(filename_output, 'noisy') % V_sd 저장

%% Plot
figure;
subplot(2,1,1);
plot(t_vec, original,'LineWidth',1.5); hold on;
plot(t_vec,noisy,'--', 'LineWidth',1.5);
xlabel('Time (sec)'); ylabel('Voltage [V]');
legend('V_{est}','V_{SD}');
title('Markov Noise Voltage data');

subplot(2,1,2);
plot(t_vec, eps_values , 'LineWidth',1.5);
xlabel('Time (sec)'); ylabel('Noise');
title('Markov Noise');

difference = noisy - original;% = original X eps_k
figure;
plot(t_vec, difference,'LineWidth',1.5)
xlabel('Time (sec)'); ylabel('Noise');
title('noise(SD-original)');

