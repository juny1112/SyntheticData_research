clear; clc; close all;

load('pulse_data.mat'); % V_est 데이터 로드

rng(9) %seed

%% MarkovNoise 함수 호출
original = V_est;
epsilon_percent_span = 5; % ±5%
initial_state = 51; % 임의로 지정. 처음 상태를 noise 0% 에서 시작(상태 개수 N=101개)
sigma = 0.005; % 그래프 보고 판단

[noisy, states, final_state, P, epsilon_vector, eps_values] = ...
    MarkovNoise(original, epsilon_percent_span, initial_state, sigma);

save('noise.mat', 'noisy') % V_sd 저장

%% Plot
figure;
subplot(2,1,1);
plot(t_vec, original, 'LineWidth',1.5); hold on;
plot(t_vec,noisy, 'LineWidth',1.5);
xlabel('Time (sec)'); ylabel('Voltage [V]');
legend('V_{est}','V_{SD}');
title('Markov Noise Voltage data');

subplot(2,1,2);
plot(t_vec, eps_values , 'LineWidth',1.5);
xlabel('Time (sec)'); ylabel('Noise');
title('Markov Noise');


