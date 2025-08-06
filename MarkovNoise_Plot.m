clear; clc; close all;

%% 1) t_vec 및 파라미터 설정
t_end = 2000;    % [sec]
dt    = 1;    % [sec]
t_vec = (0:dt:t_end)';  % 세로 벡터
original = zeros(size(t_vec));  
epsilon_percent_span = 5;  % ±[%]
initial_state        = 51; % 임의의 시작 상태
sigma                = 5;
numseeds             = 10; 

%% 2) 시드별 eps_values 생성 및 플로팅
figure('Name','Seed별 eps\_values','Position',[100 100 800 1200]);

for seed = 1:numseeds
    rng(seed);
    % MarkovNoise 함수 호출
    [~, ~, ~, ~, ~, eps_values] = MarkovNoise_idx(original, epsilon_percent_span, initial_state, sigma);
    % 여기서 eps_values는 길이(length(t_vec)) × 1 벡터

    % 서브플롯 (10행 1열 중 i번째)
    subplot(numseeds,1,seed);
    plot(t_vec, eps_values, 'LineWidth', 1.0);
    ylabel(sprintf('Noise (seed=%d)', seed));
    xlabel('Time (s)');
    grid on;
end
sgtitle('Markov noise for each seed');

