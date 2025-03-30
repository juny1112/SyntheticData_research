clear; clc; close all;

filename_input = 'UDDS_data.mat';
filename_output = 'noise_UDDS_seed10.mat'; % seed 10개 준 합성데이터
filename_combine = 'combine_data.mat'; % t_vec, I_vec, V_SD 데이터 합친 테이블

load(filename_input); % t_vec, I_vec, V_est 데이터 로드

%% MarkovNoise 함수 호출
original = V_est;
epsilon_percent_span = 5; % ±5%
initial_state = 51; % 임의로 지정. 처음 상태를 noise 0% 에서 시작(상태 개수 N=101개)
sigma = 0.005; % 그래프 및 전이행렬 보고 판단

numseeds = 10; % seed 개수(=생성 데이터 개수)
noisedata = struct;

for seed = 1:numseeds
    rng(seed);
    
    [noisy, states, final_state, P, epsilon_vector, eps_values] = ...
    MarkovNoise(original, epsilon_percent_span, initial_state, sigma);

    fieldname = sprintf('V_SD%d',seed);
    noisedata.(fieldname) = noisy;
    
    % figure;
    % subplot(2,1,1);
    % plot(t_vec, original,'LineWidth',1.5); hold on;
    % plot(t_vec,noisy,'--', 'LineWidth',1.5);
    % xlabel('Time (sec)'); ylabel('Voltage [V]');
    % legend('V_{est}','V_{SD}');
    % title(sprintf('Markov Noise Voltage Data (Seed = %d)', seed));
    % 
    % subplot(2,1,2);
    % plot(t_vec, eps_values , 'LineWidth',1.5);
    % xlabel('Time (sec)'); ylabel('Noise');
    % title('Markov Noise');
    % 
    % difference = noisy - original;% = original X eps_k
    % figure;
    % plot(t_vec, difference,'LineWidth',1.5)
    % xlabel('Time (sec)'); ylabel('Noise');
    % title(sprintf('Noise (SD - original) (Seed = %d)', seed));

end

save(filename_output, '-struct', 'noisedata') % V_SD1~10 저장

%% t_vec, I_vec와 noise 데이터 결합

columnname = [{'t_vec', 'I_vec'}, arrayfun(@(i) sprintf('V_SD%d', i), ...
    1:numseeds, 'UniformOutput', false)]; % 컬럼명 t, I, V_SD 저장

N = length(t_vec);

combined_matrix = zeros(N, 2+numseeds); % 2+numseeds = t + I + V_SD개수
combined_matrix(:,1) = t_vec(:);  
combined_matrix(:,2) = I_vec(:);  

for seed = 1:numseeds
    fieldname = sprintf('V_SD%d', seed);
    combined_matrix(:, 2+seed) = noisedata.(fieldname)(:);
end

% 행렬을 테이블로 변환하고, 각 열에 이름(컬럼명)을 부여
combine_T = array2table(combined_matrix, 'VariableNames', columnname);

save(filename_combine, 'combine_T');


% cf. 지금 이 코드는 V_SD를 'noise_UDDS_seed10.mat'에 field로 각각 하나씩 저장하고, 
% combine_data.mat에 테이블로 t,I,V_SD가 하나의 테이블에 나타내는 코드임.
% 굳이 필드에 하나씩 저장할 필요가 있나 싶은 문제가 있음. 
% 어떻게 저장하냐에 따라서 fitting 코드 달라짐 -> fitting 코드는 field에서 하나씩 피팅하는게 더 간편












