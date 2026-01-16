clear; clc; close all;

% Driving data 목록 정의
driving_files = {
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\udds_unit_time_scaled_current.xlsx',
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\us06_unit_time_scaled_current.xlsx',
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_CITY1_time_scaled_current.xlsx',
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_HW1_time_scaled_current.xlsx'
};

% para_hats 저장할 변수
all_para_hats = struct;
all_rmse      = struct;

% loop (input: driving data(t, I) -> output: V_est, V_SD, ECM parameters hat(R, tau), RMSE)
for fileIdx = 1:length(driving_files)
    filename = driving_files{fileIdx};
    data     = readtable(filename);

    t_vec = data.Var1;   % [sec]
    I_vec = data.Var2;   % [Ah]
    N = numel(t_vec);
    OCV = zeros(N, 1);
    X     = [0.001 0.001 10];  % [R0[Ω], R1[Ω], tau1[sec]]

    %% 1RC 모델 전압 생성
    V_est    = RC_model_1(X, t_vec, I_vec, OCV);
    base_name = sprintf('load%d', fileIdx);
    save([base_name '_data.mat'], 't_vec', 'I_vec', 'V_est');
    
    %% Markov Noise 생성
    original             = V_est;
    epsilon_percent_span = 5;     % ±1 %
    initial_state = 51;
    sigma         = 5;
    nSeeds        = 10;           % seed 1~10 (+ Non_noise = V_est)
    noisedata            = struct;

    for seed = 1:nSeeds
        rng(seed);
        [noisy,~,~,~,~,~] = MarkovNoise_idx(original, epsilon_percent_span, initial_state, sigma);
        noisedata.(sprintf('V_SD%d',seed)) = noisy;
    end

    noise_filename = ['noise_' base_name '_seed10.mat'];
    save(noise_filename, '-struct', 'noisedata');  % V_SD1~10 저장

    %% Cost Surface 계산 및 3D 플롯
    for i = 1:nSeeds
        V_SD   = noisedata.(sprintf('V_SD%d', i));
        R0     = 0.001;
        R1_vec = 0.0001:0.0001:0.0030;
        tau1_vec = 1:50;

        cost_surface = zeros(length(tau1_vec), length(R1_vec));

        % cost surface 계산
        for r1_idx = 1:length(R1_vec)
            for tau_idx = 1:length(tau1_vec)
                R1       = R1_vec(r1_idx);
                tau1     = tau1_vec(tau_idx);
                para_try = [R0, R1, tau1];
                cost_surface(tau_idx, r1_idx) = RMSE_1RC(V_SD, para_try, t_vec, I_vec, OCV);
            end
        end

        % Meshgrid 생성
        [R1_grid, tau1_grid] = meshgrid(R1_vec, tau1_vec);

        % Figure 생성 및 Plot
        figure('Name', sprintf('Cost Surface seed%d', i), 'NumberTitle', 'off');
        hSurf = surface(R1_grid, tau1_grid, cost_surface);
        set(hSurf, 'EdgeColor', 'none', 'FaceColor', 'interp');
        view(3);
        grid on; 
        xlabel('R_1 [\Omega]');
        ylabel('\tau_1 [sec]');
        zlabel('RMSE');
        colorbar;
        shading interp;
        hold on;

        % 최적 지점 찾기
        [min_cost, linIdx] = min(cost_surface(:));
        [r, c]             = ind2sub(size(cost_surface), linIdx);
        best_tau1          = tau1_vec(r);
        best_R1            = R1_vec(c);

        % 최적 지점 마커
        hStar = plot3(best_R1, best_tau1, min_cost, 'r*', ...
                      'MarkerSize', 12, 'LineWidth', 2);

        % 범례 추가
        legend_str = sprintf('R_1^* = %.3f, \\tau_1^* = %.3f', best_R1, best_tau1);
        legend(hStar, legend_str, 'Location', 'best');

        % 그래프 제목
        title(sprintf('Cost Surface seed%d', i));

        hold off;
    end
    
end

%% Cost function (weight 무시)
function cost = RMSE_1RC(data, para, t, I, ocv)
    model = RC_model_1(para, t, I, ocv);
    cost  = sqrt(mean((data - model).^2));
end
