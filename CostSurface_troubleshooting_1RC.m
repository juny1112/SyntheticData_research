clear; clc; close all;

% Driving data 목록 정의
driving_files = {
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\udds_unit_time_scaled_current.xlsx',
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\us06_unit_time_scaled_current.xlsx',
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_CITY1_time_scaled_current.xlsx',
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_HW1_time_scaled_current.xlsx'
};


% loop (input: driving data(t, I) (+ ECM parameters(R, tau)) / output: V_est, V_SD, ECM parameters hat(R, tau), RMSE)
for fileIdx = 1:length(driving_files)
    filename = driving_files{fileIdx};
    data = readtable(filename);

    t_vec = data.Var1; % [sec]
    I_vec = data.Var2; % [Ah]
    N = numel(t_vec);
    OCV = zeros(N, 1);
    X = [0.001 0.001 10]; % [R0[ohm], R1[ohm], tau1[sec]]

    % 1RC 모델 전압 생성
    V_est = RC_model_1(X, t_vec, I_vec, OCV); % 초기값 직접 입력
    base_name = sprintf('load%d', fileIdx);
    save([base_name '_data.mat'], 't_vec', 'I_vec', 'V_est');


    % Markov Noise
    original = V_est;
    epsilon_percent_span = 1;     % ±1 %
    initial_state = 51;
    sigma         = 5;
    numseeds = 10; % seed 개수
    noisedata = struct;

    for seed = 1:numseeds
        rng(seed);
        [noisy,~,~,~,~,~] = MarkovNoise_idx(original, epsilon_percent_span, initial_state, sigma);
        noisedata.(sprintf('V_SD%d',seed)) = noisy;
    end

    noise_filename = ['noise_' base_name '_seed10.mat'];
    save(noise_filename, '-struct', 'noisedata'); % V_SD1~10 저장
    
    % Fitting
    R0 = 0.001; % R0 고정
    p0  = [0.0008 12];  % [R1; tau1]
    lb = [0 0.1];
    ub = p0*10;
    R1_vec = 0.0001:0.0001:0.0030;
    tau1_vec = 1:50;

    para_hats = zeros(numseeds, 5);
    RMSE_list = zeros(numseeds, 1);
    MaxIteration = zeros(numseeds, 1);
    cost_surface = zeros(length(tau1_vec), length(R1_vec));

    % Seed별 cost surface + 최적화 경로 시각화
    % 1) cost surface
    for i = 1:numseeds
        V_SD = noisedata.(sprintf('V_SD%d', i));

        for ii = 1:length(R1_vec)
            for jj = 1:length(tau1_vec)
                R1 = R1_vec(ii);
                tau1 = tau1_vec(jj);

                para_try = [R0, R1, tau1];
                cost_try = RMSE_1RC(V_SD,para_try, t_vec, I_vec, OCV);
                cost_surface(jj, ii) = cost_try;

            end
        end

        % 2) 3D surface plot
        [R1_grid, tau1_grid] = meshgrid(R1_vec, tau1_vec);

        figure('Name',sprintf('Seed %d Cost Surface',i),'NumberTitle','off');
        h = surface(R1_grid, tau1_grid, cost_surface);
        set(h, 'EdgeColor','none', 'FaceColor','interp');
        view(3); shading interp; hold on;
        xlabel('R1 [\Omega]'); ylabel('\tau_1 [sec]');
        zlabel('RMSE'); colorbar;
        title(sprintf('Seed %d',i));

        % 3) 매 iteration 마다 경로 그려짐
        options = optimset('display','iter', ... % 변경
            'MaxIter',3000, ... % fitting 시간 오래 걸려서3000번만(잘되는건 100번 이하로 다 됨. 많아봤자 3000번대)
            'MaxFunEvals',1e6, ...
            'OutputFcn',@plotIter, ... % 경로 그리는 함수
            'TolFun',1e-14, ...
            'TolX',1e-15);

        [para_hat,fval,exitflag,output] = fmincon( ...
            @(p) RMSE_1RC(V_SD, [R0, p], t_vec, I_vec, OCV), ...
            p0, [],[],[],[], lb, ub, [], options);
        para_hats(i, :) = [R0, para_hat, exitflag, output.iterations];

        % 결과 저장
        all_para_hats.(base_name) = para_hats;
        all_rmse.(base_name) = RMSE_list;
        
    end
end



% RMSE(cost) 함수
function cost = RMSE_1RC(data,para,t,I, OCV)
  model = RC_model_1(para, t, I, OCV);
  cost  = sqrt(mean((data - model).^2));
end

% plotIter
function stop = plotIter(p, optimValues, state)
    % fmincon 진행 경로 시각화용 OutputFcn
    stop = false;
    persistent hLine cnt
    switch state
        case 'init'
            hold on;
            % 궤적용 검은 선 생성 (범례 제외)
            hLine = plot3(NaN, NaN, NaN, '-k', 'LineWidth', 1.5, 'HandleVisibility', 'off');
            % 초기점 노란색 점 (범례 제외)
            scatter3(p(1), p(2), optimValues.fval, 80, 'y', 'filled', 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
            cnt = 0;
        case 'iter'
            cnt = cnt + 1;
            % 궤적 좌표 업데이트
            set(hLine, ...
                'XData', [get(hLine, 'XData'), p(1)], ...
                'YData', [get(hLine, 'YData'), p(2)], ...
                'ZData', [get(hLine, 'ZData'), optimValues.fval]);
            drawnow;
        case 'done'
            % 최종점 초록색 점으로 표시 (범례 제외)
            scatter3(p(1), p(2), optimValues.fval, 80, 'g', 'filled', 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
            clear hLine cnt;
    end
end






