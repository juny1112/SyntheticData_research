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
    X = [0.001 0.0005 0.0005 6 60]; % [R0[ohm], R1[ohm], R2[ohm], tau1[sec], tau2[sec]]

    % 2RC 모델 전압 생성
    V_est = RC_model_2(X, t_vec, I_vec); % 초기값 직접 입력
    base_name = sprintf('load%d', fileIdx);
    save([base_name '_data.mat'], 't_vec', 'I_vec', 'V_est');

    % Markov Noise
    original = V_est;
    epsilon_percent_span = 2; % ±[%]
    initial_state = 51;
    sigma = 0.005;
    numseeds = 10; % seed 개수
    noisedata = struct;

    for seed = 1:numseeds
        rng(seed);
        [noisy, ~, ~, ~, ~, ~] = MarkovNoise(original, epsilon_percent_span, initial_state, sigma);
        noisedata.(sprintf('V_SD%d', seed)) = noisy;
    end

    noise_filename = ['noise_' base_name '_seed10.mat'];
    save(noise_filename, '-struct', 'noisedata'); % V_SD1~10 저장
    
    % Fitting
    p0  = [0.0012 0.0006 0.0004 5 70];  % [tau1; tau2]
    lb = [0 0 0 0.1 0.1];
    ub = p0*10;
    
    tau1_vec = 10.^(linspace(-1,1.1,21));
    tau2_vec = 10.^(linspace(1,2.1,21));40:100; % 보통 50~70 사이. 엄청 큰 값도 있어서 어떻게 잡아야될지 모르겠음

    para_hats = zeros(numseeds, 8);
    RMSE_list = zeros(numseeds, 1);
    cost_surface = zeros(length(tau2_vec), length(tau1_vec));

    % Seed별 cost surface + 최적화 경로 시각화
    % 1) cost surface
    for i = 1:numseeds
        V_SD = noisedata.(sprintf('V_SD%d', i));

        % R0, R1, R2 fitting
        R_init = [0.0012 0.0006 0.0004];
        p0_R   = R_init;       % fmincon 에 넘길 초기값(스칼라)
        lb_R   = [0 0 0];
        ub_R   = R_init * 10;
       
        % fmincon option
        options_R = optimset( ...
            'Display',      'off', ...
            'MaxIter',      3000, ... % R fitting 하는거니까 줄임
            'MaxFunEvals',  1e6, ...
            'TolFun',       1e-14, ...
            'TolX',         1e-15);

        for ii = 1:length(tau1_vec)
            for jj = 1:length(tau2_vec)
                tau1 = tau1_vec(ii);
                tau2 = tau2_vec(jj);

                % 고정된 tau1,2에서 R0 R1 R2 fitting
                [R_hat, fval] = fmincon( ...
                    @(p) RMSE_2RC(V_SD, [p, tau1, tau2], t_vec, I_vec), ...
                    p0_R, [],[],[],[], lb_R, ub_R, [], options_R);

                cost_surface(jj, ii) = fval;

            end
        end

        % 3D surface plot
        [tau1_grid, tau2_grid] = meshgrid(tau1_vec, tau2_vec);

        figure;
        h = surface(tau1_grid, tau2_grid, cost_surface);
        set(h, 'EdgeColor', 'none', 'FaceColor', 'interp');

        view(3);
        xlabel('\tau_1 [sec]');
        ylabel('\tau_2 [sec]');
        zlabel('RMSE');
        colorbar;

        shading interp;

        % 3) 매 iteration 마다 경로 그려짐 % 여기서부터 고쳐야함~~~~~
        options = optimset('display','iter', ... % 변경
            'MaxIter',3000, ... % fitting 시간 오래 걸려서3000번만
            'MaxFunEvals',1e6, ...
            'OutputFcn',@plotIter, ... % 경로 그리는 함수
            'TolFun',1e-14, ...
            'TolX',1e-15);

        [para_hat,fval,exitflag,output] = fmincon( ...
            @(para) RMSE_2RC(V_SD, para, t_vec, I_vec), ...
            p0, [],[],[],[], lb, ub, [], options);
        para_hats(i, :) = [para_hat, exitflag, output.iterations, fval];

        % 결과 저장
        all_para_hats.(base_name) = para_hats;
        all_rmse.(base_name) = RMSE_list;
        
    end
end

% RMSE(cost) 함수
function cost = RMSE_2RC(data,para,t,I)
  model = RC_model_2(para, t, I);
  cost  = sqrt(mean((data - model).^2));
end

% plotIter
function stop = plotIter(p, optimValues, state)
  stop = false;
  persistent hLine cnt
  switch state
    case 'init'
      hold on;
      hLine = plot3(NaN, NaN, NaN, '-k', 'LineWidth', 1.5);
      cnt = 0;

    case 'iter'
    cnt = cnt + 1;
    % 1) 시작점: 노랑
    if cnt == 1
        scatter3(p(4), p(5), optimValues.fval, ...
                 40, 'y', 'filled', 'MarkerEdgeColor', 'k');
    end

    % 2) 선 업데이트 (tau1 vs tau2 vs cost)
    x = [get(hLine, 'XData'), p(4)];
    y = [get(hLine, 'YData'), p(5)];
    z = [get(hLine, 'ZData'), optimValues.fval];
    set(hLine, 'XData', x, 'YData', y, 'ZData', z);

    % 3) 10스텝마다 중간: 빨강
    if mod(cnt,10) == 0
        scatter3(p(4), p(5), optimValues.fval, ...
                 40, 'r', 'filled', 'MarkerEdgeColor', 'k');
    end

    drawnow;

    case 'done'
      % 4) 최종점
      scatter3(p(4), p(5), optimValues.fval, ...
               40, 'g', 'filled', 'MarkerEdgeColor', 'k');
      clear hLine cnt
  end
end


