
% TroubleShooting_MS최적화_1RC + iteration step

clear; clc; close all;

% Driving data 목록 정의
driving_files = {
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\udds_unit_time_scaled_current.xlsx',
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\us06_unit_time_scaled_current.xlsx',
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_CITY1_time_scaled_current.xlsx',
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_HW1_time_scaled_current.xlsx'
};

% MultiStart
ms = MultiStart( ...
    'UseParallel', true , ...
    'Display', 'iter');

for fileIdx = 1:length(driving_files)
    data = readtable(driving_files{fileIdx});
    t_vec = data.Var1;    % [sec]
    I_vec = data.Var2;    % [A]

    % 초기 모델 전압 생성 및 저장
    X0    = [0.001, 0.001, 60];   % [R0, R1, τ1]
    V_est = RC_model_1(X0, t_vec, I_vec);
    save(sprintf('load%d_data.mat', fileIdx), 't_vec', 'I_vec', 'V_est');

    % Markov Noise 생성
    eps_span = 5; 
    init_state = 51; 
    sigma = 0.005; 
    numSeeds = 10;

    for s = 1:numSeeds
        rng(s);
        [noisy, ~, ~, ~, ~, ~] = MarkovNoise(V_est, eps_span, init_state, sigma);
        noisedata.(sprintf('V_SD%d', s)) = noisy;
    end
    save(sprintf('noise_load%d_seed10.mat', fileIdx), '-struct', 'noisedata');

    % fmincon 옵션 설정
    R0 = 0.001;
    p0 = [0.0008, 50];
    lb = [0, 0.1];
    ub = [0.003, 140];
    opts = optimoptions('fmincon', 'Display', 'iter', 'MaxIterations', 1e3, ...
         'MaxFunctionEvaluations', 1e4, 'TolFun', 1e-14, 'TolX', 1e-15, ...
         'OutputFcn', @plotIter);

    results = zeros(numSeeds, numel(p0)+2);  % [R1, τ1, exitflag, iterations]
    startPts = RandomStartPointSet('NumStartPoints', 20);

    % Seed별 최적화
    for i = 1:numSeeds
        V_SD = noisedata.(sprintf('V_SD%d', i));

        % 1) Cost-surface 계산
        R1_vec   = linspace(lb(1), ub(1), 50);
        tau1_vec = linspace(lb(2), ub(2), 100);
        cost_surf = zeros(numel(tau1_vec), numel(R1_vec));
        for ii = 1:numel(R1_vec)
            for jj = 1:numel(tau1_vec)
                cost_surf(jj, ii) = RMSE_1RC(V_SD, [R0, R1_vec(ii), tau1_vec(jj)], t_vec, I_vec);
            end
        end

        % 2) 3D Cost-surface Plot 및 최적점(red star)
        [R1_grid, tau1_grid] = meshgrid(R1_vec, tau1_vec);
        figure('Name', sprintf('Seed %d Cost Surface', i), 'NumberTitle', 'off');
        surf(R1_grid, tau1_grid, cost_surf, 'EdgeColor', 'none', 'FaceColor', 'interp', 'HandleVisibility', 'off');
        view(3); shading interp; hold on;
        xlabel('R1 [Ω]'); ylabel('τ_1 [sec]'); zlabel('RMSE');
        title(sprintf('Cost Surface (Seed %d)', i)); colorbar;

        % Cost-surface 최적점 찾기
        [minVal, minIdx] = min(cost_surf(:));
        [row, col] = ind2sub(size(cost_surf), minIdx);
        optR1  = R1_vec(col);
        optTau = tau1_vec(row);
        scatter3(optR1, optTau, minVal, 150, 'r*', 'LineWidth', 2, ...
                 'DisplayName', sprintf('Cost Surface Opt: R1=%.4f, τ1=%.1f', optR1, optTau));

        % 3) MultiStart 최적화 실행
        problem = createOptimProblem('fmincon', 'objective', @(p) RMSE_1RC(V_SD, [R0, p], t_vec, I_vec), ...
                   'x0', p0, 'lb', lb, 'ub', ub, 'options', opts);
        [bestP, bestFval, exitflag, ~, sols] = run(ms, problem, startPts);

        % MultiStart 최적값 표시(green circle)
        scatter3(bestP(1), bestP(2), bestFval, 150, 'go', 'LineWidth', 2, ...
                 'DisplayName', sprintf('MultiStart Opt: R1=%.4f, τ1=%.1f', bestP(1), bestP(2)));
        legend('Location', 'best', 'AutoUpdate', 'off');

        % 4) 반복 횟수 기록
        idx = find([sols.Fval] == bestFval, 1);
        results(i, :) = [bestP, exitflag, sols(idx).Output.iterations];
    end

    % 결과 저장
    save(sprintf('results_load%d.mat', fileIdx), 'results');
end

%% 보조 함수들
function cost = RMSE_1RC(data, para, t, I)
    cost = sqrt(mean((data - RC_model_1(para, t, I)).^2));
end

function stop = plotIter(p, optimValues, state)
    stop = false;
    persistent hLine cnt
    switch state
        case 'init'
            hold on;
            hLine = plot3(nan, nan, nan, '-k', 'LineWidth', 1.5, 'HandleVisibility', 'off');
            scatter3(p(1), p(2), optimValues.fval, 80, 'y', 'filled', ...
                'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
            cnt = 0;
        case 'iter'
            cnt = cnt + 1;
            set(hLine, 'XData', [get(hLine,'XData'), p(1)], ...
                       'YData', [get(hLine,'YData'), p(2)], ...
                       'ZData', [get(hLine,'ZData'), optimValues.fval]);
            drawnow;
        case 'done'
            scatter3(p(1), p(2), optimValues.fval, 80, 'g', 'filled', ...
                     'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
            clear hLine cnt;
    end
end

