clear; clc; close all;

% Driving data 목록 정의
driving_files = {
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\udds_unit_time_scaled_current.xlsx',
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\us06_unit_time_scaled_current.xlsx',
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_CITY1_time_scaled_current.xlsx',
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_HW1_time_scaled_current.xlsx'
};


for fileIdx = 1:length(driving_files)
    filename = driving_files{fileIdx};
    data = readtable(filename);

    t_vec = data.Var1; % 시간 [sec]
    I_vec = data.Var2; % 전류 [A]

    % 초기 파라미터 (2RC 모델용)
    X = [0.001 0.0005 0.0005 6 60]; % [R0[Ω], R1[Ω], R2[Ω], τ1[s], τ2[s]]

    % 2RC 모델 전압 생성
    V_est = RC_model_2(X, t_vec, I_vec);

    % Markov Noise 생성 파라미터
    original            = V_est;
    epsilon_percent_span = 5;     % ±[%]
    initial_state       = 51;     
    sigma               = 0.005;  
    numseeds            = 10;     % 시드 개수
    noisedata = struct;

    % 여러 시드로 V_SD1~V_SD10 생성
    for seed = 1:numseeds
        rng(seed);
        [noisy, ~, ~, ~, ~, ~] = MarkovNoise(original, epsilon_percent_span, initial_state, sigma);
        noisedata.(sprintf('V_SD%d', seed)) = noisy;
    end

    % 각 시드별로 cost surface 그리기
    for i = 1:numseeds
        V_SD = noisedata.(sprintf('V_SD%d', i));

        % τ1, τ2 값의 그리드 벡터
        tau1_vec = 10.^(linspace(-1, 1.1, 31));
        tau2_vec = 10.^(linspace( 1, 2.1, 61));

        % fmincon 옵션
        options_R = optimset( ...
            'Display',     'off', ...
            'MaxIter',     3000, ...
            'MaxFunEvals', 1e5, ...
            'TolFun',      1e-14, ...
            'TolX',        1e-15);

        cost_surface = zeros(numel(tau2_vec), numel(tau1_vec));

        % R0,R1,R2 최적화 반복
        for ii = 1:numel(tau1_vec)
            for jj = 1:numel(tau2_vec)
                
                p0_R = [0.0012 0.0006 0.0004 tau1_vec(ii) tau2_vec(jj)];
                lb_R = [0 0 0 tau1_vec(ii) tau2_vec(jj)];
                ub_R = [p0_R(1)*10 p0_R(2)*10 p0_R(3)*10 tau1_vec(ii) tau2_vec(jj)];

                [~, cost_surface(jj,ii)] = fmincon( ...
                    @(p) RMSE_2RC(V_SD, p, t_vec, I_vec), ...
                    p0_R, [],[],[],[], lb_R, ub_R, [], options_R);
            end
        end

        % 3D surface plot
        [T1, T2] = meshgrid(tau1_vec, tau2_vec);
        surf(T1, T2, cost_surface, 'EdgeColor','none','FaceColor','interp');
        view(3);
        xlabel('\tau_1 [s]');
        ylabel('\tau_2 [s]');
        zlabel('RMSE');
        colorbar;
        shading interp;
        hold on;

        % 최적 지점 찾기
        [min_cost, linIdx] = min(cost_surface(:));
        [r, c]             = ind2sub(size(cost_surface), linIdx);
        best_tau1          = tau1_vec(c);
        best_tau2          = tau2_vec(r);

        % 최적 지점 플롯
        hStar = plot3(best_tau1, best_tau2, min_cost, 'r*', ...
                      'MarkerSize',12,'LineWidth',2);

        % legend에 τ₁*, τ₂* 값 표시
        legend_str = sprintf('\\tau_1^* = %.3f s,  \\tau_2^* = %.3f s', ...
                             best_tau1, best_tau2);
        legend(hStar, legend_str, 'Location', 'best');

        hold off;
    end
end

% RMSE 계산 함수 (2RC 모델용)
function cost = RMSE_2RC(data, para, t, I)
    model = RC_model_2(para, t, I);
    cost  = sqrt(mean((data - model).^2));
end


