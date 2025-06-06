clear; clc; close all;

% Driving data 목록 정의
driving_files = {
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\udds_unit_time_scaled_current.xlsx',
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\us06_unit_time_scaled_current.xlsx',
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_CITY1_time_scaled_current.xlsx',
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_HW1_time_scaled_current.xlsx'
};

% para_hats 저장할 변수
all_para_hats = struct;
all_rmse = struct;

%% loop (input: driving data(t, I) (+ ECM parameters(R, tau)) / output: V_est, V_SD, ECM parameters hat(R, tau), RMSE)

for fileIdx = 1:length(driving_files)
    filename = driving_files{fileIdx};
    data = readtable(filename);

    t_vec = data.Var1; % [sec]
    I_vec = data.Var2; % [Ah]
    X = [0.001 0.001 10]; % [R0[ohm], R1[ohm], tau1[sec]]

    % % dt = 0.1 s 보간
    % dt_target = 0.1;                          
    % t_vec_fine = (t_vec(1) : dt_target : t_vec(end)).';   
    % I_vec_fine = interp1(t_vec, I_vec, t_vec_fine, 'linear');  
    % 
    % t_vec = t_vec_fine;
    % I_vec = I_vec_fine;
    % clear t_vec_fine I_vec_fine

    %% 1RC 모델 전압 생성
    V_est = RC_model_1(X, t_vec, I_vec);
    base_name = sprintf('load%d', fileIdx);
    save([base_name '_data.mat'], 't_vec', 'I_vec', 'V_est');

    % plot
    figure('Name', sprintf('%s - I & V Plots', base_name), 'NumberTitle', 'off');
    % 1) Plot input current
    subplot(3,1,1);
    plot(t_vec, I_vec, 'r-', 'LineWidth', 1.2);
    xlabel('Time (sec)');
    ylabel('Current (A)');
    title(sprintf('%s - Current Profile', base_name));
    grid on;
    % 2) Plot voltage
    subplot(3,1,2);
    plot(t_vec, V_est, 'b-', 'LineWidth', 1.2);
    xlabel('Time (sec)');
    ylabel('Voltage (V)');
    title(sprintf('%s - RC Model Voltage', base_name));
    grid on;
    % 3) Plot input current & voltage with yyaxis
    subplot(3,1,3);
    yyaxis left;
    plot(t_vec, I_vec, 'r-', 'LineWidth', 1.2);
    ylabel('Current (A)');
    ax = gca;
    ax.YColor = 'r';
    yyaxis right;
    plot(t_vec, V_est, 'b-', 'LineWidth', 1.2);
    ylabel('Voltage (V)');
    ax.YColor = 'b';
    xlabel('Time (sec)');
    title(sprintf('%s - Current & Voltage', base_name));
    legend('Current (I)', 'Voltage (V)');
    grid on;
    
    %% Markov Noise
    original = V_est;
    epsilon_percent_span = 3; % ±[%]
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

    %% Fitting
    para0 = [0.0012 0.0008 8];
    lb = [0 0 0.001];
    ub = para0 * 10;

    options = optimset('display','off', ...
        'MaxIter',1e4, ...
        'MaxFunEvals',1e5, ...
        'TolFun',1e-14, ...
        'TolX',1e-15);

    para_hats = zeros(numseeds, length(para0)+2);
    RMSE_list = zeros(numseeds, 1);
    
    figure('Name', sprintf('%s - All Seed Fitting Results', base_name), 'NumberTitle', 'off');

    for i = 1:numseeds
        V_SD = noisedata.(sprintf('V_SD%d', i));
        [para_hat, fval, exitflag, output] = fmincon(@(para)RMSE_1RC(V_SD,para,t_vec,I_vec),para0,[],[],[],[],lb,ub,[],options);    % 변경
        para_hats(i, :) = [para_hat, exitflag, output.iterations];

        V_0 = RC_model_1(para0, t_vec, I_vec);
        V_hat = RC_model_1(para_hat, t_vec, I_vec);

        RMSE_value = sqrt(mean((V_SD - V_hat).^2));
        RMSE_list(i) = RMSE_value; 

        % Plot V fitting
        subplot(2,5,i); % 2행 5열 subplot
        plot(t_vec,V_SD, '-k', LineWidth=1.5); hold on
        plot(t_vec,V_est,'-g', LineWidth=1.5);
        plot(t_vec,V_hat,'-r', LineWidth=1.5);
        plot(t_vec,V_0,'--b', LineWidth=1.5);

        title(sprintf('Seed %d', i));
        if i == 1
            legend({'Synthetic','Original','Fitted','Initial'}, 'Location','northeast');
        end

        if i == 1 || i == 6
            ylabel('Voltage (V)');
        end

        if i > 5
            xlabel('Time (sec)');
        end
        grid on;

    end
    
    %% Summary statistics of parameters
    sgtitle(sprintf('%s - 1RC Fitting for All Seeds', base_name));  % 전체 제목

    mean_para = mean(para_hats, 1);  min_para  = min(para_hats, [], 1);
    max_para  = max(para_hats, [], 1); var_para  = var(para_hats, 0, 1);  % 0은 "표본 분산" 의미
    
    mean_RMSE = mean(RMSE_list, 1); min_RMSE  = min(RMSE_list, [], 1);
    max_RMSE  = max(RMSE_list, [], 1);  var_RMSE  = var(RMSE_list, 0, 1);


    fprintf('>> [%s] 파라미터 요약 통계 :\n', base_name);
    fprintf('   [Mean]     R0 = %.8f, R1 = %.8f, tau = %.6f, RMSE = %.8f\n', mean_para(1), mean_para(2), mean_para(3), mean_RMSE); 
    fprintf('   [Min]      R0 = %.8f, R1 = %.8f, tau = %.6f, RMSE = %.8f\n', min_para(1),  min_para(2),  min_para(3), min_RMSE);
    fprintf('   [Max]      R0 = %.8f, R1 = %.8f, tau = %.6f, RMSE = %.8f\n', max_para(1),  max_para(2),  max_para(3), max_RMSE);
    fprintf('   [Variance] R0 = %.8f, R1 = %.8f, tau = %.6f, RMSE = %.8f\n\n', var_para(1), var_para(2), var_para(3), var_RMSE);

    all_para_hats.(base_name) = para_hats;
    all_rmse.(base_name) = RMSE_list;

end


%% Cost function (weight 무시)
function cost = RMSE_1RC(data,para,t,I)
    model = RC_model_1(para, t, I);
    cost = sqrt(mean((data - model).^2));
end


