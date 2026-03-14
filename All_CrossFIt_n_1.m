% ======================================================================
%  pulse + driving Loop  (n-RC 모델 참값 생성 + 1RC 피팅 + MS)
% ======================================================================
clear; clc; close all;

%% ------------------------------------------------------------------
% (A-3) DRT 로 n-RC 참 파라미터 만들기
% ------------------------------------------------------------------
n       = 50;
tau_min = 0.1;         % s
tau_max = 400;         % s
A_tot   = 2.0e-3;      % 총 저항 [Ohm]
R0      = 1e-3;        % Ohm

% ---------- trimodal 예시 ----------
% tau_mode = [0.4; 3; 88];        
% sigma10  = [0.210; 0.200; 0.190];  
% w        = [1e-3 5e-2 1];

% ---------- bimodal 예시 ----------
% tau_mode = [3; 88];        
% sigma10  = [0.200; 0.190]; 
% w        = [5e-2 1];  

% ---------- unimodal 예시 ----------
tau_mode = [16];        
sigma10  = [0.200]; 
w        = [1]; 

% true 값 비교용
[mu10, tau_mean, tau_std, tau_median, mu_n, sigma_n, all_modes_table] = ...
    DRT_mu_sigma(tau_mode, sigma10);

% 저항 분포 생성
[theta, r_mode, r_theta, dth, R, tau, g_tau, A_modes, w_used] = ...
    DRT_Rtau(n, tau_min, tau_max, mu10, sigma10, A_tot, w);

% n-RC 참 파라미터 벡터
% RC_model_n이 [R0; R(:); tau(:)] 형식을 받는다고 가정
X_true = [R0; R(:); tau(:)];

%% ------------------------------------------------------------------
% (B) 전류 프로파일 목록 (struct 또는 .xlsx 혼용 가능)
% ------------------------------------------------------------------
% Pulse
dt = 1; 
t_start = 0;  
t_end = 180;
pulse_start = 0; 
pulse_end = 180;

pulse.t = (t_start:dt:t_end)';       
pulse.I = zeros(size(pulse.t));
pulse.I(pulse.t>=pulse_start & pulse.t<=pulse_end) = 1;     

% Driving data
driving_paths = {
"G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\us06_0725.xlsx"
"G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\WLTP_0725.xlsx"
"G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\BSL_CITY1_0725.xlsx"
"G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\BSL_CITY2_0726.xlsx"
"G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\BSL_HW1_0725.xlsx"
"G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\BSL_HW2_0725.xlsx"
"G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\hwfet_0725.xlsx"
"G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\udds_0725.xlsx"
};

% Pulse 켜기
% driving_files = {pulse};

% Driving 켜기
driving_files = driving_paths;

% 둘 다 켜기
% driving_files = [{pulse}; driving_paths];

%% ------------------------------------------------------------------
% (C) 마르코프 노이즈 설정
% ------------------------------------------------------------------
epsilon_percent_span = 2;     
initial_state = 51;
sigma = 5;
nSeeds = 10;                  

%% ------------------------------------------------------------------
% (D) MultiStart 설정
% ------------------------------------------------------------------
ms = MultiStart("UseParallel",true,"Display","off");
nStartPts = 20;
startPts  = RandomStartPointSet('NumStartPoints', nStartPts);

opt = optimset('display','off','MaxIter',1e3,'MaxFunEvals',1e4, ...
               'TolFun',1e-14,'TolX',1e-15);

%% ------------------------------------------------------------------
% (E) 루프 시작
% ------------------------------------------------------------------
all_para_hats = struct;
all_rmse      = struct;
all_summary   = struct;

for fileIdx = 1:length(driving_files)

    %% 1) 파일 읽기
    item = driving_files{fileIdx};
    if isstruct(item)
        t_vec = item.t(:);
        I_vec = item.I(:);
        base_name = sprintf('load%d',fileIdx);
    else
        tbl      = readtable(item);
        t_vec    = tbl{:,1};
        I_vec    = tbl{:,2};
        [~,name,~] = fileparts(item);
        base_name = char(name);
    end

    %% 2) 참 전압 V_est (n-RC 모델 사용)
    V_est = RC_model_n(X_true, t_vec, I_vec, n);

    save([base_name '_data.mat'],'t_vec','I_vec','V_est');

    %% 3) 원본 플롯
    figure('Name',sprintf('%s - I & V Plots',base_name),'NumberTitle','off');
    subplot(3,1,1);
    plot(t_vec,I_vec,'r-','LineWidth',1.2);
    xlabel('Time (sec)'); ylabel('Current (A)');
    title(sprintf('%s - Current Profile', base_name), 'Interpreter','none'); 
    grid on;

    subplot(3,1,2);
    plot(t_vec,V_est,'b-','LineWidth',1.2);
    xlabel('Time (sec)'); ylabel('Voltage (V)');
    title(sprintf('%s - nRC Model Voltage', base_name), 'Interpreter','none'); 
    grid on;

    subplot(3,1,3);
    yyaxis left;  
    plot(t_vec,I_vec,'r-','LineWidth',1.2);
    ylabel('Current (A)'); 
    ax = gca; ax.YColor = 'r';

    yyaxis right; 
    plot(t_vec,V_est,'b-','LineWidth',1.2);
    ylabel('Voltage (V)'); 
    ax.YColor = 'b';

    xlabel('Time (sec)');
    title(sprintf('%s - Current & Voltage', base_name), 'Interpreter','none');
    legend('Current (I)','Voltage (V)'); 
    grid on;

    %% 4) Markov 0~10 세트 생성
    noisedata = struct;
    noisedata.Non_noise = V_est;   % seed 0 = 원본
    for seed = 1:nSeeds
        rng(seed);
        [noisy,~,~,~,~,~] = MarkovNoise_idx(V_est, ...
            epsilon_percent_span, initial_state, sigma);
        noisedata.(sprintf('V_SD%d',seed)) = noisy;
    end
    save(['noise_' base_name '_seed10.mat'],'-struct','noisedata');

    %% 5) 1RC 피팅 설정
    % para = [R0, R1, tau1]
    para0 = [0.0012; 0.0008; 12];
    lb    = [0; 0; 0.001];
    ub    = para0 * 10;

    % OCV=0 가정 (과전압만 보기)
    OCV = zeros(size(t_vec));

    para_hats = zeros(nSeeds+1, numel(para0) + 2);   % + exitflag +iter
    RMSE_list = zeros(nSeeds+1, 1);

    figure('Name',sprintf('%s - All Seed 1RC Fitting Results',base_name), ...
           'NumberTitle','off');

    seednames    = [{'Non_noise'}, arrayfun(@(s)sprintf('V_SD%d',s),1:nSeeds,'UniformOutput',false)];
    displaynames = [{'Non noise'}, arrayfun(@(s)sprintf('seed%d',s), 1:nSeeds,'UniformOutput',false)];

    for s = 1:(nSeeds+1)
        V_SD = noisedata.(seednames{s});

        problem = createOptimProblem('fmincon', ...
            'objective', @(p) RMSE_1RC(V_SD, p, t_vec, I_vec, OCV), ...
            'x0', para0, 'lb', lb, 'ub', ub, 'options', opt);

        [bestP, bestFval, eflg, ~, sltns] = run(ms, problem, startPts);

        % ----- iterations 안전하게 뽑기 -----
        if isempty(sltns)
            iter = NaN;
        else
            idx  = find([sltns.Fval] == bestFval, 1);
            iter = sltns(idx).Output.iterations;
        end

        para_hats(s,:) = [bestP.' eflg iter];
        RMSE_list(s)   = bestFval;

        V_0   = RC_model_1(para0, t_vec, I_vec, OCV);
        V_hat = RC_model_1(bestP, t_vec, I_vec, OCV);

        subplot(3,4,s);   % 3×4 = 12칸 (s=1…11 사용)
        plot(t_vec, V_SD, '-k', 'LineWidth', 1.5); hold on
        plot(t_vec, V_est,'-g', 'LineWidth', 1.5);
        plot(t_vec, V_hat,'-r', 'LineWidth', 1.5);
        plot(t_vec, V_0, '--b','LineWidth', 1.5);
        legend({'Synthetic','True','Fitted','Initial'}, 'Location','northeast');
        xlabel('Time (sec)'); ylabel('Voltage (V)');
        title(displaynames{s}); grid on;
        xlim([200 400]);
    end
    sgtitle(sprintf('%s - 1RC Fitting for All Seeds', base_name), 'Interpreter','none');

    %% 6) 통계 출력
    idxSeed = 2:size(para_hats,1);     % seed1~10 행만 선택

    mean_para = mean(para_hats(idxSeed,:),1);
    min_para  = min (para_hats(idxSeed,:),[],1);
    max_para  = max (para_hats(idxSeed,:),[],1);
    std_para  = std (para_hats(idxSeed,:),0,1);

    mean_RMSE = mean(RMSE_list(idxSeed));
    min_RMSE  = min (RMSE_list(idxSeed));
    max_RMSE  = max (RMSE_list(idxSeed));
    std_RMSE  = std (RMSE_list(idxSeed));
    
    fprintf('>> [%s] 파라미터 요약 통계 (1RC) :\n', base_name);
    fprintf('   [Mean]   R0 = %.8f, R1 = %.8f, tau1 = %.8f, RMSE = %.8f\n', ...
        mean_para(1), mean_para(2), mean_para(3), mean_RMSE);
    fprintf('   [Min]    R0 = %.8f, R1 = %.8f, tau1 = %.8f, RMSE = %.8f\n', ...
        min_para(1),  min_para(2),  min_para(3),  min_RMSE);
    fprintf('   [Max]    R0 = %.8f, R1 = %.8f, tau1 = %.8f, RMSE = %.8f\n', ...
        max_para(1),  max_para(2),  max_para(3),  max_RMSE);
    fprintf('   [STD]    R0 = %.8f, R1 = %.8f, tau1 = %.8f, RMSE = %.8f\n\n', ...
        std_para(1),  std_para(2),  std_para(3),  std_RMSE);
    
    %% 7) 테이블로 정리
    rowNames = {'Mean','Min','Max','STD'};
    varNames = {'R0','R1','tau1','RMSE'};

    T = table( ...
        [mean_para(1); min_para(1); max_para(1); std_para(1)], ...
        [mean_para(2); min_para(2); max_para(2); std_para(2)], ...
        [mean_para(3); min_para(3); max_para(3); std_para(3)], ...
        [mean_RMSE   ; min_RMSE   ; max_RMSE   ; std_RMSE  ], ...
        'VariableNames', varNames, 'RowNames', rowNames);

    all_para_hats.(base_name) = para_hats;
    all_rmse.(base_name)      = RMSE_list;
    all_summary.(base_name)   = T;

    %% 8) g(τ) 마킹(τ_mode/τ_mean/τ_median/mean fitted tau1)
    mean_tau1 = mean_para(3);

    fig_gtau = figure('Name', sprintf('%s - g(tau) plot', base_name), ...
        'NumberTitle','off', 'Color','w');
    semilogx(tau, g_tau, 'LineWidth', 1.8); grid on; hold on
    xlabel('$\tau\ (\mathrm{s})$','Interpreter','latex');
    ylabel('$g(\tau)\;[\Omega/\mathrm{s}]$','Interpreter','latex');
    title('$g(\tau)\ \mathrm{vs}\ \tau$','Interpreter','latex');

    c_mode   = [0.10 0.10 0.10];  ls_mode   = '-';    
    c_mean   = [0.10 0.60 0.10];  ls_mean   = '--';   
    c_median = [0.10 0.30 0.90];  ls_median = '-.';   
    c_fit1   = [0.85 0.25 0.15];  ls_fit1   = ':';    

    if exist('tau_min','var') && exist('tau_max','var')
        xlim([tau_min, tau_max]);
    end

    if ~isempty(tau_mode)
        xline(tau_mode(1), ls_mode, 'Color', c_mode, 'LineWidth', 1.5, ...
            'DisplayName', '$\tau_{mode}$');
        arrayfun(@(v) xline(v, ls_mode, 'Color', c_mode, 'LineWidth', 1.5, ...
            'HandleVisibility','off'), tau_mode(2:end));
    end

    if ~isempty(tau_mean)
        xline(tau_mean(1), ls_mean, 'Color', c_mean, 'LineWidth', 1.5, ...
            'DisplayName', '$\tau_{\mathrm{mean}}$');
        arrayfun(@(v) xline(v, ls_mean, 'Color', c_mean, 'LineWidth', 1.5, ...
            'HandleVisibility','off'), tau_mean(2:end));
    end

    if ~isempty(tau_median)
        xline(tau_median(1), ls_median, 'Color', c_median, 'LineWidth', 1.5, ...
            'DisplayName', '$\tau_{\mathrm{median}}$');
        arrayfun(@(v) xline(v, ls_median, 'Color', c_median, 'LineWidth', 1.5, ...
            'HandleVisibility','off'), tau_median(2:end));
    end

    xline(mean_tau1, ls_fit1, 'Color', c_fit1, 'LineWidth', 1.7, ...
        'DisplayName', '$\overline{\tau}_1^{\,\mathrm{fit}}$');

    leg = legend('show','Location','best');
    set(leg,'Interpreter','latex');

end

%% ------------------------------------------------------------------
%  RMSE (1-RC)
% ------------------------------------------------------------------
function cost = RMSE_1RC(data, para, t, I, OCV)
    model = RC_model_1(para, t, I, OCV);
    cost  = sqrt(mean((data - model).^2));
end