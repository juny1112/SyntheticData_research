% ======================================================================
%  pulse + driving Loop  (n-RC 모델 참값 생성 + 2RC 피팅 + MS + τ-inequality)
% ======================================================================
clear; clc; close all;

%% ------------------------------------------------------------------
% (A-1) bimodal 로 n-RC 참 파라미터 만들기
% ------------------------------------------------------------------

% n     = 40;                     % RC 쌍 수
% n_mode1  = 10;                  % 모드1에 10개 할당
% tau_peak = [6 60];              % 두 피크 중심 [s]
% sigma    = [10 20];             % 표준편차 [s]
% tau_rng  = [0.1 150];           % τ 범위 [s]
% R0       = 0.001;                      
% 
% [X, R_i, tau_i] = bimodal_Rtau(n, n_mode1, tau_peak, sigma, tau_rng, R0);
% 
% % X_true = X;   % (1+2n)×1  참값 벡터
% n_cut = 20; % 원하는 n
% X_true = X([1 2:n_cut+1 n+2:n+n_cut+1]); % bimodal에서 R 너무 작은 값 제외   
% R_i      = R_i(1:n_cut);          % 저항 배열 축소
% tau_i    = tau_i(1:n_cut);        % τ 배열 축소
% n = n_cut; % 전압 생성시 n_cut 개의 nRC model 사용하기 위함


%% ------------------------------------------------------------------
% (A-2) multimodal 로 n-RC 참 파라미터 만들기
% ------------------------------------------------------------------

n        = 50;                  % 총 RC 쌍
n_mode12 = [15 15];              % mode 할당 n 개수
tau_pk   = [0.4 3 88];           % τ 중심 [s]
sigma = [0.07 0.18 0.2];         % 표준편차
areaRel  = [1e-3 5e-2 1];      % 면적 비
R_total  = 0.002;                % 총 저항
R_mode_sum   = R_total * areaRel / sum(areaRel);  % 각 mode 별 저항 [Ω]
tau_rng  = [0.1 400];           % 배치 범위
R0       = 1e-3;                 % R0

[X,R_i,tau_i] = multimodal_Rtau(n, n_mode12, ...
                              tau_pk, sigma, R_mode_sum, ...
                              tau_rng, R0);
X_true = X;  

%% ------------------------------------------------------------------
% (B) 전류 프로파일 목록 (struct 또는 .xlsx 혼용 가능)
% ------------------------------------------------------------------
% Pulse
dt = 1; t_start = 0;  t_end = 180;
pulse_start = 0; pulse_end = 180;

pulse.t = (t_start:dt:t_end)';       
pulse.I = zeros(size(pulse.t));
pulse.I(pulse.t>=pulse_start & pulse.t<=pulse_end) = 1;     

% Driving data
driving_paths = {
     'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\udds_unit_time_scaled_current.xlsx',
     'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\us06_unit_time_scaled_current.xlsx',
     'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_CITY1_time_scaled_current.xlsx',
     'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_HW1_time_scaled_current.xlsx'
 };

% Pulse 켜기
% driving_files = {pulse};
% 
% % Driving 켜기
% driving_files = driving_paths;
% 
% 둘 다 켜기
driving_files = [{pulse}; driving_paths];

%% ------------------------------------------------------------------
% (C) 마르코프 노이즈 설정
% ------------------------------------------------------------------
epsilon_percent_span = 1;     % ±5 %
initial_state = 51;
sigma = 5;
nSeeds = 10;                  % seed 1~10  (+ seed 0 = 원본)

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
        base_name = name;
    end

    %% 2) 참 전압 V_est (n-RC 모델 사용)
    V_est = RC_model_n(X_true, t_vec, I_vec, n);

    save([base_name '_data.mat'],'t_vec','I_vec','V_est');

    %% 3) 원본 플롯 (그림 그대로)
    figure('Name',sprintf('%s - I & V Plots',base_name),'NumberTitle','off');
    subplot(3,1,1);
    plot(t_vec,I_vec,'r-','LineWidth',1.2);
    xlabel('Time (sec)'); ylabel('Current (A)');
    title(sprintf('%s - Current Profile',base_name)); grid on;

    subplot(3,1,2);
    plot(t_vec,V_est,'b-','LineWidth',1.2);
    xlabel('Time (sec)'); ylabel('Voltage (V)');
    title(sprintf('%s - RC Model Voltage',base_name)); grid on;

    subplot(3,1,3);
    yyaxis left;  plot(t_vec,I_vec,'r-','LineWidth',1.2);
    ylabel('Current (A)'); ax=gca; ax.YColor='r';
    yyaxis right; plot(t_vec,V_est,'b-','LineWidth',1.2);
    ylabel('Voltage (V)'); ax.YColor='b';
    xlabel('Time (sec)');
    title(sprintf('%s - Current & Voltage',base_name));
    legend('Current (I)','Voltage (V)'); grid on;

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

    %% 5) 2RC 피팅 설정
    % para = [R0, R1, R2, tau1, tau2] (총 5개)
    para0 = [0.0012; 0.0008; 0.0012; 6; 70];
    lb = [0; 0; 0; 0.001; 0.001];
    ub = para0 * 10;
    % OCV=0 가정 (과전압만 보기)
    OCV = zeros(size(t_vec));

    % τ1 < τ2 부등식: [0 0 0 1 -1] · para ≤ 0
    A_lin = [0, 0, 0, 1, -1];
    b_lin = 0;

    para_hats = zeros(nSeeds+1, numel(para0) + 2);   % +exitflag +iter
    RMSE_list = zeros(nSeeds+1, 1);

    figure('Name',sprintf('%s - All Seed 2RC Fitting Results',base_name), ...
           'NumberTitle','off');

    seednames    = [{'Non_noise'}, arrayfun(@(s)sprintf('V_SD%d',s),1:nSeeds,'UniformOutput',false)];
    displaynames = [{'Non noise'}, arrayfun(@(s)sprintf('seed%d',s), 1:nSeeds,'UniformOutput',false)];

    for s = 1:(nSeeds+1)
        V_SD = noisedata.(seednames{s});

        problem = createOptimProblem('fmincon', ...
            'objective', @(p)RMSE_2RC(V_SD, p, t_vec, I_vec, OCV), ...
            'x0', para0, 'lb', lb, 'ub', ub, ...
            'Aineq', A_lin, 'bineq', b_lin, 'options', opt);

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

        V_0   = RC_model_2(para0, t_vec, I_vec, OCV);
        V_hat = RC_model_2(bestP, t_vec, I_vec, OCV);

        subplot(3,4,s);   % 3×4 = 12칸 (s=1…11 사용)
        plot(t_vec, V_SD, '-k', 'LineWidth', 1.5); hold on
        plot(t_vec, V_est,'-g', 'LineWidth', 1.5);
        plot(t_vec, V_hat,'-r', 'LineWidth', 1.5);
        plot(t_vec, V_0, '--b','LineWidth', 1.5);
        legend({'Synthetic','True','Fitted','Initial'}, 'Location','northeast');
        xlabel('Time (sec)'); ylabel('Voltage (V)');
        title(displaynames{s}); grid on;
        %xlim([200 400]);
    end
    sgtitle(sprintf('%s - 2RC Fitting for All Seeds', base_name));

    %% 6) 통계 출력 (원본 형식 유지)
    idxSeed = 2:size(para_hats,1);     % ← seed1~10 행만 선택

    mean_para = mean(para_hats(idxSeed,:),1);
    min_para  = min (para_hats(idxSeed,:),[],1);
    max_para  = max (para_hats(idxSeed,:),[],1);
    std_para  = std (para_hats(idxSeed,:),0,1);

    mean_RMSE = mean(RMSE_list(idxSeed));
    min_RMSE  = min (RMSE_list(idxSeed));
    max_RMSE  = max (RMSE_list(idxSeed));
    std_RMSE  = std (RMSE_list(idxSeed));
    
    fprintf('>> [%s] 파라미터 요약 통계 (2RC) :\n', base_name);
    fprintf('   [Mean]   R0 = %.8f, R1 = %.8f, R2 = %.8f, tau1 = %.8f, tau2 = %.8f, RMSE = %.8f\n', ...
        mean_para(1), mean_para(2), mean_para(3), mean_para(4), mean_para(5), mean_RMSE);
    fprintf('   [Min]    R0 = %.8f, R1 = %.8f, R2 = %.8f, tau1 = %.8f, tau2 = %.8f, RMSE = %.8f\n', ...
        min_para(1),  min_para(2),  min_para(3),  min_para(4),  min_para(5),  min_RMSE);
    fprintf('   [Max]    R0 = %.8f, R1 = %.8f, R2 = %.8f, tau1 = %.8f, tau2 = %.8f, RMSE = %.8f\n', ...
        max_para(1),  max_para(2),  max_para(3),  max_para(4),  max_para(5),  max_RMSE);
    fprintf('   [STD]    R0 = %.8f, R1 = %.8f, R2 = %.8f, tau1 = %.8f, tau2 = %.8f, RMSE = %.8f\n\n', ...
        std_para(1),  std_para(2),  std_para(3),  std_para(4),  std_para(5),  std_RMSE);
    
    %% 7) 테이블로 정리해 all_summary 에 저장
    rowNames = {'Mean','Min','Max','STD'};
    varNames = {'R0','R1','R2','tau1','tau2','RMSE'};
    T = table( ...
        [mean_para(1); min_para(1); max_para(1); std_para(1)], ...
        [mean_para(2); min_para(2); max_para(2); std_para(2)], ...
        [mean_para(3); min_para(3); max_para(3); std_para(3)], ...
        [mean_para(4); min_para(4); max_para(4); std_para(4)], ...
        [mean_para(5); min_para(5); max_para(5); std_para(5)], ...
        [mean_RMSE   ; min_RMSE   ; max_RMSE   ; std_RMSE  ], ...
        'VariableNames', varNames, 'RowNames', rowNames);

    all_para_hats.(base_name) = para_hats;
    all_rmse.(base_name)      = RMSE_list;
    all_summary.(base_name)   = T;
end

%% ------------------------------------------------------------------
%  RMSE (2-RC)
% ------------------------------------------------------------------
function cost = RMSE_2RC(data, para, t, I, OCV)
    model = RC_model_2(para, t, I, OCV);
    cost  = sqrt(mean((data - model).^2));
end
