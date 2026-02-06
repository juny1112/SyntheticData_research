% ======================================================================
%  pulse + driving Loop  (n-RC 모델 참값 생성 + 2RC 피팅 + MS + τ-inequality)
% ======================================================================
clear; clc; close all;
%% ------------------------------------------------------------------
% (A-3) DRT 로 n-RC 참 파라미터 만들기
% ------------------------------------------------------------------
n       = 50;
tau_min = 0.1;         % s
tau_max = 400;         % s
A_tot = 2.0e-3;        % 총 저항 [Ω]
R0     = 1e-3;         % Ω

% bimodal
tau_mode = [3; 88];        % s
sigma10  = [0.200; 0.190]; % decades
w        = [5e-2 1];       % 모드 면적 비 (저항 비)

% true 값 비교용
[mu10, tau_mean, tau_std, tau_median, mu_n, sigma_n, all_modes_table] = ...
    DRT_mu_sigma(tau_mode, sigma10);
% 저항 분포 생성
[theta, r_mode, r_theta, dth, R, tau, g_tau, A_modes, w_used] = ...
    DRT_Rtau(n, tau_min, tau_max, mu10, sigma10, A_tot, w);
X_true = [R0; R(:); tau(:)];   % 열벡터로

%% ------------------------------------------------------------------
% (B) 전류 프로파일 목록
% ------------------------------------------------------------------
% Driving data
driving_paths = {
     "G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\BSL_CITY1_0725.xlsx"
 };

driving_files = driving_paths;

%% ------------------------------------------------------------------
% (C) 마르코프 노이즈 설정
% ------------------------------------------------------------------
epsilon_percent_span = 0;     % ±5 %
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
               'TolFun',eps,'TolX',eps);

%% ------------------------------------------------------------------
% [추가됨] 엑셀 저장 경로 설정
% ------------------------------------------------------------------
excel_save_dir = 'G:\공유 드라이브\Battery Software Group (2025)\Internship\25년_동계인턴\전기영\Figure\4주차 FFT\FFT_random sine wave';
if ~exist(excel_save_dir, 'dir')
    mkdir(excel_save_dir);
end

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
    % save([base_name '_data.mat'],'t_vec','I_vec','V_est'); % 필요시 주석 해제
    
    %% 3) 원본 플롯 (생략 가능)
    % (속도 향상을 위해 생략하거나 필요하면 주석 해제하여 사용하세요)
    
    %% 4) Markov 0~10 세트 생성
    noisedata = struct;
    noisedata.Non_noise = V_est;   % seed 0 = 원본
    for seed = 1:nSeeds
        rng(seed);
        [noisy,~,~,~,~,~] = MarkovNoise_idx(V_est, ...
            epsilon_percent_span, initial_state, sigma);
        noisedata.(sprintf('V_SD%d',seed)) = noisy;
    end
    
    %% 5) 2RC 피팅 설정
    % para = [R0, R1, R2, tau1, tau2] (총 5개)
    para0 = [0.0012; 0.0008; 0.0012; 6; 70];
    lb = [0; 0; 0; 0.001; 0.001];
    ub = para0 * 10;
    OCV = zeros(size(t_vec));
    
    % τ1 < τ2 부등식
    A_lin = [0, 0, 0, 1, -1];
    b_lin = 0;
    
    para_hats = zeros(nSeeds+1, numel(para0) + 2);
    RMSE_list = zeros(nSeeds+1, 1);
    
    % Fitting Loop
    for s = 1:(nSeeds+1)
        if s == 1
            seed_key = 'Non_noise';
        else
            seed_key = sprintf('V_SD%d', s-1);
        end
        V_SD = noisedata.(seed_key);
        
        problem = createOptimProblem('fmincon', ...
            'objective', @(p)RMSE_2RC(V_SD, p, t_vec, I_vec, OCV), ...
            'x0', para0, 'lb', lb, 'ub', ub, ...
            'Aineq', A_lin, 'bineq', b_lin, 'options', opt);
            
        [bestP, bestFval, eflg, ~, sltns] = run(ms, problem, startPts);
        
        if isempty(sltns)
            iter = NaN;
        else
            idx  = find([sltns.Fval] == bestFval, 1);
            iter = sltns(idx).Output.iterations;
        end
        para_hats(s,:) = [bestP.' eflg iter];
        RMSE_list(s)   = bestFval;
    end
    
    %% 6) 통계 출력
    idxSeed = 2:size(para_hats,1);     % seed 1~10 행만 선택 (seed 0 제외)
    
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
    
    %% 7) 테이블로 정리
    rowNames = {'Mean','Min','Max','STD'};
    varNames = {'R0','R1','R2','tau1','tau2','RMSE'};
    
    % 통계 테이블 생성
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

    %% 8) [추가됨] 엑셀 파일로 저장
    % 파일명 생성 (예: udds_0725..._Summary.xlsx)
    excel_filename = sprintf('%s_Summary.xlsx', base_name);
    full_excel_path = fullfile(excel_save_dir, excel_filename);
    
    % RowNames(Mean, Min 등)를 포함하여 저장
    writetable(T, full_excel_path, 'WriteRowNames', true);
    fprintf('   >> 엑셀 저장 완료: %s\n\n', full_excel_path);

    %% 9) g(τ) 플롯 등 (생략 가능, 필요시 유지)
    
end

%% ------------------------------------------------------------------
%  RMSE (2-RC) 함수
% ------------------------------------------------------------------
function cost = RMSE_2RC(data, para, t, I, OCV)
    model = RC_model_2(para, t, I, OCV);
    cost  = sqrt(mean((data - model).^2));
end