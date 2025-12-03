clear; clc; close all;

%% ------------------------------------------------------------------
% (A) 2-RC 참 파라미터 정의  [R0, R1, R2, tau1, tau2]
% ------------------------------------------------------------------
X_true = [0.001 0.0005 0.0005 6 60];

%% ------------------------------------------------------------------
% (B) 전류 프로파일 목록 (struct 또는 .xlsx 혼용 가능)
% ------------------------------------------------------------------

% Pulse
dt = 0.1; t_start = 0;  t_end = 10;
pulse_start = 0; pulse_end = 10;

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
driving_files = {pulse};
% 
% % Driving 켜기
% driving_files = driving_paths;
% 
% % 둘 다 켜기
% driving_files = [{pulse}; driving_paths];

%% ------------------------------------------------------------------
% (C) 마르코프 노이즈 설정
% ------------------------------------------------------------------
epsilon_percent_span = 1;   % ±5%
initial_state        = 51;
sigma                = 5;
nSeeds               = 10;  % seed1 … seed10  (+ Non_noise)

%% ------------------------------------------------------------------
% (D) MultiStart / fmincon 설정
% ------------------------------------------------------------------
ms        = MultiStart("UseParallel",true,"Display","off");
nStartPts = 20;
startPts  = RandomStartPointSet('NumStartPoints',nStartPts);

opt   = optimset('display','off','MaxIter',1e3,'MaxFunEvals',1e4, ...
                 'TolFun',1e-14,'TolX',1e-15);

para0 = [0.0012 0.0006 0.0004 5 70];      % 초기 guess
lb    = [0 0 0 0.001 0.001];
ub    = para0*10;

% inequality: tau1 < tau2  ≡  1*tau1 – 1*tau2 ≤ 0
% A * p <= b, where p = [R0 R1 R2 tau1 tau2]
A_lin = [0 0 0 1 -1];
b_lin = 0;

%% ------------------------------------------------------------------
% (E) 루프 시작
% ------------------------------------------------------------------
all_para_hats = struct;
all_rmse      = struct;

for fileIdx = 1:numel(driving_files)

    % ── 1) 데이터 로드 ─────────────────────────────────────────────
    item = driving_files{fileIdx};
    if isstruct(item)
        t_vec = item.t(:);     I_vec = item.I(:);
        base_name = sprintf('load%d',fileIdx);
    else
        tbl       = readtable(item);
        t_vec     = tbl{:,1};  I_vec = tbl{:,2};
        [~, base_name, ~] = fileparts(item);    
    end
    
    % OCV=0 가정 (과전압만 보기)
    OCV = zeros(size(t_vec));

    % ── 2) 원본 전압(V_est) 계산 ──────────────────────────────────
    V_est = RC_model_2(X_true,t_vec,I_vec,OCV);
    save([base_name '_data.mat'],'t_vec','I_vec','V_est');

    % ── 3) 입력 전류·전압 플롯 ───────────────────────────────────
    figure('Name',sprintf('%s - I & V Plots',base_name),'NumberTitle','off');
    subplot(3,1,1);
        plot(t_vec,I_vec,'r-','LineWidth',1.2);
        xlabel('Time (s)'); ylabel('Current (A)'); title('Current'); grid on;
    subplot(3,1,2);
        plot(t_vec,V_est,'b-','LineWidth',1.2);
        xlabel('Time (s)'); ylabel('Voltage (V)'); title('2-RC Voltage'); grid on;
    subplot(3,1,3);
        yyaxis left;  plot(t_vec,I_vec,'r-','LineWidth',1.2); ylabel('Current (A)');
        yyaxis right; plot(t_vec,V_est,'b-','LineWidth',1.2); ylabel('Voltage (V)');
        xlabel('Time (s)'); title('Current & Voltage'); grid on;

    % ── 4) Markov Noise + Non_noise 세트 ──────────────────────────
    noisedata           = struct;
    noisedata.Non_noise = V_est;             % 원본
    for seed = 1:nSeeds
        rng(seed);
        [noisy,~,~,~,~,~] = MarkovNoise_idx( ...
            V_est, epsilon_percent_span, initial_state, sigma);
        noisedata.(sprintf('V_SD%d',seed)) = noisy;
    end
    save(['noise_' base_name '_seed10.mat'],'-struct','noisedata');

    % ── 5) MultiStart Fitting ────────────────────────────────────
    seednames    = [{'Non_noise'}, arrayfun(@(s)sprintf('V_SD%d',s), ...
                                 1:nSeeds,'UniformOutput',false)];
    displaynames = [{'Non noise'}, arrayfun(@(s)sprintf('Seed %d',s), ...
                                 1:nSeeds,'UniformOutput',false)];

    nSeries   = numel(seednames);
    para_hats = zeros(nSeries, numel(para0)+2);   % + exitflag, iter
    RMSE_list = zeros(nSeries,1);

    figure('Name',sprintf('%s - All Seed Fitting Results',base_name), ...
           'NumberTitle','off');

    for s = 1:nSeries
        V_SD = noisedata.(seednames{s});

        problem = createOptimProblem('fmincon', ...
            'objective',@(p) RMSE_2RC(V_SD,p,t_vec,I_vec,OCV), ...
            'x0',para0,'lb',lb,'ub',ub, ...
            'Aineq',A_lin,'bineq',b_lin,'options',opt);

        [bestP,bestFval,eflg,~,sltns] = run(ms,problem,startPts);

        % 반복 횟수 추출
        iter = NaN;
        if ~isempty(sltns)
            idx  = find([sltns.Fval]==bestFval,1);
            iter = sltns(idx).Output.iterations;
        end

        para_hats(s,:) = [bestP eflg iter];
        RMSE_list(s)   = bestFval;

        % 결과 플롯 ------------------------------------------------
        V_hat = RC_model_2(bestP,t_vec,I_vec,OCV);
        V_0   = RC_model_2(para0 ,t_vec,I_vec,OCV);

        subplot(3,4,s);                      % 12칸(3×4)
        plot(t_vec,V_SD ,'-k','LineWidth',1.5); hold on
        plot(t_vec,V_est,'-g','LineWidth',1.5);
        plot(t_vec,V_hat,'-r','LineWidth',1.5);
        plot(t_vec,V_0 ,'--b','LineWidth',1.2);
        legend({'Synthetic','True','Fitted','Initial'},'Location','northeast');
        xlabel('Time (s)'); ylabel('Voltage (V)');
        title(displaynames{s}); grid on;
        % xlim([200 400]);
    end
    sgtitle(sprintf('%s - 2RC Fitting for All Seeds',base_name));

    % ── 6) 통계 출력 ─────────────────────────────────────────────
    % Summary statistics of parameters
    idxSeed = 2:size(para_hats,1);     % ← seed1~10 행만 선택

    mean_para = mean(para_hats(idxSeed,:),1);
    min_para  = min (para_hats(idxSeed,:),[],1);
    max_para  = max (para_hats(idxSeed,:),[],1);
    std_para  = std (para_hats(idxSeed,:),0,1);

    mean_RMSE = mean(RMSE_list(idxSeed));
    min_RMSE  = min (RMSE_list(idxSeed));
    max_RMSE  = max (RMSE_list(idxSeed));
    std_RMSE  = std (RMSE_list(idxSeed));

    fprintf('>> [%s] 파라미터 요약 통계 :\n', base_name);
    fprintf('   [Mean]   R0 = %.8f, R1 = %.8f, R2 = %.8f, tau1 = %.8f, tau2 = %.8f, RMSE = %.8f\n', mean_para(1), mean_para(2), mean_para(3), mean_para(4), mean_para(5), mean_RMSE);
    fprintf('   [Min]    R0 = %.8f, R1 = %.8f, R2 = %.8f, tau1 = %.8f, tau2 = %.8f, RMSE = %.8f\n', min_para(1),  min_para(2),  min_para(3), min_para(4), min_para(5), min_RMSE);
    fprintf('   [Max]    R0 = %.8f, R1 = %.8f, R2 = %.8f, tau1 = %.8f, tau2 = %.8f, RMSE = %.8f\n', max_para(1),  max_para(2),  max_para(3), max_para(4), max_para(5), max_RMSE);
    fprintf('   [STD]    R0 = %.8f, R1 = %.8f, R2 = %.8f, tau1 = %.8f, tau2 = %.8f, RMSE = %.8f\n\n', std_para(1), std_para(2), std_para(3), std_para(4), std_para(5), std_RMSE);

    all_para_hats.(base_name) = para_hats;
    all_rmse.(base_name)      = RMSE_list;
end

%% ------------------------------------------------------------------
%  (F) 보조 함수 : RMSE (2-RC)
% ------------------------------------------------------------------
function cost = RMSE_2RC(data,para,t,I,OCV)
model = RC_model_2(para,t,I,OCV);
cost  = sqrt(mean((data - model).^2));
end