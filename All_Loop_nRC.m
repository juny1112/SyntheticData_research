% ======================================================================
%  pulse + driving Loop  (n-RC 모델 + bimodal 참값 + MS + τ-inequality)
% ======================================================================
clear; clc; close all;

%% ------------------------------------------------------------------
% (A) bimodal 로 n-RC 참 파라미터 만들기
% ------------------------------------------------------------------
n     = 40;                     % RC 쌍 수
n_mode1  = 10;                  % 모드1에 10개 할당
tau_peak = [6 60];              % 두 피크 중심 [s]
sigma    = [10 20];             % 표준편차 [s]
tau_rng  = [0.1 150];           % τ 범위 [s]
R0_true       = 0.001;                      

[X, R_i, tau_i] = bimodal_Rtau(n, n_mode1, tau_peak, sigma, tau_rng, R0_true);

% X_true = X;   % (1+2n)×1  참값 벡터
n_cut = 27; % 원하는 n
X_true = X([1 2:n_cut+1 n+2:n+n_cut+1]); % bimodal에서 R 너무 작은 값 제외   
R_i      = R_i(1:n_cut);          % 저항 배열 축소
tau_i    = tau_i(1:n_cut);        % τ 배열 축소
n = n_cut; % 전압 생성시 n_cut 개의 nRC model 사용하기 위함

%% ------------------------------------------------------------------
% (B) 전류 프로파일 목록 (struct 또는 .xlsx 혼용 가능)
% ------------------------------------------------------------------
% Pulse
dt = 0.1;   t_end = 180;
pulse.t = (0:dt:t_end)';         
pulse.I = zeros(size(pulse.t));
pulse.I(pulse.t>=10 & pulse.t<=20) = 1;     % 10~20 s
driving_files = { pulse };        

% % Driving data
% driving_files = {
%      'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\udds_unit_time_scaled_current.xlsx',
%      'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\us06_unit_time_scaled_current.xlsx',
%      'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_CITY1_time_scaled_current.xlsx',
%      'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_HW1_time_scaled_current.xlsx'
%  };

%% ------------------------------------------------------------------
% (C) 마르코프 노이즈 설정
% ------------------------------------------------------------------
epsilon_percent_span = 5;     % ±5 %
initial_state = 51;
sigma = 5;
nSeeds = 10;                  % seed 1~10  (+ seed 0 = 원본)

%% ------------------------------------------------------------------
% (D) MultiStart 설정
% ------------------------------------------------------------------
ms = MultiStart("UseParallel",true,"Display","off");
nStartPts = 20;
startPts  = RandomStartPointSet('NumStartPoints', nStartPts);
opt = optimset('display','off','MaxIter',1e3,'MaxFunEvals',1e5, ...
               'TolFun',1e-14,'TolX',1e-15); % 여기만 MaxFunEvals 조정

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
        base_name = erase(item,'.xlsx');
    end

    %% 2) 참 전압 V_est (n-RC)
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

    %% 4) Markov Non_noise + seed1~10 세트 -------------------------------
    noisedata            = struct;
    noisedata.Non_noise  = V_est;                  
    for seed = 1:nSeeds
        rng(seed);
        [noisy,~,~,~,~,~] = MarkovNoise_idx(V_est, ...
            epsilon_percent_span, initial_state, sigma);
        noisedata.(sprintf('V_SD%d',seed)) = noisy;
    end
    save(['noise_' base_name '_seed10.mat'],'-struct','noisedata');

    %% 5) Fitting 설정 (길이 = 1+2n)
    para0 = [R0_true*0.9; R_i*1.2; tau_i*1.1]; % ~20% 에러
    lb    = zeros(size(para0));
    lb(n+2:end) = 0.001;  % tau lb
    ub    = para0*10;

    % τ1<τ2<…<τn (inequality 조건)
    A_lin = zeros(n-1, numel(para0));
    for k = 1:n-1
        A_lin(k,1+n+k)   =  1;    % τ_k
        A_lin(k,1+n+k+1) = -1;    % -τ_{k+1}
    end
    b_lin = zeros(n-1,1);

    seednames    = [{'Non_noise'}, arrayfun(@(s)sprintf('V_SD%d',s), ...
        1:nSeeds,'UniformOutput',false)];
    displaynames = [{'Non noise'}, arrayfun(@(s)sprintf('seed%d',s), ...
        1:nSeeds,'UniformOutput',false)];

    para_hats = zeros(numel(seednames), numel(para0)+2); % +exitflag +iter
    RMSE_list = zeros(numel(seednames),1);

    figure('Name',sprintf('%s - All Seed Fitting Results',base_name), ...
           'NumberTitle','off');

    seednames = fieldnames(noisedata);   
    for s = 1:numel(seednames)
        V_SD = noisedata.(seednames{s});

        problem = createOptimProblem('fmincon', ...
            'objective', @(p)RMSE_nRC(V_SD,p,t_vec,I_vec,n), ...
            'x0', para0, 'lb', lb, 'ub', ub, ...
            'Aineq', A_lin, 'bineq', b_lin, 'options', opt);

        [bestP,bestFval,eflg,~,sltns] = run(ms,problem,startPts);
        % ----- iterations 안전하게 뽑기 -----
        if isempty(sltns)              % 아무 로컬 솔루션도 없으면…
            iter = NaN;                % 반복 횟수는 기록 불가 → NaN
        else
            idx  = find([sltns.Fval] == bestFval, 1);
            iter = sltns(idx).Output.iterations;
        end

        para_hats(s,:) = [bestP.' eflg iter];
        RMSE_list(s)   = bestFval;

        V_0   = RC_model_n(para0,t_vec,I_vec,n);
        V_hat = RC_model_n(bestP,t_vec,I_vec,n);

        subplot(3,4,s);   % 3×4 = 12칸 (11개 사용)
        plot(t_vec,V_SD,'-k','LineWidth',1.5); hold on
        plot(t_vec,V_est,'-g','LineWidth',1.5);
        plot(t_vec,V_hat,'-r','LineWidth',1.5);
        plot(t_vec,V_0 ,'--b','LineWidth',1.5);
        legend({'Synthetic','True','Fitted','Initial'},'Location','northeast');
        xlabel('Time (sec)'); ylabel('Voltage (V)');
        title(displaynames{s}); grid on;
    end
    sgtitle(sprintf('%s - nRC Fitting for All Seeds',base_name));

    %% 6) 통계 출력 (원본 형식 유지)
    mean_para = mean(para_hats,1);   min_para = min(para_hats,[],1);
    max_para  = max(para_hats,[],1); std_para = std(para_hats,0,1);
    mean_RMSE = mean(RMSE_list);     min_RMSE = min(RMSE_list);
    max_RMSE  = max(RMSE_list);      std_RMSE = std(RMSE_list);

    fprintf('>> [%s] 파라미터 요약 통계 :\n',base_name);
    fprintf('   [Mean] R0=%.8f ... tau%d=%.8f RMSE=%.8f\n',...
        mean_para(1), n, mean_para(1+2*n), mean_RMSE);
    fprintf('   [Min]  R0=%.8f ... tau%d=%.8f RMSE=%.8f\n',...
        min_para(1),  n, min_para(1+2*n),  min_RMSE);
    fprintf('   [Max]  R0=%.8f ... tau%d=%.8f RMSE=%.8f\n',...
        max_para(1),  n, max_para(1+2*n),  max_RMSE);
    fprintf('   [STD]  R0=%.8f ... tau%d=%.8f RMSE=%.8f\n\n',...
        std_para(1),  n, std_para(1+2*n),  std_RMSE);

    all_para_hats.(base_name) = para_hats;
    all_rmse.(base_name)      = RMSE_list;

end

%% ===============================================================
%  τ–R 분포 플롯 (모든 드라이빙 파일 순회 예시)
% ================================================================
fn = fieldnames(all_para_hats);      % {'load1', 'udds', ...}

for k = 1:numel(fn)
    base_name = fn{k};
    A         = all_para_hats.(base_name);   % 이번 파일의 para_hats
    n_plot    = n;                           % 현재 RC 개수 (=27)

    idx_R   = 2         : n_plot+1;
    idx_tau = n_plot+2  : 2*n_plot+1;

    % ── Seed 0 ───────────────────────────────────────
    figure('Name',sprintf('[%s] Seed 0 : R–τ',base_name),'NumberTitle','off');
    stem(A(1,idx_tau), A(1,idx_R)*1e3,'filled','MarkerSize',4,'LineWidth',1.2);
    xlabel('\tau (s)'); ylabel('R (m\Omega)'); grid on;
    title(sprintf('[%s] Seed 0 : R–\\tau',base_name));

    % ── 모든 Seed ────────────────────────────────────
    figure('Name',sprintf('[%s] All seeds : R–τ',base_name),'NumberTitle','off');
    hold on; clr = lines(size(A,1));
    for s = 1:size(A,1)
        plot(A(s,idx_tau), A(s,idx_R)*1e3, '-o', ...
             'Color',clr(s,:), 'MarkerSize',3,'LineWidth',1);
    end
    xlabel('\tau (s)'); ylabel('R (m\Omega)'); grid on; box on;
    title(sprintf('[%s] All seeds : R–\\tau',base_name));
    legend({'Non noise','seed1','seed2','seed3','seed4','seed5',...
            'seed6','seed7','seed8','seed9','seed10'}, ...
            'Location','eastoutside');
end


%% ------------------------------------------------------------------
%  RMSE (n-RC)
% ------------------------------------------------------------------
function cost = RMSE_nRC(data,para,t,I,n)
    model = RC_model_n(para,t,I,n);
    cost  = sqrt(mean((data - model).^2));
end
