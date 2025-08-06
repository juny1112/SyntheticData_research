% ======================================================================
%  2-RC 가상 셀 → Markov-Noise 합성 → MultiStart 피팅 전 과정
%  (OCV = 0 V 가정, 마르코프 노이즈 과정 실시간 시각화 포함)
% ======================================================================
clear; clc; close all;

%% (A) 2-RC ‘참’ 파라미터   [R0 R1 R2 tau1 tau2]
X_true = [0.001 0.0005 0.0005 6 60];

%% (B) 전류 프로파일 목록  (Pulse 예시·엑셀 주행패턴 혼용 가능)
% Pulse
dt = 1; t_start = 0;  t_end = 180;
pulse_start = 0; pulse_end = 180;

pulse.t = (t_start:dt:t_end)';       
pulse.I = zeros(size(pulse.t));
pulse.I(pulse.t>=pulse_start & pulse.t<=pulse_end) = 1;    

% driving_files = {pulse};                 % Pulse만
% driving_files = driving_paths;           % 엑셀만
driving_files = {pulse};                   % ← 원하는 대로 선택

%% (C) Markov-Noise 파라미터
epsilon_percent_span = 5;    % ±1 %
initial_state        = 51;
sigma                = 5;
nSeeds               = 10;   % seed1 … seed10  (+ Non_noise)

%% (D) MultiStart / fmincon 설정
ms        = MultiStart("UseParallel",true,"Display","off");
startPts  = RandomStartPointSet('NumStartPoints',20);
opt = optimset('display','off','MaxIter',1e3,'MaxFunEvals',1e4,...
               'TolFun',1e-14,'TolX',1e-15);

para0 = [0.0012 0.0006 0.0004 5 70];
lb    = [0 0 0 0.001 0.001];
ub    = para0*10;
A_lin = [0 0 0 1 -1];   b_lin = 0;   % tau1 < tau2

%% (E) 메인 루프
for idx = 1:numel(driving_files)

    % 1) 프로파일 로드 --------------------------------------------------
    item = driving_files{idx};
    if isstruct(item)        % Pulse 구조체
        t_vec = item.t(:);   I_vec = item.I(:);
        base  = sprintf('load%d',idx);
    else                      % 엑셀
        T   = readtable(item);
        t_vec = T{:,1};      I_vec = T{:,2};
        [~,base,~] = fileparts(item);
    end

    OCV = zeros(size(t_vec));                        % OCV = 0 V 가정
    V_est = RC_model_2(X_true,t_vec,I_vec,OCV);      % 참 전압

    %% (F) Markov-Noise 생성 & 시각화 ---------------------------------
    noisedata.Non_noise = V_est;
    for seed = 1:nSeeds
        rng(seed);
        [noisy,eps_k,~,~,~,~] = MarkovNoise_idx( ...
            V_est, epsilon_percent_span, initial_state, sigma);
        noisedata.(sprintf('V_SD%d',seed)) = noisy;

        % ── 확인 Plot ──────────────────────────────────────────────
        figure('Name',sprintf('%s – Seed %d Noise',base,seed),...
               'NumberTitle','off','Position',[100 100 900 700]);

        plot(t_vec, V_est, 'Color',[0 0.45 0.74],'LineWidth',1.5); hold on   % 파랑
        plot(t_vec, noisy , '--','Color',[0.85 0.33 0.10],'LineWidth',1.5); % 주황 점선
        xlabel('Time (sec)'); ylabel('Voltage [V]');
        legend({'V_{est}','V_{SD}'},'Location','best');
        title('Markov Noise Voltage data');

        % subplot(2,1,1);
        %     plot(t_vec,V_est,'b','LineWidth',1.5); hold on
        %     plot(t_vec,noisy ,'r--','LineWidth',1.5);
        %     xlabel('Time (s)'); ylabel('Voltage (V)');
        %     legend('V_{est}','V_{SD}','Location','best');
        %     title(sprintf('Markov Noise Voltage  (seed %d)',seed)); grid on;
        % 
        % subplot(2,1,2);
        %     plot(t_vec,eps_k,'k','LineWidth',1.2);
        %     xlabel('Time (s)'); ylabel('\epsilon_k (%)');
        %     title('Markov Noise \epsilon_k'); grid on;
        % 
        % figure('Name',sprintf('%s – Seed %d ΔV',base,seed),...
        %        'NumberTitle','off','Position',[1050 300 450 300]);
        %     plot(t_vec,noisy-V_est,'m','LineWidth',1.5);
        %     xlabel('Time (s)'); ylabel('ΔV  (V_{SD}-V_{est})');
        %     title('Noise Difference'); grid on;
    end
    save(['noise_' base '_seed10.mat'],'-struct','noisedata');

    %% (G) MultiStart 피팅 ---------------------------------------------
    seednames = [{'Non_noise'}, arrayfun(@(s)sprintf('V_SD%d',s),1:nSeeds,'un',0)];
    nSeries   = numel(seednames);
    para_hats = zeros(nSeries,numel(para0)+2);   % + exitflag iter
    RMSE_list = zeros(nSeries,1);

    figure('Name',[base ' – Seed별 피팅'],'NumberTitle','off');

    for s = 1:nSeries
        V_SD = noisedata.(seednames{s});
        problem = createOptimProblem('fmincon',...
            'objective',@(p) RMSE_2RC(V_SD,p,t_vec,I_vec,OCV),...
            'x0',para0,'lb',lb,'ub',ub,...
            'Aineq',A_lin,'bineq',b_lin,'options',opt);

        [bestP,bestF,eflag,~,sol] = run(ms,problem,startPts);
        iter = NaN; if ~isempty(sol)
            iter = sol(find([sol.Fval]==bestF,1)).Output.iterations;
        end

        para_hats(s,:) = [bestP eflag iter];
        RMSE_list(s)   = bestF;

        % 결과 Plot -------------------------------------------------
        V_fit = RC_model_2(bestP,t_vec,I_vec,OCV);
        V_0   = RC_model_2(para0,t_vec,I_vec,OCV);

        subplot(3,4,s);
            plot(t_vec,V_SD,'k','LineWidth',1.2); hold on
            plot(t_vec,V_est,'g','LineWidth',1.2);
            plot(t_vec,V_fit,'r','LineWidth',1.2);
            plot(t_vec,V_0 ,'b--','LineWidth',1);
            title(seednames{s}); grid on;
    end
    sgtitle([base ' – 2RC Fitting']);

    %% (H) 파라미터 통계 ----------------------------------------------
    mean_para = mean(para_hats(2:end,1:5),1);
    min_para  = min (para_hats(2:end,1:5),[],1);
    max_para  = max (para_hats(2:end,1:5),[],1);
    std_para  = std (para_hats(2:end,1:5),0,1);

    mean_RMSE = mean(RMSE_list(2:end));
    fprintf('\n>> %s 파라미터 요약 (Seed1~10)\n',base);
    fprintf('   Mean: R0=%.6g R1=%.6g R2=%.6g tau1=%.4g tau2=%.4g  RMSE=%.6g\n',...
            mean_para,mean_RMSE);
    fprintf('   Min : R0=%.6g R1=%.6g R2=%.6g tau1=%.4g tau2=%.4g  RMSE=%.6g\n',...
            min_para ,min(RMSE_list(2:end)));
    fprintf('   Max : R0=%.6g R1=%.6g R2=%.6g tau1=%.4g tau2=%.4g  RMSE=%.6g\n',...
            max_para ,max(RMSE_list(2:end)));
    fprintf('   Std : R0=%.6g R1=%.6g R2=%.6g tau1=%.4g tau2=%.4g  RMSE=%.6g\n',...
            std_para ,std(RMSE_list(2:end)));
end

% ----------------------------------------------------------------------
%  보조 함수
% ----------------------------------------------------------------------
function V = RC_model_2(X,t,I,OCV)
    R0=X(1); R1=X(2); R2=X(3); tau1=X(4); tau2=X(5);
    dt=[1;diff(t)]; N=numel(t); V=zeros(N,1); Vc1=0; Vc2=0;
    for k=1:N
        IR0 = R0*I(k);
        a1=exp(-dt(k)/tau1); a2=exp(-dt(k)/tau2);
        if k>1
            Vc1 = Vc1*a1 + R1*(1-a1)*I(k-1);
            Vc2 = Vc2*a2 + R2*(1-a2)*I(k-1);
        end
        V(k)=OCV(k)+IR0+Vc1+Vc2;
    end
end
function cost = RMSE_2RC(data,p,t,I,OCV)
    cost = sqrt(mean( (data - RC_model_2(p,t,I,OCV)).^2 ));
end
