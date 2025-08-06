% 수정해야될 것: seed 별로 결과 나오게 + 최적해도 나오게 + 범례에 다 표시되게 

% ======================================================================
%  Pulse + Driving Loop
%  (40-RC 모델 참값 생성 → 2-RC 피팅 + MultiStart + τ-inequality
%   + τ₁–τ₂ Cost-Surface 계산)
% ======================================================================
clear; clc; close all;

%% ------------------------------------------------------------------ (A)
%  40-RC 바이모달 참 파라미터 X_true 생성
% ----------------------------------------------------------------------
n        = 40;                % RC 쌍 수
n_mode1  = 10;                % 모드1 RC 개수
tau_peak = [6 60];            % 두 피크 중심 [s]
sigma    = [10 20];           % 표준편차 [s]
tau_rng  = [0.1 150];         % τ 범위 [s]
R0_true  = 0.001;             % 직렬저항(참값)

[X_true, R_i, tau_i] = bimodal_Rtau(n, n_mode1, tau_peak, ...
                                    sigma, tau_rng, R0_true);

%% ------------------------------------------------------------------ (B)
%  드라이빙 전류 파일/프로필 목록
% ----------------------------------------------------------------------
% Pulse
dt    = 0.1;   t_end = 180;
pulse.t = (0:dt:t_end)';         
pulse.I = zeros(size(pulse.t));
pulse.I(pulse.t>=10 & pulse.t<=20) = 1;     % 10~20 s
driving_files = { pulse };        

% % driving
% driving_files = {
%     'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\udds_unit_time_scaled_current.xlsx'
%     'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\us06_unit_time_scaled_current.xlsx'
%     'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_CITY1_time_scaled_current.xlsx'
%     'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_HW1_time_scaled_current.xlsx'
% };

%% ------------------------------------------------------------------ (C)
%  마르코프 노이즈 설정
% ----------------------------------------------------------------------
epsilon_percent_span = 5;      % ±5 %
initial_state        = 51;
sigma_noise          = 5;
nSeeds               = 10;     % seed 1~10 + Non_noise

%% ------------------------------------------------------------------ (D)
%  MultiStart 옵션 (2-RC 피팅용)
% ----------------------------------------------------------------------
ms         = MultiStart("UseParallel",true,"Display","off");
nStartPts  = 20;
startPts   = RandomStartPointSet('NumStartPoints', nStartPts);
opt_fmin   = optimset('display','off','MaxIter',1e3,'MaxFunEvals',1e4, ...
                      'TolFun',1e-14,'TolX',1e-15);

%% ------------------------------------------------------------------ (E)
%  메인 루프 (드라이빙 파일별)
% ----------------------------------------------------------------------
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

    % ------------------------------------------------- 3) 시드별 노이즈 전압
    noisedata            = struct;
    noisedata.Non_noise  = V_est;
    for seed = 1:nSeeds
        rng(seed);
        [noisy,~,~,~,~,~] = MarkovNoise_idx(V_est, ...
            epsilon_percent_span, initial_state, sigma_noise);
        noisedata.(sprintf('V_SD%d',seed)) = noisy;
    end
    save(['noise_' base_name '_seed10.mat'],'-struct','noisedata');

    % ------------------------------------------------- 4) 2-RC 피팅 설정
    % 파라미터: [R0 R1 R2 τ1 τ2]
    para0 = [0.0012 0.0008 0.0012 5 70];
    lb    = [0 0 0 0.001 0.001];
    ub    = para0 * 10;
    A_lin = [0 0 0 1 -1];   % τ1 < τ2
    b_lin = 0;

    seedlist   = [{'Non_noise'}, arrayfun(@(s)sprintf('V_SD%d',s),1:nSeeds,'UniformOutput',false)];
    dispnames  = [{'Non noise'}, arrayfun(@(s)sprintf('seed%d',s),1:nSeeds,'UniformOutput',false)];

    % % ------------------------------------------------- 5) τ1–τ2 Cost-Surface
    fprintf('\n>> [%s]  τ1–τ2 cost-surface 계산 중…\n', base_name);

    tau1_vec = logspace(-1, 1.1, 41);  
    tau2_vec = logspace( 1, 2, 61);  
    cost_surface = nan(numel(tau2_vec), numel(tau1_vec));

    opt_R = optimset('display','off','MaxIter',2e3,'MaxFunEvals',5e4,...
                     'TolFun',1e-12,'TolX',1e-12);
    R_init = [0.001 0.0008 0.0012];

    parfor ii = 1:numel(tau1_vec)
        tau1 = tau1_vec(ii);
        row_cost = nan(numel(tau2_vec), 1);

        for jj = 1:numel(tau2_vec)
            tau2 = tau2_vec(jj);
            p0 = [R_init tau1 tau2];
            lbR = [0 0 0 tau1 tau2];
            ubR = [p0(1:3)*10 tau1 tau2];
            [~,fval] = fmincon(@(p)RMSE_2RC(V_est,p,t_vec,I_vec), ...
                               p0,[],[],[],[],lbR,ubR,[],opt_R);
             row_cost(jj) = fval
            
        end
        cost_surface(:, ii) = row_cost;
    end

    figure('Name',[base_name ' – τ1–τ2 Cost Surface'],'NumberTitle','off');
    [T1,T2] = meshgrid(tau1_vec, tau2_vec);
    surf(T1, T2, cost_surface,'EdgeColor','none'); view(45,35);
    set(gca,'XScale','log','YScale','log'); colorbar; colormap parula;
    xlabel('\tau_1 (s)'); ylabel('\tau_2 (s)'); zlabel('RMSE');
    title(sprintf('%s  | 40-RC → 2-RC Cost Surface',base_name));

    [minC, idx] = min(cost_surface(:));
    [r,c]       = ind2sub(size(cost_surface),idx);
    best_tau1   = tau1_vec(c); best_tau2 = tau2_vec(r);
    hold on; plot3(best_tau1,best_tau2,minC,'r*','MarkerSize',12,'LineWidth',2);
    text(best_tau1,best_tau2,minC, ...
        sprintf('  \\tau_1^*=%.3g  \\tau_2^*=%.3g',best_tau1,best_tau2), ...
        'Color','w','FontWeight','bold');
end

%% ------------------------------------------------------------------ 함수 정의
function cost = RMSE_2RC(data,para,t,I)
    model = RC_model_2(para,t,I);
    cost  = sqrt(mean((data - model).^2));
end
