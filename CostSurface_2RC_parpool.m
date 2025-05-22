%% ========================================================================
%  2-RC 모델 τ1–τ2 Cost-Surface (seed 병렬) + 결과 저장
% ========================================================================
clear; clc; close all;

% ── 0. 병렬 풀 열기 (없으면 생성) ─────────────────────────────────────────
if isempty(gcp('nocreate'))
    parpool;                                     % 로컬 코어 전부 사용
end

% ── 1. 드라이빙 데이터 파일 목록 ─────────────────────────────────────────
driving_files = {
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\udds_unit_time_scaled_current.xlsx',
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\us06_unit_time_scaled_current.xlsx',
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_CITY1_time_scaled_current.xlsx',
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_HW1_time_scaled_current.xlsx'
};

for fileIdx = 1:numel(driving_files)
    filename  = driving_files{fileIdx};
    data      = readtable(filename);

    t_vec = data.Var1;      % [sec]
    I_vec = data.Var2;      % [A]  

    % (기준 파라미터) 2-RC 전압
    X0    = [0.001 0.0005 0.0005 6 60];
    V_est = RC_model_2(X0,t_vec,I_vec);

    % ── 2. Markov Noise (10 seeds) ───────────────────────────────────────
    eps_span   = 5;
    init_state = 51;
    sigma      = 0.005;
    numSeeds   = 10;

    V_SD_cell = cell(numSeeds,1);               % parfor 전달용 Cell
    for s = 1:numSeeds
        rng(s);
        [noisy,~,~,~,~,~] = MarkovNoise(V_est,eps_span,init_state,sigma);
        V_SD_cell{s} = noisy;
    end

    % ── 3. 결과 저장 배열/셀 초기화 ──────────────────────────────────────
    best_tau1 = zeros(numSeeds,1);
    best_tau2 = zeros(numSeeds,1);
    best_rmse = zeros(numSeeds,1);
    cost_cell = cell(numSeeds,1);               % Cost-surface 저장

    % 공통 그리드
    tau1_vec = 10.^(linspace(-1, 1.1, 101));     % 0.1 ~ ≈12.6 s (31 개)
    tau2_vec = 10.^(linspace( 1, 2.1, 81));     % 10  ~ ≈126  s (81 개)

    opts = optimset('Display','off','MaxIter',1e3,...
                    'MaxFunEvals',1e4,'TolFun',1e-14,'TolX',1e-15);

    % ── 4. ★ Seed 병렬 계산 (parfor) ────────────────────────────────────
    parfor i = 1:numSeeds
        V_SD = V_SD_cell{i};
        cost_surface = zeros(numel(tau2_vec), numel(tau1_vec));

        for ii = 1:numel(tau1_vec)
            for jj = 1:numel(tau2_vec)
                p0 = [0.0012 0.0006 0.0004 tau1_vec(ii) tau2_vec(jj)];
                lb = [0 0 0 tau1_vec(ii) tau2_vec(jj)];
                ub = [p0(1:3)*10 tau1_vec(ii) tau2_vec(jj)];

                [~,fval] = fmincon(@(p)RMSE_2RC(V_SD,p,t_vec,I_vec), ...
                                   p0,[],[],[],[],lb,ub,[],opts);
                cost_surface(jj,ii) = fval;
            end
        end

        % 최적값
        [minCost,idxLin] = min(cost_surface(:));
        [r,c]            = ind2sub(size(cost_surface),idxLin);
        best_tau1(i)     = tau1_vec(c);
        best_tau2(i)     = tau2_vec(r);
        best_rmse(i)     = minCost;

        cost_cell{i} = cost_surface;    % 플롯용 저장
    end   % parfor end

    % ── 5. 결과 테이블 & MAT 저장 ──────────────────────────────────────
    seedID   = (1:numSeeds).';
    resultTB = table(seedID,best_tau1,best_tau2,best_rmse,...
                     'VariableNames',{'Seed','Tau1','Tau2','RMSE'});
    disp(resultTB);

    baseName = sprintf('load%d',fileIdx);
    save(['bestTauTable_' baseName '.mat'],'resultTB');

    % ── 6. Cost-surface 그림 (병렬 끝난 뒤 순차) ────────────────────────
    [T1,T2] = meshgrid(tau1_vec,tau2_vec);
    for i = 1:numSeeds
        fig = figure('Name',sprintf('Seed %d Cost Surface',i),'NumberTitle','off');
        % surf: 범례 제외
        surf(T1,T2,cost_cell{i},'EdgeColor','none','FaceColor','interp',...
             'HandleVisibility','off');
        view(3); shading interp; colorbar;
        xlabel('\tau_1 [s]'); ylabel('\tau_2 [s]'); zlabel('RMSE'); hold on;
        % 빨간★
        hStar = plot3(best_tau1(i),best_tau2(i),best_rmse(i),'r*',...
                      'MarkerSize',12,'LineWidth',2);
        title(sprintf('Cost Surface – Seed %d',i));
        legend(hStar, sprintf('\\tau_1^*=%.3f  \\tau_2^*=%.3f', ...
                              best_tau1(i),best_tau2(i)), 'Location','best');
    end
end

%% ─────────────────────────────────────────────────────────────────────────
% 2-RC 모델 RMSE 함수
% ─────────────────────────────────────────────────────────────────────────
function cost = RMSE_2RC(data,p,t,I)
    model = RC_model_2(p,t,I);
    cost  = sqrt( mean( (data-model).^2 ) );
end
