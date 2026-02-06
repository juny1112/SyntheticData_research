%% ========================================================================
%  2-RC τ1–τ2 Cost-Surface (seed 병렬) + MultiStart(5변수) + 결과 저장
% ========================================================================
clear; clc; close all;

% ── 병렬 풀 ─────────────────────────────────────────────────────────────
if isempty(gcp('nocreate'))
    parpool;
end

% ── MultiStart 객체 (★ MS는 parfor 밖에서 실행) ─────────────────────────
ms = MultiStart('UseParallel', true, 'Display', 'iter');
startPts = RandomStartPointSet('NumStartPoints', 20);

driving_files = {
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_CITY1_time_scaled_current.xlsx'
};

for fileIdx = 1:numel(driving_files)
    filename = driving_files{fileIdx};
    data = readtable(filename);

    t_vec = data.Var1(:);
    I_vec = data.Var2(:);
    N = numel(t_vec);
    OCV = zeros(N,1);

    % 기준 전압
    X0    = [0.001 0.0005 0.001 6 60];
    V_est = RC_model_2(X0, t_vec, I_vec, OCV);

    % ── Markov Noise (10 seeds) ─────────────────────────────────────────
    epsilon_percent_span = 1;
    initial_state = 51;
    sigma = 5;
    numSeeds = 1;

    V_SD_cell = cell(numSeeds,1);
    for s = 1:numSeeds
        rng(s);
        [noisy,~,~,~,~,~] = MarkovNoise_idx(V_est, epsilon_percent_span, initial_state, sigma);
        V_SD_cell{s} = noisy(:);   % ★ 필수
    end

    % ── Cost-surface grid ───────────────────────────────────────────────
    tau1_vec = 10.^(linspace(-1, 1.1, 101));   % 0.1 ~ 12.6 s
    tau2_vec = 10.^(linspace( 1, 2, 81));    % 10  ~ 50  s
    [T1,T2]  = meshgrid(tau1_vec, tau2_vec);

    % fmincon 옵션 (cost-surface용: R만 fitting)
    optsSurf = optimoptions('fmincon','Display','off',...
        'MaxIterations',1e3,'MaxFunctionEvaluations',1e4,...
        'OptimalityTolerance',1e-14,'StepTolerance',1e-15);

    % 결과 저장
    best_tau1 = zeros(numSeeds,1);
    best_tau2 = zeros(numSeeds,1);
    best_rmse = zeros(numSeeds,1);
    cost_cell = cell(numSeeds,1);

    % ── 1) ★ seed별 cost-surface는 parfor로 계산 ─────────────────────────
    parfor i = 1:numSeeds
        V_SD = V_SD_cell{i};
        cost_surface = zeros(numel(tau2_vec), numel(tau1_vec));

        for ii = 1:numel(tau1_vec)
            for jj = 1:numel(tau2_vec)
                % tau 고정, R0 R1 R2만 최적화
                p0 = [0.0012 0.0006 0.0004 tau1_vec(ii) tau2_vec(jj)];
                lb = [0      0      0      tau1_vec(ii) tau2_vec(jj)];
                ub = [p0(1:3)*10     tau1_vec(ii) tau2_vec(jj)];

                [~,fval] = fmincon(@(p) RMSE_2RC(V_SD,p,t_vec,I_vec,OCV), ...
                                   p0,[],[],[],[],lb,ub,[],optsSurf);
                cost_surface(jj,ii) = fval;
            end
        end

        % surface 최적점
        [minCost,idxLin] = min(cost_surface(:));
        [r,c] = ind2sub(size(cost_surface), idxLin);

        best_tau1(i) = tau1_vec(c);
        best_tau2(i) = tau2_vec(r);
        best_rmse(i) = minCost;

        cost_cell{i} = cost_surface;
    end

    % ── 2) ★ parfor 끝난 뒤: seed별 MultiStart(5변수) 실행 ───────────────
    bestP_MS   = zeros(numSeeds,5);
    bestF_MS   = zeros(numSeeds,1);
    exit_MS    = zeros(numSeeds,1);
    iter_MS    = zeros(numSeeds,1);

    for i = 1:numSeeds
        V_SD = V_SD_cell{i};

        % cost-surface figure
        figure('Name',sprintf('Seed %d Cost Surface',i),'NumberTitle','off');
        surf(T1,T2,cost_cell{i},'EdgeColor','none','FaceColor','interp','HandleVisibility','off');
        view(3); shading interp; colorbar; hold on;
        xlabel('\tau_1 [s]'); ylabel('\tau_2 [s]'); zlabel('RMSE');
        title(sprintf('Cost Surface – Seed %d', i));

        % 빨간★(surface best)
        hStar = plot3(best_tau1(i),best_tau2(i),best_rmse(i),'r*',...
            'MarkerSize',12,'LineWidth',2);

        % ---- MultiStart 설정: 5변수 전체 최적화 ----
        % x = [R0 R1 R2 tau1 tau2]
        x0 = [0.0012 0.0006 0.0004 best_tau1(i) best_tau2(i)]; % surface best 근처로 시작
        lb = [0 0 0  tau1_vec(1) tau2_vec(1)];
        ub = [0.02 0.02 0.02  tau1_vec(end) tau2_vec(end)];    % R upper는 상황 맞게 조절

        optsMS = optimoptions('fmincon','Display','iter',...
            'MaxIterations',1e3,'MaxFunctionEvaluations',2e4,...
            'OptimalityTolerance',1e-14,'StepTolerance',1e-15,...
            'OutputFcn', @(x,ov,st) plotIterTau(x,ov,st)); % ★ 경로 plot

        problem = createOptimProblem('fmincon',...
            'objective', @(x) RMSE_2RC(V_SD,x,t_vec,I_vec,OCV),...
            'x0', x0,'lb',lb,'ub',ub,'options',optsMS);

        [xBest,fBest,exitflag,~,sols] = run(ms, problem, startPts);

        bestP_MS(i,:) = xBest;
        bestF_MS(i)   = fBest;
        exit_MS(i)    = exitflag;

        % iterations 기록
        idx = find([sols.Fval] == fBest, 1);
        if ~isempty(idx)
            iter_MS(i) = sols(idx).Output.iterations;
        else
            iter_MS(i) = NaN;
        end

        % 초록○(MultiStart best)
        hMS = plot3(xBest(4), xBest(5), fBest, 'go','MarkerSize',10,'LineWidth',2);

        legend([hStar,hMS], ...
            {sprintf('Surface Opt: \\tau1=%.3f \\tau2=%.3f',best_tau1(i),best_tau2(i)), ...
             sprintf('MultiStart Opt: \\tau1=%.3f \\tau2=%.3f',xBest(4),xBest(5))}, ...
            'Location','best','AutoUpdate','off');
    end

    % ── 3) 결과 테이블 저장 ─────────────────────────────────────────────
    resultTB = table((1:numSeeds).', best_tau1, best_tau2, best_rmse, ...
                     bestP_MS(:,1), bestP_MS(:,2), bestP_MS(:,3), bestP_MS(:,4), bestP_MS(:,5), ...
                     bestF_MS, exit_MS, iter_MS, ...
        'VariableNames', {'Seed','Tau1_surf','Tau2_surf','RMSE_surf',...
                          'R0_MS','R1_MS','R2_MS','Tau1_MS','Tau2_MS',...
                          'RMSE_MS','Exit_MS','Iter_MS'});
    disp(resultTB);

    save(sprintf('bestTauTable_MS_load%d.mat',fileIdx), 'resultTB');
end

%% ─────────────────────────────────────────────────────────────────────────
function cost = RMSE_2RC(data,p,t,I,OCV)
    data = data(:);
    model = RC_model_2(p, t(:), I(:), OCV(:));
    model = model(:);

    % 안전장치
    assert(numel(data)==numel(model), 'RMSE_2RC: length mismatch (data=%d, model=%d)', ...
        numel(data), numel(model));

    cost = sqrt(mean((data - model).^2));
end

%% ── iteration path: (tau1, tau2, fval)만 surface 위에 찍기 ──────────────
function stop = plotIterTau(x, optimValues, state)
    stop = false;
    persistent hLine
    switch state
        case 'init'
            hold on;
            hLine = plot3(nan,nan,nan,'-k','LineWidth',1.5,'HandleVisibility','off');
            plot3(x(4),x(5),optimValues.fval,'yo','MarkerSize',6,'LineWidth',1.0,'HandleVisibility','off');
        case 'iter'
            set(hLine, 'XData', [get(hLine,'XData'), x(4)], ...
                       'YData', [get(hLine,'YData'), x(5)], ...
                       'ZData', [get(hLine,'ZData'), optimValues.fval]);
            drawnow;
        case 'done'
            plot3(x(4),x(5),optimValues.fval,'gs','MarkerSize',7,'LineWidth',1.0,'HandleVisibility','off');
            clear hLine;
    end
end
