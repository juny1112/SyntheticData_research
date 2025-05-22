clear; clc; close all;

if isempty(gcp('nocreate'))
    parpool;                 % 기본 설정(로컬 코어 수만큼) 워커 시작
end

% Driving data 목록 정의
driving_files = {
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\udds_unit_time_scaled_current.xlsx',
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\us06_unit_time_scaled_current.xlsx',
    'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_CITY1_time_scaled_current.xlsx',
    %'G:\공유 드라이브\Battery Software Lab\Protocols\Driving Load\55.6Ah_NE (분리열화실험)\BSL_HW1_time_scaled_current.xlsx'
};


for fileIdx = 1:length(driving_files)
    filename = driving_files{fileIdx};
    data = readtable(filename);

    t_vec = data.Var1;  % [sec]
    I_vec = data.Var2;  % [A]

    % 2RC 기준 파라미터
    X = [0.001 0.0005 0.0005 6 60];

    V_est = RC_model_2(X, t_vec, I_vec);

    % Markov Noise 설정
    eps_span = 5; init_state = 51; sigma = 0.005; numseeds = 10;
    noisedata = struct;
    for seed = 1:numseeds
        rng(seed);
        [noisy,~,~,~,~,~] = MarkovNoise(V_est, eps_span, init_state, sigma);
        noisedata.(sprintf('V_SD%d',seed)) = noisy;
    end

    % ────────────────────────────────────────────────────────────
    % ★ (1) 결과 저장용 배열 초기화
    % ────────────────────────────────────────────────────────────
    best_tau1 = zeros(numseeds,1);   % τ1*
    best_tau2 = zeros(numseeds,1);   % τ2*
    best_rmse = zeros(numseeds,1);   % 최소 RMSE

    % 시드별 Cost-surface
    for i = 1:numseeds
        V_SD = noisedata.(sprintf('V_SD%d', i));

        tau1_vec = 10.^(linspace(-1, 1.1, 21));
        tau2_vec = 10.^(linspace( 1, 2.1, 31));
        cost_surface = zeros(numel(tau2_vec), numel(tau1_vec));

        opts = optimset('Display','off','MaxIter',3000,...
                        'MaxFunEvals',1e5,'TolFun',1e-14,'TolX',1e-15);

        for ii = 1:numel(tau1_vec)
            for jj = 1:numel(tau2_vec)
                p0 = [0.0012 0.0006 0.0004 tau1_vec(ii) tau2_vec(jj)];
                lb = [0 0 0 tau1_vec(ii) tau2_vec(jj)];
                ub = [p0(1:3)*10 tau1_vec(ii) tau2_vec(jj)];
                [~,cost_surface(jj,ii)] = fmincon(@(p)RMSE_2RC(V_SD,p,t_vec,I_vec),...
                                                  p0,[],[],[],[],lb,ub,[],opts);
            end
        end

        % 최적 지점
        [min_cost,linIdx] = min(cost_surface(:));
        [r,c]             = ind2sub(size(cost_surface),linIdx);
        best_tau1(i)      = tau1_vec(c);
        best_tau2(i)      = tau2_vec(r);
        best_rmse(i)      = min_cost;     % ★ 배열에 저장

        % 3D plot (원본 코드 그대로)
        [T1,T2] = meshgrid(tau1_vec,tau2_vec);
        figure;
        surf(T1,T2,cost_surface,'EdgeColor','none','FaceColor','interp');
        view(3); shading interp; colorbar;
        xlabel('\tau_1 [s]'); ylabel('\tau_2 [s]'); zlabel('RMSE'); hold on;
        plot3(best_tau1(i),best_tau2(i),min_cost,'r*','MarkerSize',12,'LineWidth',2);
        legend(sprintf('\\tau_1^*=%.3f  \\tau_2^*=%.3f',best_tau1(i),best_tau2(i)));
    end  % seed loop 끝

    % ────────────────────────────────────────────────────────────
    % ★ (2) 테이블로 변환 → Workspace 에 바로 보기
    % ────────────────────────────────────────────────────────────
    seedID   = (1:numseeds).';
    resultTB = table(seedID,best_tau1,best_tau2,best_rmse,...
                     'VariableNames',{'Seed','Tau1','Tau2','RMSE'});

    % 확인용 출력
    disp(resultTB);
end

% 2RC RMSE 함수
function cost = RMSE_2RC(data,p,t,I)
    model = RC_model_2(p,t,I);
    cost  = sqrt(mean((data-model).^2));
end