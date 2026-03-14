% ========================================================================
% Driving-profile 기반 DRT sweep 식별성 검증 (600 s 제한, tau_mode 기준 표시)
%
% [목적]
%  - 입력: 실제 주행부하 current profile (각 파일의 앞 600초만 사용)
%  - 시스템: unimodal DRT 기반 n-RC ground truth
%  - sweep 변수: tau_mode (logspace로 이동)
%  - 절차:
%       DRT 생성 -> nRC synthetic voltage 생성 -> Markov noise 추가
%       -> 1RC fitting -> tau_fit, R_fit, RMSE 비교
%
% [출력 Figure]
%  1) 대표 DRT 분포 예시
%       - g(tau)
%       - R-tau (line + transparent fill)
%  2) Summary plot only
%       - tau_fit vs true tau_mode
%       - R_fit   vs true tau_mode
%       - RMSE    vs true tau_mode
%
% [필수 외부 함수]
%  - DRT_mu_sigma.m
%  - DRT_Rtau.m
%  - RC_model_n.m
%  - MarkovNoise_idx.m
%
% ========================================================================
clear; clc; close all;

%% -----------------------------------------------------------------------
% 1) DRT / Ground Truth 기본 설정
% ------------------------------------------------------------------------
n       = 50;          % nRC branch 수
tau_min = 0.1;         % DRT tau 최소 [s]
tau_max = 400;         % DRT tau 최대 [s]
A_tot   = 2.0e-3;      % 총 polarization resistance [Ohm]
R0_true = 1.0e-3;      % ohmic resistance [Ohm]

% ---- unimodal DRT width 고정 ----
sigma10 = 0.200;       % log10(tau) 기준 표준편차 (decade)
w       = 1;           % unimodal 이므로 1

% ---- tau_mode sweep: logspace 방식 ----
% 예: 3 ~ 88 s 사이를 로그 균등 분할
nTauSweep = 30;
tau_mode_sweep = logspace(log10(3), log10(88), nTauSweep).';

%% -----------------------------------------------------------------------
% 2) Driving data 경로
% ------------------------------------------------------------------------
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

nLoads = numel(driving_paths);
num_tau_cases = numel(tau_mode_sweep);

%% -----------------------------------------------------------------------
% 3) Noise / Optimization 설정
% ------------------------------------------------------------------------
epsilon_percent_span = 2;
initial_state = 51;
sigma_noise   = 5;

ms = MultiStart("UseParallel", true, "Display", "off");
nStartPts = 20;
startPts  = RandomStartPointSet('NumStartPoints', nStartPts);

opt = optimset('display','off', ...
               'MaxIter', 300, ...
               'MaxFunEvals', 1000, ...
               'TolFun', eps, ...
               'TolX', eps);

% 1RC fitting initial guess / bound
para0 = [0.001; 0.002; 10];     % [R0, R1, tau1]
lb    = [0;     0;     0.001];
ub    = [0.01;  0.01;  300];

%% -----------------------------------------------------------------------
% 4) 시간 길이 제한
% ------------------------------------------------------------------------
t_limit = 600;   % 각 주행부하의 앞 600초만 사용

%% -----------------------------------------------------------------------
% 5) 결과 저장용
% ------------------------------------------------------------------------
Results = struct;

tau_fit_all  = nan(nLoads, num_tau_cases);
R_fit_all    = nan(nLoads, num_tau_cases);
R0_fit_all   = nan(nLoads, num_tau_cases);
RMSE_all     = nan(nLoads, num_tau_cases);

tau_mode_true_all   = nan(nLoads, num_tau_cases);
tau_mean_true_all   = nan(nLoads, num_tau_cases);
tau_median_true_all = nan(nLoads, num_tau_cases);

%% -----------------------------------------------------------------------
% 6) 대표 DRT 분포 plot (tau_mode 기준 표시)
% ------------------------------------------------------------------------
idx_show = round(linspace(1, num_tau_cases, 5));   % 대표 case 5개
cmap = lines(numel(idx_show));

figure('Name', 'Representative DRT distributions', ...
       'Position', [100, 100, 1300, 500], 'Color', 'w');

% ------------------ (1) g(tau) ------------------
subplot(1,2,1); hold on; grid on;
for j = 1:numel(idx_show)
    i = idx_show(j);
    tau_mode = tau_mode_sweep(i);

    [mu10, tau_mean, tau_std, tau_median, mu_n, sigma_n, T_modes] = ...
        DRT_mu_sigma(tau_mode, sigma10);

    [theta, r_mode, r_theta, dth, R_true, tau_true, g_tau, A_modes, w_used] = ...
        DRT_Rtau(n, tau_min, tau_max, mu10, sigma10, A_tot, w);

    semilogx(tau_true, g_tau, 'LineWidth', 2, ...
        'Color', cmap(j,:), ...
        'DisplayName', sprintf('tau_mode=%.2f s', tau_mode));
end
xlabel('\tau (s)');
ylabel('g(\tau) [\Omega/s]');
title('Representative DRT: g(\tau)');
legend('Location', 'best');
xlim([tau_min tau_max]);

% ------------------ (2) R - tau : line + transparent fill ------------------
subplot(1,2,2); hold on; grid on;
for j = 1:numel(idx_show)
    i = idx_show(j);
    tau_mode = tau_mode_sweep(i);

    [mu10, tau_mean, tau_std, tau_median, mu_n, sigma_n, T_modes] = ...
        DRT_mu_sigma(tau_mode, sigma10);

    [theta, r_mode, r_theta, dth, R_true, tau_true, g_tau, A_modes, w_used] = ...
        DRT_Rtau(n, tau_min, tau_max, mu10, sigma10, A_tot, w);

    % line + fill
    x_fill = [tau_true(:); flipud(tau_true(:))];
    y_fill = [R_true(:); zeros(size(R_true(:)))];

    patch(x_fill, y_fill, cmap(j,:), ...
        'FaceAlpha', 0.18, ...
        'EdgeColor', 'none', ...
        'HandleVisibility', 'off');

    semilogx(tau_true, R_true, '-', 'LineWidth', 2, ...
        'Color', cmap(j,:), ...
        'DisplayName', sprintf('tau_mode=%.2f s', tau_mode));
end
set(gca, 'XScale', 'log');
xlabel('\tau (s)');
ylabel('R_i (\Omega)');
title('Representative DRT: discrete R-\tau');
legend('Location', 'best');
xlim([tau_min tau_max]);

%% -----------------------------------------------------------------------
% 7) 메인 루프
% ------------------------------------------------------------------------
fprintf('>> 총 %d개 driving load × %d개 tau_mode sweep 시작 (각 load 앞 %d s 사용)\n', ...
    nLoads, num_tau_cases, t_limit);

for fileIdx = 1:nLoads

    %% 7-1) load 읽기 + 앞 600초만 사용
    item = driving_paths{fileIdx};
    tbl = readtable(item);

    t_raw = tbl{:,1};
    I_raw = tbl{:,2};

    idx_use = t_raw <= t_limit;

    t_vec = t_raw(idx_use);
    I_vec = I_raw(idx_use);

    % 시간축 0부터 재정렬
    t_vec = t_vec - t_vec(1);

    [~, base_name, ~] = fileparts(item);
    fprintf('\n[%d/%d] Load: %s (0~%d s 사용)\n', fileIdx, nLoads, base_name, t_limit);

    OCV = zeros(size(t_vec));

    % per-load 결과 저장용
    tau_fit_vec  = nan(num_tau_cases,1);
    R_fit_vec    = nan(num_tau_cases,1);
    R0_fit_vec   = nan(num_tau_cases,1);
    RMSE_vec     = nan(num_tau_cases,1);

    tau_mode_vec   = nan(num_tau_cases,1);
    tau_mean_vec   = nan(num_tau_cases,1);
    tau_median_vec = nan(num_tau_cases,1);

    for i = 1:num_tau_cases
        tau_mode = tau_mode_sweep(i);

        fprintf('   - tau_mode case [%2d/%2d] = %.4f s ... ', i, num_tau_cases, tau_mode);

        %% 7-2) unimodal DRT 생성
        [mu10, tau_mean, tau_std, tau_median, mu_n, sigma_n, T_modes] = ...
            DRT_mu_sigma(tau_mode, sigma10);

        [theta, r_mode, r_theta, dth, R_true, tau_true, g_tau, A_modes, w_used] = ...
            DRT_Rtau(n, tau_min, tau_max, mu10, sigma10, A_tot, w);

        X_true = [R0_true; R_true(:); tau_true(:)];

        %% 7-3) nRC synthetic voltage 생성
        V_true = RC_model_n(X_true, t_vec, I_vec, n);

        %% 7-4) noise 추가
        rng(1);
        [V_meas, ~, ~, ~, ~, ~] = MarkovNoise_idx( ...
            V_true, epsilon_percent_span, initial_state, sigma_noise);

        %% 7-5) 1RC fitting
        problem = createOptimProblem('fmincon', ...
            'objective', @(p) RMSE_1RC(V_meas, p, t_vec, I_vec, OCV), ...
            'x0', para0, 'lb', lb, 'ub', ub, 'options', opt);

        [bestP, bestRMSE] = run(ms, problem, startPts);

        %% 7-6) 결과 저장
        R0_fit_vec(i)  = bestP(1);
        R_fit_vec(i)   = bestP(2);
        tau_fit_vec(i) = bestP(3);
        RMSE_vec(i)    = bestRMSE;

        tau_mode_vec(i)   = tau_mode;
        tau_mean_vec(i)   = tau_mean;
        tau_median_vec(i) = tau_median;

        fprintf('done (tau_fit=%.4f s, R_fit=%.6f Ohm, RMSE=%.3f mV)\n', ...
            bestP(3), bestP(2), bestRMSE*1000);
    end

    %% 7-7) load별 결과 저장
    Results.(base_name).tau_mode_true   = tau_mode_vec;
    Results.(base_name).tau_mean_true   = tau_mean_vec;
    Results.(base_name).tau_median_true = tau_median_vec;

    Results.(base_name).tau_fit  = tau_fit_vec;
    Results.(base_name).R_fit    = R_fit_vec;
    Results.(base_name).R0_fit   = R0_fit_vec;
    Results.(base_name).RMSE     = RMSE_vec;

    tau_fit_all(fileIdx,:)        = tau_fit_vec.';
    R_fit_all(fileIdx,:)          = R_fit_vec.';
    R0_fit_all(fileIdx,:)         = R0_fit_vec.';
    RMSE_all(fileIdx,:)           = RMSE_vec.';

    tau_mode_true_all(fileIdx,:)   = tau_mode_vec.';
    tau_mean_true_all(fileIdx,:)   = tau_mean_vec.';
    tau_median_true_all(fileIdx,:) = tau_median_vec.';
end

%% -----------------------------------------------------------------------
% 8) Summary plot only (all loads comparison)
% ------------------------------------------------------------------------
figure('Name', 'Summary - All Loads Comparison', ...
       'Position', [80, 80, 1200, 950], 'Color', 'w');

% ------------------ (1) tau_fit ------------------
subplot(3,1,1); hold on;
for fileIdx = 1:nLoads
    [~, nm, ~] = fileparts(driving_paths{fileIdx});
    loglog(tau_mode_sweep, tau_fit_all(fileIdx,:), '-o', 'LineWidth', 1.3, ...
        'DisplayName', nm);
end
loglog(tau_mode_sweep, tau_mode_sweep, 'k--', 'LineWidth', 1.5, ...
    'DisplayName', 'y=x');
grid on;
xlabel('True \tau_{mode} (s)');
ylabel('Estimated \tau_{fit} (s)');
title('Summary: \tau_{fit} vs true \tau_{mode}');
legend('Location', 'eastoutside');

% ------------------ (2) R_fit ------------------
subplot(3,1,2); hold on;
for fileIdx = 1:nLoads
    [~, nm, ~] = fileparts(driving_paths{fileIdx});
    semilogx(tau_mode_sweep, R_fit_all(fileIdx,:), '-s', 'LineWidth', 1.3, ...
        'DisplayName', nm);
end
yline(A_tot, 'k--', 'LineWidth', 1.5, 'DisplayName', 'True total R (=A_{tot})');
grid on;
xlabel('True \tau_{mode} (s)');
ylabel('Estimated R_1 (\Omega)');
title('Summary: R_{fit} vs true \tau_{mode}');
legend('Location', 'eastoutside');

% ------------------ (3) RMSE ------------------
subplot(3,1,3); hold on;
for fileIdx = 1:nLoads
    [~, nm, ~] = fileparts(driving_paths{fileIdx});
    loglog(tau_mode_sweep, RMSE_all(fileIdx,:)*1000, '-d', 'LineWidth', 1.3, ...
        'DisplayName', nm);
end
grid on;
xlabel('True \tau_{mode} (s)');
ylabel('RMSE (mV)');
title('Summary: RMSE vs true \tau_{mode}');
legend('Location', 'eastoutside');

%% -----------------------------------------------------------------------
% 9) 결과 저장
% ------------------------------------------------------------------------
save('DRT_sweep_driving_1RC_results_600s.mat', ...
    'Results', 'tau_mode_sweep', ...
    'tau_fit_all', 'R_fit_all', 'R0_fit_all', 'RMSE_all', ...
    'tau_mode_true_all', 'tau_mean_true_all', 'tau_median_true_all', ...
    't_limit', 'sigma10', 'A_tot', 'R0_true', 'tau_min', 'tau_max', 'n');

fprintf('\n>> 완료: DRT_sweep_driving_1RC_results_600s.mat 저장됨\n');

%% =======================================================================
% 로컬 함수
% =======================================================================
function cost = RMSE_1RC(data, para, t, I, OCV)
    model = RC_model_1(para, t, I, OCV);
    cost  = sqrt(mean((data - model).^2));
end

function V_est = RC_model_1(X, t_vec, I_vec, OCV)
    R0   = X(1);
    R1   = X(2);
    tau1 = X(3);

    dt = [1; diff(t_vec)];
    N  = length(t_vec);

    V_est = zeros(N,1);
    Vrc1 = 0;

    for k = 1:N
        IR0 = R0 * I_vec(k);
        alpha1 = exp(-dt(k)/tau1);

        if k > 1
            Vrc1 = Vrc1*alpha1 + R1*(1-alpha1)*I_vec(k-1);
        end

        V_est(k) = OCV(k) + IR0 + Vrc1;
    end
end