% ========================================================================
% Driving-profile 기반 2-modal DRT sweep 식별성 검증 (600 s 제한)
% [방식 A] 두 mode를 함께 log-shift
%
% [목적]
%  - 입력: 실제 주행부하 current profile (각 파일의 앞 600초만 사용)
%  - 시스템: 2-modal DRT 기반 n-RC ground truth
%  - fitting: 2RC
%  - sweep 방식:
%       tau_mode_base = [3; 88] 에 대해
%       tau_mode = tau_mode_base * 10^(delta)
%       즉 두 mode를 같은 log 간격으로 함께 좌우 이동
%
% [출력 Figure]
%  1) 대표 2-modal DRT 분포 예시
%       - g(tau)
%       - R-tau (line + transparent fill)
%  2) Summary plot only
%       - tau1_fit vs true tau_mode1
%       - tau2_fit vs true tau_mode2
%       - R1_fit   vs true tau_mode1
%       - R2_fit   vs true tau_mode2
%       - RMSE     vs true tau_mode2
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

% ---- 2-modal DRT base setting ----
tau_mode_base = [3; 88];        % s
sigma10       = [0.200; 0.190]; % decades
w             = [5e-2; 1];      % mode area ratio

% ---- 방식 A: 두 mode를 함께 log-shift ----
% delta_sweep가 log10 축 이동량(decade)
% 예: -0.5 ~ +0.5 decade
nShift = 30;
delta_sweep = linspace(-0.5, 0.5, nShift).';   % decade shift
num_cases = numel(delta_sweep);

% 실제 tau_mode case 미리 계산
tau_mode1_case = nan(num_cases,1);
tau_mode2_case = nan(num_cases,1);
for i = 1:num_cases
    tau_mode_tmp = tau_mode_base .* 10.^delta_sweep(i);
    tau_mode1_case(i) = tau_mode_tmp(1);
    tau_mode2_case(i) = tau_mode_tmp(2);
end

fprintf('=== 방식 A: 2-modal DRT common log-shift ===\n');
fprintf('Base tau_mode = [%.3f, %.3f] s\n', tau_mode_base(1), tau_mode_base(2));
fprintf('delta range   = [%.3f, %.3f] decade (%d cases)\n', ...
    delta_sweep(1), delta_sweep(end), num_cases);
fprintf('tau_mode1 range = [%.3f, %.3f] s\n', min(tau_mode1_case), max(tau_mode1_case));
fprintf('tau_mode2 range = [%.3f, %.3f] s\n\n', min(tau_mode2_case), max(tau_mode2_case));

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
               'MaxIter', 500, ...
               'MaxFunEvals', 3000, ...
               'TolFun', eps, ...
               'TolX', eps);

% 2RC fitting initial guess / bound
% para = [R0, R1, R2, tau1, tau2]
para0 = [0.001; 0.0005; 0.0015; 3; 88];
lb    = [0;     0;      0;      0.001; 0.001];
ub    = [0.01;  0.01;   0.01;   300;   300];

% tau1 <= tau2 제약
Aineq = [0 0 0 1 -1];
bineq = 0;

%% -----------------------------------------------------------------------
% 4) 시간 길이 제한
% ------------------------------------------------------------------------
t_limit = 600;

%% -----------------------------------------------------------------------
% 5) 결과 저장용
% ------------------------------------------------------------------------
Results = struct;

R0_fit_all   = nan(nLoads, num_cases);
R1_fit_all   = nan(nLoads, num_cases);
R2_fit_all   = nan(nLoads, num_cases);
tau1_fit_all = nan(nLoads, num_cases);
tau2_fit_all = nan(nLoads, num_cases);
RMSE_all     = nan(nLoads, num_cases);

tau_mode1_true_all = nan(nLoads, num_cases);
tau_mode2_true_all = nan(nLoads, num_cases);

%% -----------------------------------------------------------------------
% 6) 대표 DRT 분포 plot
% ------------------------------------------------------------------------
idx_show = round(linspace(1, num_cases, 5));
cmap = lines(numel(idx_show));

figure('Name', 'Representative 2-modal DRT distributions', ...
       'Position', [100, 100, 1300, 500], 'Color', 'w');

% ------------------ (1) g(tau) ------------------
subplot(1,2,1); hold on; grid on;
for j = 1:numel(idx_show)
    i = idx_show(j);

    delta_dec = delta_sweep(i);
    tau_mode = tau_mode_base .* 10.^delta_dec;

    [mu10, tau_mean, tau_std, tau_median, mu_n, sigma_n, T_modes] = ...
        DRT_mu_sigma(tau_mode, sigma10);

    [theta, r_mode, r_theta, dth, R_true, tau_true, g_tau, A_modes, w_used] = ...
        DRT_Rtau(n, tau_min, tau_max, mu10, sigma10, A_tot, w);

    semilogx(tau_true, g_tau, 'LineWidth', 2, ...
        'Color', cmap(j,:), ...
        'DisplayName', sprintf('[%.2f, %.2f] s', tau_mode(1), tau_mode(2)));
end
xlabel('\tau (s)');
ylabel('g(\tau) [\Omega/s]');
title('Representative 2-modal DRT: g(\tau)');
xlim([tau_min tau_max]);
lgd = legend('Location', 'best');
title(lgd, 'tau_{mode}');

% ------------------ (2) R - tau : line + transparent fill ------------------
subplot(1,2,2); hold on; grid on;
for j = 1:numel(idx_show)
    i = idx_show(j);

    delta_dec = delta_sweep(i);
    tau_mode = tau_mode_base .* 10.^delta_dec;

    [mu10, tau_mean, tau_std, tau_median, mu_n, sigma_n, T_modes] = ...
        DRT_mu_sigma(tau_mode, sigma10);

    [theta, r_mode, r_theta, dth, R_true, tau_true, g_tau, A_modes, w_used] = ...
        DRT_Rtau(n, tau_min, tau_max, mu10, sigma10, A_tot, w);

    x_fill = [tau_true(:); flipud(tau_true(:))];
    y_fill = [R_true(:); zeros(size(R_true(:)))];

    patch(x_fill, y_fill, cmap(j,:), ...
        'FaceAlpha', 0.15, ...
        'EdgeColor', 'none', ...
        'HandleVisibility', 'off');

    semilogx(tau_true, R_true, '-', 'LineWidth', 2, ...
        'Color', cmap(j,:), ...
        'DisplayName', sprintf('[%.2f, %.2f] s', tau_mode(1), tau_mode(2)));
end
set(gca, 'XScale', 'log');
xlabel('\tau (s)');
ylabel('R_i (\Omega)');
title('Representative 2-modal DRT: discrete R-\tau');
xlim([tau_min tau_max]);
lgd = legend('Location', 'best');
title(lgd, 'tau_{mode}');

%% -----------------------------------------------------------------------
% 7) 메인 루프
% ------------------------------------------------------------------------
fprintf('>> 총 %d개 driving load × %d개 DRT shift case 시작 (각 load 앞 %d s 사용)\n', ...
    nLoads, num_cases, t_limit);

for fileIdx = 1:nLoads

    %% 7-1) load 읽기 + 앞 600초만 사용
    item = driving_paths{fileIdx};
    tbl = readtable(item);

    t_raw = tbl{:,1};
    I_raw = tbl{:,2};

    idx_use = t_raw <= t_limit;

    t_vec = t_raw(idx_use);
    I_vec = I_raw(idx_use);

    % 시간축 재정렬
    t_vec = t_vec - t_vec(1);

    [~, base_name, ~] = fileparts(item);
    fprintf('\n[%d/%d] Load: %s (0~%d s 사용)\n', fileIdx, nLoads, base_name, t_limit);

    OCV = zeros(size(t_vec));

    % per-load 결과 저장용
    R0_fit_vec   = nan(num_cases,1);
    R1_fit_vec   = nan(num_cases,1);
    R2_fit_vec   = nan(num_cases,1);
    tau1_fit_vec = nan(num_cases,1);
    tau2_fit_vec = nan(num_cases,1);
    RMSE_vec     = nan(num_cases,1);

    tau_mode1_vec = nan(num_cases,1);
    tau_mode2_vec = nan(num_cases,1);

    for i = 1:num_cases
        delta_dec = delta_sweep(i);
        tau_mode = tau_mode_base .* 10.^delta_dec;

        fprintf('   - case [%2d/%2d] delta=%.3f dec, tau_mode=[%.3f, %.3f] s ... ', ...
            i, num_cases, delta_dec, tau_mode(1), tau_mode(2));

        %% 7-2) 2-modal DRT 생성
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

        %% 7-5) 2RC fitting
        problem = createOptimProblem('fmincon', ...
            'objective', @(p) RMSE_2RC(V_meas, p, t_vec, I_vec, OCV), ...
            'x0', para0, ...
            'lb', lb, 'ub', ub, ...
            'Aineq', Aineq, 'bineq', bineq, ...
            'options', opt);

        [bestP, bestRMSE] = run(ms, problem, startPts);

        %% 7-6) 결과 저장
        R0_fit_vec(i)   = bestP(1);
        R1_fit_vec(i)   = bestP(2);
        R2_fit_vec(i)   = bestP(3);
        tau1_fit_vec(i) = bestP(4);
        tau2_fit_vec(i) = bestP(5);
        RMSE_vec(i)     = bestRMSE;

        tau_mode1_vec(i) = tau_mode(1);
        tau_mode2_vec(i) = tau_mode(2);

        fprintf('done (tau_fit=[%.3f, %.3f] s, RMSE=%.3f mV)\n', ...
            bestP(4), bestP(5), bestRMSE*1000);
    end

    %% 7-7) load별 결과 저장
    Results.(base_name).tau_mode1_true = tau_mode1_vec;
    Results.(base_name).tau_mode2_true = tau_mode2_vec;

    Results.(base_name).R0_fit   = R0_fit_vec;
    Results.(base_name).R1_fit   = R1_fit_vec;
    Results.(base_name).R2_fit   = R2_fit_vec;
    Results.(base_name).tau1_fit = tau1_fit_vec;
    Results.(base_name).tau2_fit = tau2_fit_vec;
    Results.(base_name).RMSE     = RMSE_vec;

    R0_fit_all(fileIdx,:)   = R0_fit_vec.';
    R1_fit_all(fileIdx,:)   = R1_fit_vec.';
    R2_fit_all(fileIdx,:)   = R2_fit_vec.';
    tau1_fit_all(fileIdx,:) = tau1_fit_vec.';
    tau2_fit_all(fileIdx,:) = tau2_fit_vec.';
    RMSE_all(fileIdx,:)     = RMSE_vec.';

    tau_mode1_true_all(fileIdx,:) = tau_mode1_vec.';
    tau_mode2_true_all(fileIdx,:) = tau_mode2_vec.';
end

%% -----------------------------------------------------------------------
% 8) Summary plot only
% ------------------------------------------------------------------------
figure('Name', 'Summary - All Loads Comparison (2DRT -> 2RC)', ...
       'Position', [80, 60, 1300, 1000], 'Color', 'w');

% ------------------ (1) tau1_fit ------------------
subplot(3,2,1); hold on;
for fileIdx = 1:nLoads
    [~, nm, ~] = fileparts(driving_paths{fileIdx});
    loglog(tau_mode1_true_all(fileIdx,:), tau1_fit_all(fileIdx,:), '-o', ...
        'LineWidth', 1.2, 'DisplayName', nm);
end
loglog(tau_mode1_true_all(1,:), tau_mode1_true_all(1,:), 'k--', 'LineWidth', 1.5);
grid on;
xlabel('True \tau_{mode,1} (s)');
ylabel('Estimated \tau_1 (s)');
title('\tau_1 fit vs true \tau_{mode,1}');
legend('Location', 'eastoutside');

% ------------------ (2) tau2_fit ------------------
subplot(3,2,2); hold on;
for fileIdx = 1:nLoads
    [~, nm, ~] = fileparts(driving_paths{fileIdx});
    loglog(tau_mode2_true_all(fileIdx,:), tau2_fit_all(fileIdx,:), '-o', ...
        'LineWidth', 1.2, 'DisplayName', nm);
end
loglog(tau_mode2_true_all(1,:), tau_mode2_true_all(1,:), 'k--', 'LineWidth', 1.5);
grid on;
xlabel('True \tau_{mode,2} (s)');
ylabel('Estimated \tau_2 (s)');
title('\tau_2 fit vs true \tau_{mode,2}');
legend('Location', 'eastoutside');

% ------------------ (3) R1_fit ------------------
subplot(3,2,3); hold on;
for fileIdx = 1:nLoads
    [~, nm, ~] = fileparts(driving_paths{fileIdx});
    semilogx(tau_mode1_true_all(fileIdx,:), R1_fit_all(fileIdx,:), '-s', ...
        'LineWidth', 1.2, 'DisplayName', nm);
end
grid on;
xlabel('True \tau_{mode,1} (s)');
ylabel('Estimated R_1 (\Omega)');
title('R_1 fit');

% ------------------ (4) R2_fit ------------------
subplot(3,2,4); hold on;
for fileIdx = 1:nLoads
    [~, nm, ~] = fileparts(driving_paths{fileIdx});
    semilogx(tau_mode2_true_all(fileIdx,:), R2_fit_all(fileIdx,:), '-s', ...
        'LineWidth', 1.2, 'DisplayName', nm);
end
grid on;
xlabel('True \tau_{mode,2} (s)');
ylabel('Estimated R_2 (\Omega)');
title('R_2 fit');

% ------------------ (5) RMSE ------------------
subplot(3,2,[5 6]); hold on;
for fileIdx = 1:nLoads
    [~, nm, ~] = fileparts(driving_paths{fileIdx});
    loglog(tau_mode2_true_all(fileIdx,:), RMSE_all(fileIdx,:)*1000, '-d', ...
        'LineWidth', 1.2, 'DisplayName', nm);
end
grid on;
xlabel('True \tau_{mode,2} (s)');
ylabel('RMSE (mV)');
title('RMSE');
legend('Location', 'eastoutside');

%% -----------------------------------------------------------------------
% 9) 결과 저장
% ------------------------------------------------------------------------
save('DRT2_sweep_driving_2RC_results_600s_methodA.mat', ...
    'Results', 'delta_sweep', 'tau_mode_base', ...
    'R0_fit_all', 'R1_fit_all', 'R2_fit_all', ...
    'tau1_fit_all', 'tau2_fit_all', 'RMSE_all', ...
    'tau_mode1_true_all', 'tau_mode2_true_all', ...
    't_limit', 'sigma10', 'A_tot', 'R0_true', ...
    'tau_min', 'tau_max', 'n', 'w');

fprintf('\n>> 완료: DRT2_sweep_driving_2RC_results_600s_methodA.mat 저장됨\n');

%% =======================================================================
% 로컬 함수
% =======================================================================
function cost = RMSE_2RC(data, para, t, I, OCV)
    model = RC_model_2(para, t, I, OCV);
    cost  = sqrt(mean((data - model).^2));
end

function V_est = RC_model_2(X, t_vec, I_vec, OCV)
    R0   = X(1);
    R1   = X(2);
    R2   = X(3);
    tau1 = X(4);
    tau2 = X(5);

    dt = [1; diff(t_vec)];
    N  = length(t_vec);

    V_est = zeros(N,1);
    Vrc1 = 0;
    Vrc2 = 0;

    for k = 1:N
        IR0 = R0 * I_vec(k);

        a1 = exp(-dt(k)/tau1);
        a2 = exp(-dt(k)/tau2);

        if k > 1
            Vrc1 = Vrc1*a1 + R1*(1-a1)*I_vec(k-1);
            Vrc2 = Vrc2*a2 + R2*(1-a2)*I_vec(k-1);
        end

        V_est(k) = OCV(k) + IR0 + Vrc1 + Vrc2;
    end
end