% ========================================================================
% 1-Normal DRT 식별성 검증
% ========================================================================
clear; clc; close all;

%% 1. 1-Normal DRT 기반 Ground Truth 설정 (n-RC 모델)
n       = 50;                  
tau_min = 0.1;                 
tau_max = 400;                 
A_tot   = 2.0e-3;              
R0_true = 1e-3;                

tau_mode = 10;                 
sigma10  = 0.200;              
w        = 1;                  

[mu10, tau_mean, tau_std, tau_median, mu_n, sigma_n, ~] = DRT_mu_sigma(tau_mode, sigma10);
[theta, r_mode, r_theta, dth, R_true, tau_true, g_tau, A_modes, w_used] = ...
    DRT_Rtau(n, tau_min, tau_max, mu10, sigma10, A_tot, w);

X_true = [R0_true; R_true(:); tau_true(:)];   

%% 2. 주파수 Sweep 배열 설정
f_resonance = 1 / (2 * pi * tau_mode); 

f_sweep = sort([logspace(-3, 0, 30), f_resonance]); 
num_cases = length(f_sweep);

fs = 10;                       
dt = 1/fs; 
t_end = 1500; 
t_vec = (0:dt:t_end)';

%% 3. 최적화 설정
epsilon_percent_span = 2;      
initial_state = 51;
sigma_noise = 5;

ms = MultiStart("UseParallel",true,"Display","off");
nStartPts = 20;               
startPts  = RandomStartPointSet('NumStartPoints', nStartPts);
opt = optimset('display','off','MaxIter',300,'MaxFunEvals',1000, ...
               'TolFun',eps,'TolX',eps);

para0 = [0.001; 0.002; 5]; 
lb = [0; 0; 0.001];
ub = [0.01; 0.01; 300];    
OCV = zeros(size(t_vec));

%% 4. 메인 루프
tau_est_results     = zeros(num_cases, 1);
R_est_results       = zeros(num_cases, 1);
RMSE_results        = zeros(num_cases, 1);
Sensitivity_results = zeros(num_cases, 1); 

LL_data = struct(); 

[~, fail_idx] = min(abs(f_sweep - 0.1));

fprintf('>> 총 %d개 주파수 타겟의 정밀 분석을 시작합니다... (약 1~2분 소요)\n', num_cases);

for i = 1:num_cases
    f_target = f_sweep(i);
    fprintf('   [%d/%d] 타겟: %.4f Hz ... ', i, num_cases, f_target);
    
    [I_data, f_components, amp_components] = generate_normal_fft_multisine_exact(t_vec, f_target, sigma10);
    V_true = RC_model_n(X_true, t_vec, I_data, n);
    
    rng(1); 
    [V_meas, ~, ~, ~, ~, ~] = MarkovNoise_idx(V_true, epsilon_percent_span, initial_state, sigma_noise);
    
    % 민감도 계산
    delta_tau = tau_mode * 0.01; 
    P_base = [R0_true; A_tot; tau_mode];
    P_pert = [R0_true; A_tot; tau_mode + delta_tau];
    
    V_base = RC_model_1(P_base, t_vec, I_data, OCV);
    V_pert = RC_model_1(P_pert, t_vec, I_data, OCV);
    
    dV_dtau = (V_pert - V_base) / delta_tau;
    Sensitivity_results(i) = sqrt(mean(dV_dtau.^2)); 
    
    problem = createOptimProblem('fmincon', ...
        'objective', @(p)RMSE_1RC(V_meas, p, t_vec, I_data, OCV), ...
        'x0', para0, 'lb', lb, 'ub', ub, 'options', opt);
    
    [bestP, bestRMSE] = run(ms, problem, startPts);
    
    R_est_results(i)   = bestP(2);
    tau_est_results(i) = bestP(3);
    RMSE_results(i)    = bestRMSE;
    fprintf('완료!\n');
    
    % Loss Landscape 데이터 캡처
    if abs(f_target - f_resonance) < 1e-6
        LL_data.Resonance.I_data = I_data;
        LL_data.Resonance.V_meas = V_meas;
        LL_data.Resonance.f_target = f_target;
        LL_data.Resonance.bestP = bestP;
    end
    if i == fail_idx 
        LL_data.Failed.I_data = I_data;
        LL_data.Failed.V_meas = V_meas;
        LL_data.Failed.f_target = f_target;
        LL_data.Failed.bestP = bestP;
    end
    
    %% [Detailed View] 공진 주파수일 때 a1 스타일 시각화
    if abs(f_target - f_resonance) < 1e-6
        V_1rc_res = RC_model_1(bestP, t_vec, I_data, OCV);
        residual_res = V_meas - V_1rc_res;
        
        fig_res = figure('Name', 'Detailed View: Resonance Case', 'Position', [50, 50, 1400, 900], 'Color', 'w');
                 
        subplot(3, 3, [1, 2]);
        plot(t_vec, I_data, 'b-', 'LineWidth', 1.2);
        grid on; xlabel('Time (s)'); ylabel('Current (A)');
        title({'1. Input Current Profile', ''}, 'FontWeight', 'bold'); xlim([0, 600]); 
        
        subplot(3, 3, [4, 5]);
        plot(t_vec, V_meas, 'Color', [0.7 0.7 0.7], 'DisplayName', 'V_{meas} (Noisy)'); hold on;
        plot(t_vec, V_1rc_res, 'r-', 'LineWidth', 1.5, 'DisplayName', 'V_{1RC} (Fit)');
        grid on; xlabel('Time (s)'); ylabel('Voltage (V)');
        title({sprintf('2. Voltage Fit Response (RMSE: %.2f mV)', bestRMSE * 1000), ''}, 'FontWeight', 'bold');
        legend('Location', 'northeast'); xlim([0, 600]);
        
        subplot(3, 3, 7);
        plot(t_vec, residual_res, 'k-', 'LineWidth', 1);
        grid on; xlabel('Time (s)'); ylabel('Error (V)');
        title({'3. Residual Error', ''}, 'FontWeight', 'bold'); xlim([0, 600]);
        
        ax1 = subplot(3, 3, 8);
        stem(f_components, amp_components, 'b', 'LineWidth', 1.5, 'MarkerFaceColor', 'b', 'DisplayName', 'Injected (31)'); hold on;
        xline(f_target, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Target Freq');
        set(ax1, 'XScale', 'log'); xlim(ax1, [1e-4, 1e0]); 
        grid on; xlabel(ax1, 'Frequency (Hz)'); ylabel(ax1, 'Amplitude');
        legend(ax1, 'Location', 'northwest', 'FontSize', 8);
        add_tau_top_axis(ax1, '4. Current Freq Spectrum'); 
        
        subplot(3, 3, [3, 6, 9]);
        max_g = max(g_tau); max_R = max(R_true); scale_factor = max_R / max_g; 
        
        yyaxis left;
        plot(tau_true, g_tau, 'k-', 'LineWidth', 2.5, 'DisplayName', 'True g(\tau)'); hold on;
        ylabel('g(\tau) [\Omega/s]'); set(gca, 'YColor', 'k'); ylim([0, max_g * 1.15]); 
        
        xline(tau_mode, 'k-', 'LineWidth', 1.5, 'DisplayName', sprintf('\\tau_{mode} (%.1fs)', tau_mode));
        xline(tau_mean, 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('\\tau_{mean} (%.1fs)', tau_mean));
        xline(tau_median, 'b-.', 'LineWidth', 1.5, 'DisplayName', sprintf('\\tau_{median} (%.1fs)', tau_median));
        xline(bestP(3), 'r:', 'LineWidth', 2.5, 'DisplayName', sprintf('\\tau_{fit} (%.1fs)', bestP(3)));
        
        yyaxis right;
        stem(tau_true, R_true, 'Color', [0.6 0.6 0.6], 'Marker', '.', 'LineWidth', 1.5, 'DisplayName', 'True n-RC');
        ylabel('Resistance R (\Omega)'); set(gca, 'YColor', [0.5 0.5 0.5]); ylim([0, (max_g * 1.15) * scale_factor]); 
        set(gca, 'XScale', 'log'); xlim([1e-2, 1e3]); 
        grid on; xlabel('Time Constant \tau (s)');
        title({'5. GT Spectrum (\tau Match)', ''}, 'FontWeight', 'bold');
        legend('Location', 'northwest', 'FontSize', 9);
    end
end

%% 5. Correlation Plot (4분할 & 보조축 겹침 완벽 방지)
figure('Name', 'Correlation & Identifiability Proof', 'Position', [100, 100, 1200, 900], 'Color', 'w');

ax_c1 = subplot(2, 2, 1);
scatter(f_sweep, RMSE_results * 1000, 50, 'k', 'filled'); hold on; % 점이 많아졌으므로 크기를 살짝 줄임(70->50)
xline(f_resonance, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Resonance (%.4f Hz)', f_resonance));
grid on; set(ax_c1, 'XScale', 'log', 'YScale', 'log', 'XLim', [1e-4, 1e0]); 
xlabel(ax_c1, 'Input Freq (Hz)'); ylabel(ax_c1, 'Voltage RMSE (mV)');
add_tau_top_axis(ax_c1, '1. Correlation: RMSE');

ax_c2 = subplot(2, 2, 2);
scatter(f_sweep, tau_est_results, 50, 'b', 'filled', 'DisplayName', 'Estimated \tau_1'); hold on;
yline(tau_mode, 'k-', 'LineWidth', 2, 'DisplayName', sprintf('True \\tau_{mode} (%.1f s)', tau_mode));
xline(f_resonance, 'r--', 'LineWidth', 2, 'DisplayName', 'Resonance Freq.');
yline(300, 'k:', 'HandleVisibility', 'off'); yline(0.001, 'k:', 'HandleVisibility', 'off');
grid on; set(ax_c2, 'XScale', 'log', 'YScale', 'log', 'XLim', [1e-4, 1e0]);
xlabel(ax_c2, 'Input Freq (Hz)'); ylabel(ax_c2, 'Estimated \tau_1 (s)');
legend('Location', 'southwest', 'FontSize', 8);
add_tau_top_axis(ax_c2, '2. Correlation: Estimated \tau');

ax_c3 = subplot(2, 2, 3);
scatter(f_sweep, R_est_results, 50, 'm', 'filled', 'DisplayName', 'Estimated R_1'); hold on;
yline(A_tot, 'k-', 'LineWidth', 2, 'DisplayName', sprintf('Total True R (%.3f \\Omega)', A_tot));
xline(f_resonance, 'r--', 'LineWidth', 2, 'DisplayName', 'Resonance Freq.');
grid on; set(ax_c3, 'XScale', 'log', 'XLim', [1e-4, 1e0]);
xlabel(ax_c3, 'Input Freq (Hz)'); ylabel(ax_c3, 'Estimated Resistance R_1 (\Omega)');
legend('Location', 'southwest', 'FontSize', 8);
add_tau_top_axis(ax_c3, '3. Correlation: Estimated R_1');

ax_c4 = subplot(2, 2, 4);
plot(f_sweep, Sensitivity_results, '-go', 'LineWidth', 2, 'MarkerFaceColor', 'g', 'DisplayName', '|dV / d\tau| (RMS)'); hold on;
xline(f_resonance, 'r--', 'LineWidth', 2, 'DisplayName', 'Resonance Freq.');
grid on; set(ax_c4, 'XScale', 'log', 'XLim', [1e-4, 1e0]);
xlabel(ax_c4, 'Input Freq (Hz)'); ylabel(ax_c4, 'Sensitivity Magnitude (V/s)');
legend('Location', 'northwest', 'FontSize', 8);
add_tau_top_axis(ax_c4, '4. Math Proof: Sensitivity to \tau_1');

%% 6. Loss Landscape (비용 함수 지형도) 시각화
if isfield(LL_data, 'Failed') && isfield(LL_data, 'Resonance')
    figure('Name', 'Loss Landscape Analysis', 'Position', [150, 150, 1200, 500], 'Color', 'w');

    N_grid = 40;
    tau_grid = logspace(-1, 2.5, N_grid); 
    R_grid   = linspace(1e-4, 4e-3, N_grid); 
    [Tau_X, R_Y] = meshgrid(tau_grid, R_grid);

    Cost_Res = zeros(N_grid, N_grid);
    for r = 1:N_grid
        for c = 1:N_grid
            p_temp = [R0_true; R_Y(r,c); Tau_X(r,c)];
            Cost_Res(r,c) = RMSE_1RC(LL_data.Resonance.V_meas, p_temp, t_vec, LL_data.Resonance.I_data, OCV);
        end
    end

    Cost_Fail = zeros(N_grid, N_grid);
    for r = 1:N_grid
        for c = 1:N_grid
            p_temp = [R0_true; R_Y(r,c); Tau_X(r,c)];
            Cost_Fail(r,c) = RMSE_1RC(LL_data.Failed.V_meas, p_temp, t_vec, LL_data.Failed.I_data, OCV);
        end
    end

    subplot(1, 2, 1);
    contourf(Tau_X, R_Y, Cost_Res * 1000, 30, 'LineColor', 'none'); colorbar; hold on;
    h1 = plot(tau_mode, A_tot, 'r+', 'MarkerSize', 10, 'LineWidth', 2.5);
    h2 = plot(LL_data.Resonance.bestP(3), LL_data.Resonance.bestP(2), 'w*', 'MarkerSize', 10, 'LineWidth', 1.5);
    set(gca, 'XScale', 'log'); xlabel('\tau_1 (s)'); ylabel('R_1 (\Omega)');
    title(sprintf('Loss Landscape @ Resonance (%.4f Hz)', LL_data.Resonance.f_target), 'FontWeight', 'bold');
    
    leg1 = legend([h1, h2], {'Ground Truth (+)', 'Optimum (*)'}, 'Location', 'northeast');
    set(leg1, 'Color', 'w', 'TextColor', 'k', 'EdgeColor', 'k');

    subplot(1, 2, 2);
    contourf(Tau_X, R_Y, Cost_Fail * 1000, 30, 'LineColor', 'none'); colorbar; hold on;
    h3 = plot(tau_mode, A_tot, 'r+', 'MarkerSize', 10, 'LineWidth', 2.5);
    h4 = plot(LL_data.Failed.bestP(3), LL_data.Failed.bestP(2), 'w*', 'MarkerSize', 10, 'LineWidth', 1.5);
    set(gca, 'XScale', 'log'); xlabel('\tau_1 (s)'); ylabel('R_1 (\Omega)');
    title(sprintf('Loss Landscape @ Off-resonance (%.4f Hz)', LL_data.Failed.f_target), 'FontWeight', 'bold');
    
    leg2 = legend([h3, h4], {'Ground Truth (+)', 'Optimum (*)'}, 'Location', 'northeast');
    set(leg2, 'Color', 'w', 'TextColor', 'k', 'EdgeColor', 'k');
end

%% =======================================================================
% 로컬 함수 모음
% =======================================================================
function [I_out, f_vec, amp_components] = generate_normal_fft_multisine_exact(t, f_center, sigma_f_log)
    N_freq = 31; 
    f_vec = logspace(log10(f_center) - 3*sigma_f_log, log10(f_center) + 3*sigma_f_log, N_freq);
    I_out = zeros(size(t));
    amp_components = zeros(1, N_freq);
    for k = 1:N_freq
        f = f_vec(k);
        amp = exp(-(log10(f) - log10(f_center))^2 / (2 * sigma_f_log^2));
        amp_components(k) = amp;
        phase = rand * 2 * pi; 
        I_out = I_out + amp * sin(2 * pi * f * t + phase);
    end
    max_val = max(abs(I_out));
    I_out = 2 * I_out / max_val; 
    amp_components = 2 * amp_components / max_val; 
end

function V_est = RC_model_1(X, t_vec, I_vec, OCV)
    R0 = X(1); R1 = X(2); tau1 = X(3);
    dt = [1; diff(t_vec)]; 
    N = length(t_vec);
    V_est = zeros(N, 1);
    Vrc1 = 0;
    for k = 1:N
        IR0 = R0 * I_vec(k);
        alpha1 = exp(-dt(k)/tau1);
        if k > 1
            Vrc1 = Vrc1*alpha1 + R1*(1 - alpha1)*I_vec(k-1);
        end
        V_est(k) = OCV(k) + IR0 + Vrc1;
    end
end

function cost = RMSE_1RC(data, para, t, I, OCV)
    model = RC_model_1(para, t, I, OCV);
    cost  = sqrt(mean((data - model).^2));
end

function add_tau_top_axis(ax_bottom, title_str)
    drawnow; 
    ax_top = axes('Position', ax_bottom.Position, 'XAxisLocation', 'top', ...
                  'YAxisLocation', 'right', 'Color', 'none', 'YTick', []);
    
    set(ax_top, 'XScale', 'log', 'XLim', ax_bottom.XLim);
    
    tau_ticks = [0.1, 1, 10, 100, 1000];
    f_ticks_for_tau = 1 ./ (2 * pi * tau_ticks);
    
    [f_ticks_for_tau, sort_idx] = sort(f_ticks_for_tau);
    tau_ticks = tau_ticks(sort_idx);
    
    valid_idx = f_ticks_for_tau >= ax_bottom.XLim(1) & f_ticks_for_tau <= ax_bottom.XLim(2);
    ax_top.XTick = f_ticks_for_tau(valid_idx);
    ax_top.XTickLabel = string(tau_ticks(valid_idx));
    
    xlabel(ax_top, 'Corresponding \tau (s)', 'FontSize', 9, 'Color', [0.4 0.4 0.4]);
    ax_top.XColor = [0.4 0.4 0.4]; 
    
    title(ax_top, title_str, 'FontWeight', 'bold', 'Color', 'k');
end