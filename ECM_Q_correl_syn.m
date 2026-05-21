%% =====================================================================
%  tau_feature_subplot.m
%
%  x-axis : Capacity (%)
%  y-axis : 5 derived features
%
%  Fresh reference:
%   - cell with maximum capacity
%
%  Feature direction:
%   x1 : fitted / fresh
%   x2 : fitted / fresh
%   x3 : fitted / fresh
%   x4 : fresh / fitted   <-- C2 only
%   x5 : softplus(log(fitted / fresh)) - log(2)
%
%  useLog = true:
%   x1 = log(R0 / R0_fresh)
%   x2 = log((R1 + R2) / (R1_fresh + R2_fresh))
%   x3 = log((tau2/tau1) / (tau2_fresh/tau1_fresh))
%   x4 = log(C2_fresh / C2),  C2 = tau2 / R2
%   x5 = softplus(log(R1/R1_fresh)) - log(2)
%
%  useLog = false:
%   x1 = R0 / R0_fresh
%   x2 = (R1 + R2) / (R1_fresh + R2_fresh)
%   x3 = (tau2/tau1) / (tau2_fresh/tau1_fresh)
%   x4 = C2_fresh / C2
%   x5 = softplus(log(R1/R1_fresh)) - log(2)
%
%  No file saving
% ======================================================================

clc; clear; close all;

%% === 0) log on/off ====================================================
% true  : x1~x4에 log 적용
% false : x1~x4를 raw ratio로 표시
useLog = true;

%% === 1) 결과 MAT 파일 경로 ===========================================
results_file = 'G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\20degC\2RC_fitting_600s\2RC_results_600s.mat';

if ~isfile(results_file)
    error('결과 파일을 찾을 수 없습니다:\n%s\nresults_file 경로를 확인하세요.', results_file);
end

S = load(results_file);

if ~isfield(S, 'Tbl_Load_ECM')
    error(['결과 MAT 파일 안에 Tbl_Load_ECM 변수가 없습니다.\n' ...
           '기존 fitting 코드에서 Tbl_Load_ECM 생성 및 저장 여부를 확인하세요.']);
end

Tbl_Load_ECM = S.Tbl_Load_ECM;

%% === 2) Capacity / SOH 입력 ==========================================
capacity = [57.49; 57.57; 54.00; 52.22; 53.45; 51.28; ...
            57.91; 56.51; 42.14; 57.27; 57.18; 58.40];

%% === 3) Fresh 기준 셀 =================================================
[capacity_fresh, fresh_idx] = max(capacity);

fprintf('\nFresh reference cell index = %d\n', fresh_idx);
fprintf('Fresh reference capacity   = %.2f\n', capacity_fresh);
fprintf('useLog                     = %d\n\n', useLog);

%% === 4) 분석 설정 =====================================================
loadNames = {'US06','UDDS','HWFET','WLTP','CITY1','CITY2','HW1','HW2'};
SOC_list_analysis = [70];

nLoads = numel(loadNames);
colors = lines(nLoads);

%% === 5) 결과 확인용 테이블 ============================================
Result_Features = table();
row = 0;

%% =====================================================================
%  6) SOC별 figure 생성
% ======================================================================

for si = 1:numel(SOC_list_analysis)

    soc = SOC_list_analysis(si);

    fig = figure('Color','w', ...
                 'Name', sprintf('SOC%d_5features_vs_capacity', soc), ...
                 'Position', [100 80 1450 850]);

    if useLog
        modeText = 'log ratio mode';
    else
        modeText = 'raw ratio mode';
    end

    sgtitle(sprintf('SOC%d: 5 Derived Features vs Capacity (%s, Fresh cell #%d, %.2f%%)', ...
        soc, modeText, fresh_idx, capacity_fresh), ...
        'FontSize', 14, ...
        'FontWeight', 'bold', ...
        'Interpreter', 'none');

    ax(1) = subplot(3,2,1); hold(ax(1),'on'); grid(ax(1),'on'); box(ax(1),'on');
    ax(2) = subplot(3,2,2); hold(ax(2),'on'); grid(ax(2),'on'); box(ax(2),'on');
    ax(3) = subplot(3,2,3); hold(ax(3),'on'); grid(ax(3),'on'); box(ax(3),'on');
    ax(4) = subplot(3,2,4); hold(ax(4),'on'); grid(ax(4),'on'); box(ax(4),'on');
    ax(5) = subplot(3,2,5); hold(ax(5),'on'); grid(ax(5),'on'); box(ax(5),'on');
    ax(6) = subplot(3,2,6); axis(ax(6),'off');

    for li = 1:nLoads

        loadName = loadNames{li};

        if ~isfield(Tbl_Load_ECM, loadName)
            warning('Tbl_Load_ECM.%s 가 없습니다. 해당 부하는 건너뜁니다.', loadName);
            continue;
        end

        Tload = Tbl_Load_ECM.(loadName);

        if height(Tload) ~= numel(capacity)
            error(['%s: capacity 개수와 테이블 행 개수가 다릅니다.\n' ...
                   'capacity 개수 = %d, table 행 개수 = %d'], ...
                   loadName, numel(capacity), height(Tload));
        end

        cellNames = string(Tload.Properties.RowNames);

        %% === column names ============================================
        R0_col   = sprintf('SOC%d_R0_mOhm', soc);
        R1_col   = sprintf('SOC%d_R1_mOhm', soc);
        R2_col   = sprintf('SOC%d_R2_mOhm', soc);
        tau1_col = sprintf('SOC%d_tau1', soc);
        tau2_col = sprintf('SOC%d_tau2', soc);

        neededCols = {R0_col, R1_col, R2_col, tau1_col, tau2_col};

        for cc = 1:numel(neededCols)
            if ~ismember(neededCols{cc}, Tload.Properties.VariableNames)
                error('%s 테이블에 %s 컬럼이 없습니다.', loadName, neededCols{cc});
            end
        end

        %% === parameter vectors =======================================
        R0   = Tload.(R0_col);
        R1   = Tload.(R1_col);
        R2   = Tload.(R2_col);
        tau1 = Tload.(tau1_col);
        tau2 = Tload.(tau2_col);

        %% === fresh parameters ========================================
        R0_fresh   = R0(fresh_idx);
        R1_fresh   = R1(fresh_idx);
        R2_fresh   = R2(fresh_idx);
        tau1_fresh = tau1(fresh_idx);
        tau2_fresh = tau2(fresh_idx);

        %% === validity for fresh values ===============================
        if ~isfinite(R0_fresh) || ~isfinite(R1_fresh) || ~isfinite(R2_fresh) || ...
           ~isfinite(tau1_fresh) || ~isfinite(tau2_fresh) || ...
           R0_fresh <= 0 || R1_fresh <= 0 || R2_fresh <= 0 || ...
           tau1_fresh <= 0 || tau2_fresh <= 0

            warning('%s SOC%d: fresh 기준 파라미터가 유효하지 않아 건너뜁니다.', loadName, soc);
            continue;
        end

        %% === C2 definition ===========================================
        % 2RC model에서 tau2 = R2*C2 이므로 C2 = tau2/R2
        C2       = tau2 ./ R2;
        C2_fresh = tau2_fresh ./ R2_fresh;

        %% =================================================================
        %  Feature raw ratios
        %
        %  x1, x2, x3: fitted / fresh
        %  x4        : fresh / fitted
        %  x5        : fitted / fresh inside softplus
        % ==================================================================

        % x1 raw = R0_hat / R0_fresh
        ratio_x1 = R0 ./ R0_fresh;

        % x2 raw = (R1_hat + R2_hat) / (R1_fresh + R2_fresh)
        ratio_x2 = (R1 + R2) ./ (R1_fresh + R2_fresh);

        % x3 raw = (tau2_hat/tau1_hat) / (tau2_fresh/tau1_fresh)
        tau_ratio_hat   = tau2 ./ tau1;
        tau_ratio_fresh = tau2_fresh ./ tau1_fresh;
        ratio_x3 = tau_ratio_hat ./ tau_ratio_fresh;

        % x4 raw = C2_fresh / C2_hat
        % C2 항만 fresh가 분자
        ratio_x4 = C2_fresh ./ C2;

        % x5 = softplus(log(R1_hat/R1_fresh)) - log(2)
        ratio_x5_inner = R1 ./ R1_fresh;
        x5 = softplus_local(log(ratio_x5_inner)) - log(2);

        %% === log on/off for x1~x4 ====================================
        if useLog
            x1 = log(ratio_x1);
            x2 = log(ratio_x2);
            x3 = log(ratio_x3);
            x4 = log(ratio_x4);
        else
            x1 = ratio_x1;
            x2 = ratio_x2;
            x3 = ratio_x3;
            x4 = ratio_x4;
        end

        %% === plot =====================================================
        plot_feature_line(ax(1), capacity, x1, colors(li,:), loadName);
        plot_feature_line(ax(2), capacity, x2, colors(li,:), loadName);
        plot_feature_line(ax(3), capacity, x3, colors(li,:), loadName);
        plot_feature_line(ax(4), capacity, x4, colors(li,:), loadName);
        plot_feature_line(ax(5), capacity, x5, colors(li,:), loadName);

        %% === result table ============================================
        for k = 1:numel(capacity)

            validRow = isfinite(capacity(k)) && ...
                       isfinite(R0(k)) && isfinite(R1(k)) && isfinite(R2(k)) && ...
                       isfinite(tau1(k)) && isfinite(tau2(k)) && ...
                       R0(k) > 0 && R1(k) > 0 && R2(k) > 0 && ...
                       tau1(k) > 0 && tau2(k) > 0 && ...
                       isfinite(C2(k)) && C2(k) > 0;

            if validRow
                row = row + 1;

                Result_Features.SOC(row,1) = soc;
                Result_Features.Load(row,1) = string(loadName);
                Result_Features.Cell_ID(row,1) = cellNames(k);
                Result_Features.Capacity(row,1) = capacity(k);
                Result_Features.useLog(row,1) = useLog;

                Result_Features.R0(row,1) = R0(k);
                Result_Features.R1(row,1) = R1(k);
                Result_Features.R2(row,1) = R2(k);
                Result_Features.tau1(row,1) = tau1(k);
                Result_Features.tau2(row,1) = tau2(k);
                Result_Features.C2(row,1) = C2(k);

                Result_Features.R0_fresh(row,1) = R0_fresh;
                Result_Features.R1_fresh(row,1) = R1_fresh;
                Result_Features.R2_fresh(row,1) = R2_fresh;
                Result_Features.tau1_fresh(row,1) = tau1_fresh;
                Result_Features.tau2_fresh(row,1) = tau2_fresh;
                Result_Features.C2_fresh(row,1) = C2_fresh;

                Result_Features.ratio_x1_R0_over_R0fresh(row,1) = ratio_x1(k);
                Result_Features.ratio_x2_Rsum_over_Rsumfresh(row,1) = ratio_x2(k);
                Result_Features.ratio_x3_tauRatio_over_tauRatioFresh(row,1) = ratio_x3(k);
                Result_Features.ratio_x4_C2fresh_over_C2(row,1) = ratio_x4(k);
                Result_Features.ratio_x5_R1_over_R1fresh(row,1) = ratio_x5_inner(k);

                Result_Features.x1(row,1) = x1(k);
                Result_Features.x2(row,1) = x2(k);
                Result_Features.x3(row,1) = x3(k);
                Result_Features.x4(row,1) = x4(k);
                Result_Features.x5(row,1) = x5(k);
            end
        end

    end

    %% === subplot labels ==============================================
    xlabel(ax(1), 'Capacity (%)', 'Interpreter', 'none');
    xlabel(ax(2), 'Capacity (%)', 'Interpreter', 'none');
    xlabel(ax(3), 'Capacity (%)', 'Interpreter', 'none');
    xlabel(ax(4), 'Capacity (%)', 'Interpreter', 'none');
    xlabel(ax(5), 'Capacity (%)', 'Interpreter', 'none');

    ylabel(ax(1), '$x_1$', 'Interpreter', 'latex');
    ylabel(ax(2), '$x_2$', 'Interpreter', 'latex');
    ylabel(ax(3), '$x_3$', 'Interpreter', 'latex');
    ylabel(ax(4), '$x_4$', 'Interpreter', 'latex');
    ylabel(ax(5), '$x_5$', 'Interpreter', 'latex');

    if useLog
        title(ax(1), '$x_1=\log(\hat{R}_0/R_{0,\mathrm{fresh}})$', ...
            'Interpreter', 'latex');

        title(ax(2), '$x_2=\log((\hat{R}_1+\hat{R}_2)/(R_{1,\mathrm{fresh}}+R_{2,\mathrm{fresh}}))$', ...
            'Interpreter', 'latex');

        title(ax(3), '$x_3=\log((\hat{\tau}_2/\hat{\tau}_1)/(\tau_{2,\mathrm{fresh}}/\tau_{1,\mathrm{fresh}}))$', ...
            'Interpreter', 'latex');

        title(ax(4), '$x_4=\log(C_{2,\mathrm{fresh}}/\hat{C}_2),\quad C_2=\tau_2/R_2$', ...
            'Interpreter', 'latex');
    else
        title(ax(1), '$x_1=\hat{R}_0/R_{0,\mathrm{fresh}}$', ...
            'Interpreter', 'latex');

        title(ax(2), '$x_2=(\hat{R}_1+\hat{R}_2)/(R_{1,\mathrm{fresh}}+R_{2,\mathrm{fresh}})$', ...
            'Interpreter', 'latex');

        title(ax(3), '$x_3=(\hat{\tau}_2/\hat{\tau}_1)/(\tau_{2,\mathrm{fresh}}/\tau_{1,\mathrm{fresh}})$', ...
            'Interpreter', 'latex');

        title(ax(4), '$x_4=C_{2,\mathrm{fresh}}/\hat{C}_2,\quad C_2=\tau_2/R_2$', ...
            'Interpreter', 'latex');
    end

    title(ax(5), '$x_5=\mathrm{softplus}(\log(\hat{R}_1/R_{1,\mathrm{fresh}}))-\log(2)$', ...
        'Interpreter', 'latex');

    %% === info panel ===================================================
    text(ax(6), 0.02, 0.88, sprintf('Fresh reference cell index: %d', fresh_idx), ...
        'FontSize', 11, 'Interpreter', 'none');

    text(ax(6), 0.02, 0.76, sprintf('Fresh reference capacity: %.2f %%', capacity_fresh), ...
        'FontSize', 11, 'Interpreter', 'none');

    text(ax(6), 0.02, 0.64, sprintf('useLog: %d', useLog), ...
        'FontSize', 11, 'Interpreter', 'none');

    text(ax(6), 0.02, 0.50, 'Feature direction:', ...
        'FontSize', 11, 'FontWeight', 'bold', 'Interpreter', 'none');

    text(ax(6), 0.02, 0.40, '$x_1,x_2,x_3,x_5$: fitted / fresh', ...
        'FontSize', 11, 'Interpreter', 'latex');

    text(ax(6), 0.02, 0.32, '$x_4$: fresh / fitted', ...
        'FontSize', 11, 'Interpreter', 'latex');

    text(ax(6), 0.02, 0.22, '$C_2=\tau_2/R_2$', ...
        'FontSize', 11, 'Interpreter', 'latex');

    text(ax(6), 0.02, 0.12, '$\mathrm{softplus}(u)=\log(1+e^u)$', ...
        'FontSize', 11, 'Interpreter', 'latex');

    legend(ax(1), 'Location', 'bestoutside', 'Interpreter', 'none');

    for ai = 1:5
        if useLog || ai == 5
            yline(ax(ai), 0, '--', 'HandleVisibility', 'off');
        else
            yline(ax(ai), 1, '--', 'HandleVisibility', 'off');
        end
        set(ax(ai), 'FontSize', 10);
    end

end

%% === 7) 결과 출력 =====================================================
disp('=== 5개 feature 결과 ===');
disp(Result_Features);

fprintf('\n완료! 저장 없이 figure만 생성했습니다.\n');
fprintf('Fresh 기준 셀 index = %d, capacity = %.2f\n', fresh_idx, capacity_fresh);
fprintf('useLog = %d\n', useLog);

%% =====================================================================
%  Local functions
% =====================================================================

function y = softplus_local(u)
    % Numerically stable softplus:
    % softplus(u) = log(1 + exp(u))
    y = log1p(exp(-abs(u))) + max(u, 0);
end

function plot_feature_line(axh, capacity, yraw, colorVal, dispName)

    valid = isfinite(capacity) & isfinite(yraw);

    if nnz(valid) < 2
        return;
    end

    x = capacity(valid);
    y = yraw(valid);

    [xSorted, idx] = sort(x, 'ascend');
    ySorted = y(idx);

    plot(axh, xSorted, ySorted, '-o', ...
        'LineWidth', 1.6, ...
        'MarkerSize', 5.5, ...
        'Color', colorVal, ...
        'MarkerFaceColor', colorVal, ...
        'DisplayName', dispName);
end