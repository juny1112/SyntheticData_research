clear; clc; close all;

%% =====================================================================
%  FULL-WIDTH 3-PANEL FIGURE FOR ABSTRACT
%   (a) Driving-load fitting result
%   (b) Correlation heatmap
%   (c) SVR prediction result
%
%  - panel title 없음
%  - panel label은 axes 바깥에 annotation으로 표시
%  - (a)는 원래 코드처럼 "해당 segment 하나"에 대해 fitting
% ======================================================================

%% ===== USER SETTINGS ===================================================
baseDir_2RC = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed";
simFolder   = 'G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\20degC\이름정렬';
save_path   = 'G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\FIG_abstract';
if ~exist(save_path,'dir'), mkdir(save_path); end

% ---------- panel (a): fitting ----------
PANEL_A_TEMP      = 20;
PANEL_A_SOC_LIST  = [70 50];
PANEL_A_SOC       = 70;
PANEL_A_LOAD      = 'US06';
PANEL_A_SIM_FILE  = "";      % 예: "x01_SIM.mat" / ""이면 첫 번째 파일
fit_window_sec    = 600;

% ---------- panel (b): heatmap ----------
PANEL_B_TEMP      = 20;
PANEL_B_SOC       = 70;
PANEL_B_LOAD      = 'US06';

% ---------- panel (c): SVR prediction ----------
PANEL_C_TEMP      = 20;
PANEL_C_SOC       = 70;
PANEL_C_LOADS     = {'US06','CITY1'};
PANEL_C_TARGET    = 'QC2';   % 'QC2','QC40','DCIR10s','Rcharge','Power'
SVR_KERNEL        = 'linear';
SVR_STANDARDIZE   = true;
BOX_GRID          = [1e-2, 1e-1, 1, 10, 100];
EPS_RATIO_GRID    = [0.01, 0.03, 0.05, 0.10];
GRID_TIE_TOL      = 1e-12;

out_png = fullfile(save_path, 'abstract_fullwidth_3panel_fixed.png');
out_fig = fullfile(save_path, 'abstract_fullwidth_3panel_fixed.fig');

%% ===== REFERENCE PALETTE ==============================================
C.blue      = hex2rgb('#0073C2');
C.yellow    = hex2rgb('#EFC000');
C.red       = hex2rgb('#CD534C');
C.green     = hex2rgb('#20854E');
C.purple    = hex2rgb('#925E9F');
C.orange    = hex2rgb('#E18727');
C.lightblue = hex2rgb('#4DBBD5');
C.pink      = hex2rgb('#EE4C97');
C.brown     = hex2rgb('#7E6148');
C.gray      = hex2rgb('#747678');

heat_cmap = makeBlueWhiteRed(C.blue, C.red, 256);

%% ===== LABEL DATA ======================================================
QC2_user         = [56.14;55.93;52.55;50.52;52.13;48.53;56.34;55.15;39.89;55.63;55.57;56.86];
QC40_user        = [57.49;57.57;54;52.22;53.45;51.28;57.91;56.51;42.14;57.27;57.18;58.4];
Rcharg_user      = [2.17;1.90;3.50;2.82;2.88;3.38;2.10;1.93;6.41;2.00;2.01;2.09];
DCIR10s_T20_user = [1.60;1.41;2.33;1.88;2.06;1.93;1.35;1.53;3.55;0.82;1.52;1.47];
Power_T20_user   = [2089.79;2372.03;1427.37;1735.16;1603.14;1677.27;2476.97;2191.48;914.67;4067.20;2196.09;2278.82] / 1000;

%% ===== FIGURE LAYOUT ===================================================
f = figure('Color','w','Position',[40 100 1850 560]);
tl = tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

%% ======================================================================
%  PANEL (a): DRIVING-LOAD FITTING  (exact segment only)
% =======================================================================
ax1 = nexttile(tl,1); hold(ax1,'on');

sim_files = dir(fullfile(simFolder,"*_SIM.mat"));
if isempty(sim_files)
    error('SIM 파일을 찾지 못했습니다: %s', simFolder);
end

if strlength(PANEL_A_SIM_FILE) > 0
    idx_file = find(strcmp({sim_files.name}, PANEL_A_SIM_FILE), 1, 'first');
    if isempty(idx_file)
        error('지정한 SIM 파일을 찾지 못했습니다: %s', PANEL_A_SIM_FILE);
    end
else
    idx_file = 1;
end

Ssim = load(fullfile(simFolder, sim_files(idx_file).name), "SIM_table");
SIM_table = Ssim.SIM_table;
nSeg = height(SIM_table);

SOC_list = PANEL_A_SOC_LIST;
nSOC = numel(SOC_list);
loadNames = {'US06','UDDS','HWFET','WLTP','CITY1','CITY2','HW1','HW2'};
blkSize   = 8;
nLoads    = numel(loadNames);
use_block_mapping = true;

grp_code = zeros(nSeg,1);

% 원래 코드 방식
if use_block_mapping && (nSeg >= blkSize*nSOC)
    for g = 1:nSOC
        ii = ((g-1)*blkSize + 1) : min(g*blkSize, nSeg);
        grp_code(ii) = g;
    end
end

if any(grp_code==0)
    SOC_center = mean([SIM_table.SOC1 SIM_table.SOC2], 2, 'omitnan');

    miss = isnan(SOC_center);
    if any(miss) && ismember('SOC_vec', SIM_table.Properties.VariableNames)
        try
            SOC_center(miss) = cellfun(@(v) mean(v,'omitnan'), SIM_table.SOC_vec(miss));
        catch
        end
    end

    valid = ~isnan(SOC_center);
    if any(valid)
        [~, gmin] = min(abs(SOC_center(valid) - SOC_list), [], 2);
        vidx = find(valid);
        grp_code(vidx(grp_code(vidx)==0)) = gmin(grp_code(vidx)==0);
    end
end

load_idx = nan(nSOC, nLoads);
for g = 1:nSOC
    for l = 1:nLoads
        load_idx(g,l) = pickLoadSegIdx(g, l, grp_code, nSeg, use_block_mapping, blkSize, nLoads);
    end
end

gA = find(SOC_list == PANEL_A_SOC, 1, 'first');
lA = find(strcmp(loadNames, PANEL_A_LOAD), 1, 'first');

if isempty(gA) || isempty(lA)
    error('PANEL_A_SOC 또는 PANEL_A_LOAD 설정이 잘못되었습니다.');
end

segA = load_idx(gA, lA);
if isnan(segA)
    error('선택한 SOC / LOAD 조합에 해당하는 segment를 찾지 못했습니다.');
end

t = SIM_table.time{segA};
I = SIM_table.current{segA};
V = SIM_table.voltage{segA};
O = [];
if ismember('OCV_vec', SIM_table.Properties.VariableNames)
    O = SIM_table.OCV_vec{segA};
end

if isduration(t)
    t = seconds(t - t(1));
else
    t = t - t(1);
end

[t2, I2, V2, O2, okCrop] = cropToWindow(t, I, V, O, fit_window_sec);
if ~okCrop
    error('패널 (a): 600s crop 이후 데이터가 부족합니다.');
end

ms = MultiStart("UseParallel", true, "Display", "off");
nStart = 40;
opt = optimset('display','off','MaxIter',1e3,'MaxFunEvals',1e4,'TolFun',eps,'TolX',eps);

para0 = [0.003 0.0005 0.0005 10 100];
lb    = [0     0      0      0.1 0.1];
ub    = [0.05  0.005  0.03   100 2000];
A_lin = [0 0 0 1 -1];
b_lin = 0;

X0 = makeStartPoints_LogTau(nStart, lb, ub);
X0 = [para0; X0];
startPts = CustomStartPointSet(X0);

problem = createOptimProblem('fmincon', ...
    'objective', @(p) RMSE_2RC(V2,p,t2,I2,O2), ...
    'x0', para0, ...
    'lb', lb, ...
    'ub', ub, ...
    'Aineq', A_lin, ...
    'bineq', b_lin, ...
    'options', opt);

[Pbest, ~] = run(ms, problem, startPts);
V_fit = RC_model_2(Pbest, t2, I2, O2);

plot(ax1, t2, V2,    'Color', C.gray, 'LineWidth', 1.6);
plot(ax1, t2, V_fit, 'Color', C.red,  'LineWidth', 1.8);

xlabel(ax1, 'Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel(ax1, 'Voltage (V)', 'FontSize', 12, 'FontWeight', 'bold');
grid(ax1, 'on');
box(ax1, 'on');
set(ax1, 'FontSize', 11, 'LineWidth', 1.0);
legend(ax1, {'True','Fitted'}, 'Location', 'southwest', 'Box', 'off', 'FontSize', 10);

%% ======================================================================
%  PANEL (b): CORRELATION HEATMAP
% =======================================================================
ax2 = nexttile(tl,2);

matPath_B = fullfile(baseDir_2RC, sprintf('%ddegC', PANEL_B_TEMP), '2RC_fitting_600s', '2RC_results_600s.mat');
if ~isfile(matPath_B)
    error('패널 (b): 2RC mat 파일이 없습니다: %s', matPath_B);
end
Sh = load(matPath_B);
TblH = localGetTblECM(Sh, PANEL_B_LOAD);

raw_names = TblH.Properties.RowNames;
ids_raw = extractCellIDs(raw_names);
[ids_unique, idx_keep] = uniqueCellIDsFirst(ids_raw);
TblH = TblH(idx_keep, :);
nC_h = numel(ids_unique);

QC2_h    = ensureLength(QC2_user, nC_h);
QC40_h   = ensureLength(QC40_user, nC_h);
Rcharg_h = ensureLength(Rcharg_user, nC_h);
DCIR_h   = ensureLength(DCIR10s_T20_user, nC_h);

pNames = {'R0','R1','R2','tau1','tau2'};
X_ecm = nan(nC_h, numel(pNames));
for pi = 1:numel(pNames)
    pname = pNames{pi};
    if pi <= 3
        varName = sprintf('SOC%d_%s_mOhm', PANEL_B_SOC, pname);
    else
        varName = sprintf('SOC%d_%s', PANEL_B_SOC, pname);
    end
    if ~ismember(varName, TblH.Properties.VariableNames)
        error('패널 (b): 변수 %s 를 찾지 못했습니다.', varName);
    end
    X_ecm(:,pi) = TblH{:,varName};
end

Y_lab = [QC2_h, QC40_h, DCIR_h, Rcharg_h];
R_custom = nan(size(Y_lab,2), size(X_ecm,2));
for i = 1:size(Y_lab,2)
    for j = 1:size(X_ecm,2)
        x = X_ecm(:,j);
        y = Y_lab(:,i);
        v = isfinite(x) & isfinite(y);
        if nnz(v) >= 3 && std(x(v)) > 0 && std(y(v)) > 0
            R_custom(i,j) = corr(x(v), y(v), 'Type', 'Pearson');
        end
    end
end

imagesc(ax2, R_custom, [-1 1]);
colormap(ax2, heat_cmap);
axis(ax2, 'tight');

cb = colorbar(ax2);
cb.Label.String = 'Pearson r';
set(cb, 'Ticks', -1:0.5:1, 'TickLabels', compose('%.1f', -1:0.5:1));

ecm_display_names   = {'R_0','R_1','R_2','\tau_1','\tau_2'};
label_display_names = {'Q_{C/2}','Q_{C/40}','DCIR_{10s}','R_{charge}'};

set(ax2, ...
    'XTick', 1:numel(ecm_display_names), ...
    'YTick', 1:numel(label_display_names), ...
    'XTickLabel', ecm_display_names, ...
    'YTickLabel', label_display_names, ...
    'TickLabelInterpreter', 'tex', ...
    'FontSize', 12, ...
    'FontWeight', 'bold', ...
    'LineWidth', 1.0);

xlabel(ax2, 'ECM Parameters', 'FontSize', 12, 'FontWeight', 'bold');
ylabel(ax2, 'SOH/SOH-x Labels', 'FontSize', 12, 'FontWeight', 'bold');
box(ax2, 'on');
hold(ax2,'on');

[nr, nc] = size(R_custom);
for x = 0.5:1:nc+0.5
    plot(ax2, [x x], [0.5 nr+0.5], 'w-', 'LineWidth', 1.0);
end
for y = 0.5:1:nr+0.5
    plot(ax2, [0.5 nc+0.5], [y y], 'w-', 'LineWidth', 1.0);
end

for i = 1:nr
    for j = 1:nc
        r = R_custom(i,j);
        if isnan(r), continue; end
        idx = max(1, min(256, 1 + round((r+1)/2*255)));
        cc = heat_cmap(idx,:);
        yiq = 0.299*cc(1) + 0.587*cc(2) + 0.114*cc(3);
        if yiq < 0.5
            tcol = [1 1 1];
        else
            tcol = [0 0 0];
        end
        text(ax2, j, i, sprintf('%.2f', r), ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontSize', 11, ...
            'FontWeight', 'bold', ...
            'Color', tcol);
    end
end

%% ======================================================================
%  PANEL (c): SVR PREDICTION RESULT
% =======================================================================
ax3 = nexttile(tl,3); hold(ax3,'on');

matPath_C = fullfile(baseDir_2RC, sprintf('%ddegC', PANEL_C_TEMP), '2RC_fitting_600s', '2RC_results_600s.mat');
if ~isfile(matPath_C)
    error('패널 (c): 2RC mat 파일이 없습니다: %s', matPath_C);
end
S = load(matPath_C);

load_use = PANEL_C_LOADS;

cell_sets = cell(numel(load_use),1);
for i = 1:numel(load_use)
    T = localGetTblECM(S, load_use{i});
    cell_sets{i} = T.Properties.RowNames;
end

cell_names = cell_sets{1};
for i = 2:numel(cell_sets)
    cell_names = intersect(cell_names, cell_sets{i}, 'stable');
end
if isempty(cell_names)
    error('패널 (c): 선택한 부하 테이블들 간 RowNames 교집합이 비었습니다.');
end
nC = numel(cell_names);

pNames_2RC = {'R0','R1','R2','tau1','tau2'};
nFeat = numel(load_use) * numel(pNames_2RC);
X = nan(nC, nFeat);

col = 0;
for li = 1:numel(load_use)
    L = load_use{li};
    TblL = localGetTblECM(S, L);
    TblL = TblL(cell_names, :);
    vnames = TblL.Properties.VariableNames;

    for s = PANEL_C_SOC
        for pi = 1:numel(pNames_2RC)
            col = col + 1;
            pname = pNames_2RC{pi};

            if pi <= 3
                varName = sprintf('SOC%d_%s_mOhm', s, pname);
            else
                varName = sprintf('SOC%d_%s', s, pname);
            end

            if ismember(varName, vnames)
                X(:, col) = TblL{:, varName};
            else
                X(:, col) = NaN;
            end
        end
    end
end

base_valid_X = all(isfinite(X), 2);

switch upper(PANEL_C_TARGET)
    case 'QC2'
        y = ensureLength(QC2_user, nC);
        tname = 'QC2';
    case 'QC40'
        y = ensureLength(QC40_user, nC);
        tname = 'QC40';
    case 'DCIR10S'
        y = ensureLength(DCIR10s_T20_user, nC);
        tname = 'DCIR10s';
    case 'RCHARGE'
        y = ensureLength(Rcharg_user, nC);
        tname = 'Rcharge';
    case 'POWER'
        y = ensureLength(Power_T20_user, nC);
        tname = 'Power';
    otherwise
        error('패널 (c): 지원하지 않는 타겟입니다.');
end

idx_valid = base_valid_X & isfinite(y);
n_valid   = nnz(idx_valid);
nFeatX    = size(X,2);

if n_valid < nFeatX + 1
    error('패널 (c): 유효 샘플이 너무 적습니다.');
end

Xv = X(idx_valid, :);
yv = y(idx_valid);

yrng = max(yv) - min(yv);
if ~isfinite(yrng) || yrng <= 0
    yrng = 1;
end

best_mdl      = [];
best_yhat     = [];
best_RMSE     = inf;
best_R2       = -inf;
best_Box      = NaN;
best_EpsRatio = NaN;

for ib = 1:numel(BOX_GRID)
    boxC = BOX_GRID(ib);

    for ie = 1:numel(EPS_RATIO_GRID)
        eps_ratio = EPS_RATIO_GRID(ie);
        eps_t = eps_ratio * yrng;
        if ~isfinite(eps_t) || eps_t <= 0
            eps_t = 0.1;
        end

        mdl_try = fitrsvm(Xv, yv, ...
            'KernelFunction', SVR_KERNEL, ...
            'Standardize', SVR_STANDARDIZE, ...
            'Epsilon', eps_t, ...
            'BoxConstraint', boxC);

        y_hat_try = predict(mdl_try, Xv);

        SSres = sum((yv - y_hat_try).^2);
        SStot = sum((yv - mean(yv)).^2);
        if SStot <= eps
            R2_try = NaN;
        else
            R2_try = 1 - SSres / SStot;
        end
        RMSE_try = sqrt(mean((yv - y_hat_try).^2));

        isBetter = false;
        if RMSE_try < best_RMSE - GRID_TIE_TOL
            isBetter = true;
        elseif abs(RMSE_try - best_RMSE) <= GRID_TIE_TOL
            if (isnan(best_R2) && ~isnan(R2_try)) || (~isnan(R2_try) && R2_try > best_R2 + GRID_TIE_TOL)
                isBetter = true;
            elseif ((isnan(best_R2) && isnan(R2_try)) || abs(R2_try - best_R2) <= GRID_TIE_TOL)
                if boxC < best_Box - GRID_TIE_TOL
                    isBetter = true;
                elseif abs(boxC - best_Box) <= GRID_TIE_TOL
                    if eps_ratio < best_EpsRatio - GRID_TIE_TOL
                        isBetter = true;
                    end
                end
            end
        end

        if isBetter
            best_mdl      = mdl_try;
            best_yhat     = y_hat_try;
            best_RMSE     = RMSE_try;
            best_R2       = R2_try;
            best_Box      = boxC;
            best_EpsRatio = eps_ratio;
        end
    end
end

scatter(ax3, yv, best_yhat, 40, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
minv = min([yv; best_yhat]);
maxv = max([yv; best_yhat]);
plot(ax3, [minv maxv], [minv maxv], '--', 'Color', C.red, 'LineWidth', 1.5);

xlabel(ax3, ['True ' tname], 'FontSize', 12, 'FontWeight', 'bold');
ylabel(ax3, ['Predicted ' tname], 'FontSize', 12, 'FontWeight', 'bold');
grid(ax3, 'on');
box(ax3, 'on');
set(ax3, 'FontSize', 11, 'LineWidth', 1.0);

text(ax3, 0.04, 0.08, sprintf('SVR  |  R^2 = %.3f, RMSE = %.3f', best_R2, best_RMSE), ...
    'Units', 'normalized', 'FontSize', 10.5, 'FontWeight', 'bold', ...
    'BackgroundColor', 'w', 'Margin', 3);

axis(ax3, 'equal');
xlim(ax3, [minv maxv]);
ylim(ax3, [minv maxv]);

%% ===== PANEL LABELS OUTSIDE AXES ======================================
% drawnow;
% 
% addPanelLabelOutside(f, ax1, '(a)');
% addPanelLabelOutside(f, ax2, '(b)');
% addPanelLabelOutside(f, ax3, '(c)');

%% ===== SAVE ============================================================
exportgraphics(f, out_png, 'Resolution', 300);
savefig(f, out_fig);

disp('3-panel full-width figure 저장 완료');
disp(out_png);

%% ======================================================================
%  HELPER FUNCTIONS
% ======================================================================

function addPanelLabelOutside(figHandle, axHandle, labelText)
    oldUnits = axHandle.Units;
    axHandle.Units = 'normalized';
    pos = axHandle.Position;
    axHandle.Units = oldUnits;

    x = max(pos(1) - 0.015, 0.001);
    y = min(pos(2) + pos(4) + 0.005, 0.99);

    annotation(figHandle, 'textbox', [x y 0.04 0.05], ...
        'String', labelText, ...
        'LineStyle', 'none', ...
        'FontWeight', 'bold', ...
        'FontSize', 14, ...
        'Color', 'k', ...
        'FitBoxToText', 'on');
end

function rgb = hex2rgb(hex)
    hex = char(hex);
    if hex(1) == '#'
        hex = hex(2:end);
    end
    rgb = [hex2dec(hex(1:2)), hex2dec(hex(3:4)), hex2dec(hex(5:6))] / 255;
end

function cmap = makeBlueWhiteRed(c1, c2, n)
    if nargin < 3, n = 256; end
    n1 = floor(n/2);
    n2 = n - n1;
    cmap1 = [linspace(c1(1),1,n1)', linspace(c1(2),1,n1)', linspace(c1(3),1,n1)'];
    cmap2 = [linspace(1,c2(1),n2)', linspace(1,c2(2),n2)', linspace(1,c2(3),n2)'];
    cmap = [cmap1; cmap2];
end

function v = ensureLength(v, n)
    v = v(:);
    if numel(v) < n
        v = [v; nan(n - numel(v),1)];
    elseif numel(v) > n
        v = v(1:n);
    end
end

function T = localGetTblECM(S, loadName)
    if isfield(S,'Tbl_Load_ECM') && isstruct(S.Tbl_Load_ECM) && isfield(S.Tbl_Load_ECM, loadName)
        T = S.Tbl_Load_ECM.(loadName);
        return
    end
    varName = sprintf('Tbl_%s_ECM', loadName);
    if isfield(S, varName)
        T = S.(varName);
        return
    end
    error('ECM table not found for load=%s', loadName);
end

function ids = extractCellIDs(names_in)
    names_in = cellstr(string(names_in));
    ids = nan(numel(names_in),1);
    for k = 1:numel(names_in)
        s = strtrim(names_in{k});
        tokx = regexp(s, '^[xX]\s*0*(\d+)', 'tokens', 'once');
        if ~isempty(tokx)
            ids(k) = str2double(tokx{1});
            continue;
        end
        tok1 = regexp(s, '\d+', 'match', 'once');
        if ~isempty(tok1)
            ids(k) = str2double(tok1);
        end
    end
end

function [ids_unique, idx_keep] = uniqueCellIDsFirst(ids)
    valid = ~isnan(ids);
    ids_valid = ids(valid);
    idx_valid = find(valid);
    [ids_unique, ia] = unique(ids_valid, 'stable');
    idx_keep = idx_valid(ia);
end

function cost = RMSE_2RC(V_true, para, t, I, OCV)
    V_est = RC_model_2(para, t, I, OCV);
    cost  = sqrt(mean((V_true - V_est).^2, 'omitnan'));
end

function [t2, I2, V2, O2, ok] = cropToWindow(t, I, V, O, winSec)
    ok = true;
    if isempty(t) || numel(t) < 5
        ok = false; t2=[]; I2=[]; V2=[]; O2=[]; return
    end
    m = (t <= winSec);
    if nnz(m) < 5
        ok = false; t2=[]; I2=[]; V2=[]; O2=[]; return
    end
    t2 = t(m);
    I2 = I(m);
    V2 = V(m);
    if isempty(O)
        O2 = [];
    else
        try
            O2 = O(m);
        catch
            O2 = [];
        end
    end
end

function X0 = makeStartPoints_LogTau(nStart, lb, ub)
    X0 = nan(nStart, 5);
    lbR = lb(1:3);  ubR = ub(1:3);
    lb1 = lb(4);    ub1 = ub(4);
    lb2 = lb(5);    ub2 = ub(5);

    log_lb1 = log10(lb1); log_ub1 = log10(ub1);
    log_ub2 = log10(ub2);

    k = 0;
    maxTries = 10000;
    tries = 0;

    while k < nStart && tries < maxTries
        tries = tries + 1;
        R = lbR + (ubR - lbR).*rand(1,3);
        tau1 = 10^(log_lb1 + (log_ub1 - log_lb1)*rand);

        tau2_lb = max(tau1, lb2) * (1 + 1e-12);
        if tau2_lb >= ub2
            continue
        end
        log_tau2_lb = log10(tau2_lb);
        tau2 = 10^(log_tau2_lb + (log_ub2 - log_tau2_lb)*rand);

        k = k + 1;
        X0(k,:) = [R, tau1, tau2];
    end

    if k < nStart
        X0 = X0(1:k,:);
    end
end

function sIdx = pickLoadSegIdx(g, loadIdx, grp_code, nSeg, use_block_mapping, blkSize, nLoads)
    sIdx = NaN;
    if use_block_mapping && (blkSize >= nLoads) && (nSeg >= blkSize*max(g,1))
        tmp = (g-1)*blkSize + loadIdx;
        if tmp >= 1 && tmp <= nSeg
            sIdx = tmp;
            return
        end
    end
    idx = find(grp_code==g);
    if isempty(idx), return; end
    idx = sort(idx(:));
    if numel(idx) >= loadIdx
        sIdx = idx(loadIdx);
    end
end

function V_est = RC_model_2(X, t_vec, I_vec, OCV)

    R0   = X(1);
    R1   = X(2);
    R2   = X(3);
    tau1 = X(4);
    tau2 = X(5);

    dt = [1; diff(t_vec)];

    N = length(t_vec);
    V_est = zeros(N, 1);

    for k = 1:N
        IR0 = R0 * I_vec(k);

        alpha1 = exp(-dt(k)/tau1);
        alpha2 = exp(-dt(k)/tau2);

        if k == 1
            Vrc1 = 0;
            Vrc2 = 0;
        else
            Vrc1 = Vrc1*alpha1 + R1*(1 - alpha1)*I_vec(k-1);
            Vrc2 = Vrc2*alpha2 + R2*(1 - alpha2)*I_vec(k-1);
        end

        V_est(k) = OCV(k) + IR0 + Vrc1 + Vrc2;
    end
end