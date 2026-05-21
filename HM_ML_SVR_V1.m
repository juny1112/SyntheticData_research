%% ======================================================================
%  2RC(선택 부하: Tbl_<LOAD>_ECM) + (label 여러개) → SVR (Support Vector Regression)
%
%  - 입력:
%       2RC_results_XXX.mat (Tbl_Load_ECM 또는 Tbl_<LOAD>_ECM 포함)
%          * Tbl_<LOAD>_ECM.RowNames : 셀 이름
%          * Tbl_<LOAD>_ECM.VarNames: SOC70_R0_mOhm, SOC70_R1_mOhm, ..., SOC50_tau2
%
%  - LOAD_USE_STR에 "US06 CITY1" 처럼 입력하면 해당 주행부하들만 2RC 피처로 사용
%  - 피처명: <LOAD>_<param>_<SOC>  (예: CITY1_R0_70)
%  - label(타겟)도 pool/active 구조로 여러 개 동시 실행
%
%  - SVR 설정:
%       * 입력 X : 선택 SOC의 (선택 주행부하) ECM 파라미터 (R0,R1,R2,tau1,tau2)
%       * 출력 y : 사용자 label (토글로 선택)
%       * train/test split 없음 -> 전체 데이터로 학습하고 train 내 오차만 계산
%       * Standardize = true
%       * linear SVR
%       * BoxConstraint, Epsilon ratio grid search
%
%  - 출력(save_path):
%       각 타겟별 best 모델, 예측결과, 성능지표, 그림, grid search 결과표
% ======================================================================
clear; clc; close all;

% ======================================================================
% 저장 시 Text Interpreter 에러 방지
% ======================================================================
set(groot,'defaultTextInterpreter','none');
set(groot,'defaultAxesTickLabelInterpreter','none');
set(groot,'defaultLegendInterpreter','none');

%% ── 설정(토글) ---------------------------------------------------------
SOC_use = [70];
LOAD_USE_STR = "US06 CITY1";
TEMP_list = [20];

% ── 타겟(출력 label) 활성화 토글(기본 스칼라) ---------------------------
PRED_QC2         = true;
PRED_QC40        = true;
PRED_RCHARG      = true;
PRED_RCHARG_8090 = false;
PRED_R1S         = false;

% ── 타겟(출력 label) 활성화 토글(온도별) -------------------------------
PRED_DCIR1S_BYTEMP   = false;
PRED_DCIR10S_BYTEMP  = true;
PRED_DDELTA_BYTEMP   = false;
PRED_POWER_BYTEMP    = true;

% ── SVR 기본 설정 -------------------------------------------------------
SVR_KERNEL      = 'linear';
SVR_STANDARDIZE = true;

% ======================================================================
% GRID SEARCH 설정
% ======================================================================
BOX_GRID       = [1e-2, 1e-1, 1, 10, 100];
EPS_RATIO_GRID = [0.01, 0.03, 0.05, 0.10];

% tie-break 기준:
%   1) RMSE 최소
%   2) 같으면 R2 최대
%   3) 같으면 더 작은 BoxConstraint
%   4) 같으면 더 작은 epsilon ratio
GRID_TIE_TOL = 1e-12;

%% ── 경로/파일 ----------------------------------------------------------
matPath   = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\20degC\2RC_fitting_600s\2RC_results_600s.mat";
save_path = 'G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\ML_SVR';
if ~exist(save_path, 'dir'), mkdir(save_path); end

%% ── 주행부하 파싱/검증 --------------------------------------------------
loadNames_all = {'US06','UDDS','HWFET','WLTP','CITY1','CITY2','HW1','HW2'};

load_use = parseLoadList(LOAD_USE_STR);
load_use = upper(load_use);
all_upper = upper(string(loadNames_all));

ok = ismember(string(load_use), all_upper);
if any(~ok)
    warning('알 수 없는 주행부하 입력이 있어 제외합니다: %s', strjoin(load_use(~ok), ', '));
end
load_use = load_use(ok);

load_use_std = cell(1,numel(load_use));
for i = 1:numel(load_use)
    idx = find(all_upper == string(load_use(i)), 1, 'first');
    load_use_std{i} = loadNames_all{idx};
end
load_use = load_use_std;

if isempty(load_use)
    error('선택된 주행부하가 없습니다. LOAD_USE_STR="%s" 를 확인하세요.', string(LOAD_USE_STR));
end
fprintf('>> 선택된 주행부하: %s\n', strjoin(load_use, ', '));

%% ── 2RC 테이블 로드 ----------------------------------------------------
S = load(matPath);
getTblECM = @(loadName) localGetTblECM(S, loadName);

load_use_ok = {};
for i = 1:numel(load_use)
    try
        Ttmp = getTblECM(load_use{i});
        if istable(Ttmp)
            load_use_ok{end+1} = load_use{i}; %#ok<AGROW>
        end
    catch
        warning('mat 파일에 %s ECM 테이블이 없어 제외합니다.', load_use{i});
    end
end
load_use = load_use_ok;

if isempty(load_use)
    error('선택한 부하들에 대해 ECM 테이블을 찾지 못했습니다. matPath / 저장 변수를 확인하세요.');
end

%% ── 셀 이름 정합(부하들 간 공통 셀만) -----------------------------------
cell_sets = cell(numel(load_use),1);
for i = 1:numel(load_use)
    T = getTblECM(load_use{i});
    cell_sets{i} = T.Properties.RowNames;
end

cell_names = cell_sets{1};
for i = 2:numel(cell_sets)
    cell_names = intersect(cell_names, cell_sets{i}, 'stable');
end
if isempty(cell_names)
    error('선택한 부하 테이블들 간 RowNames(셀 이름) 교집합이 비었습니다.');
end
nC = numel(cell_names);
fprintf('>> 공통 셀 개수: %d\n', nC);

%% ── 2RC 피처 구성 ------------------------------------------------------
pNames_2RC = {'R0','R1','R2','tau1','tau2'};

feat2RC_names = {};
for li = 1:numel(load_use)
    L = load_use{li};
    for s = SOC_use(:).'
        for pi = 1:numel(pNames_2RC)
            feat2RC_names{end+1} = sprintf('%s_%s_%d', L, pNames_2RC{pi}, s); %#ok<AGROW>
        end
    end
end
nFeat = numel(feat2RC_names);

X = nan(nC, nFeat);

col = 0;
for li = 1:numel(load_use)
    L = load_use{li};
    TblL = getTblECM(L);
    TblL = TblL(cell_names, :);
    vnames = TblL.Properties.VariableNames;

    soc_list = [];
    for ii = 1:numel(vnames)
        tok = regexp(vnames{ii}, '^SOC(\d+)_', 'tokens', 'once');
        if ~isempty(tok)
            soc_list(end+1) = str2double(tok{1}); %#ok<AGROW>
        end
    end
    SOC_inTbl = unique(soc_list, 'stable');

    for s = SOC_use(:).'
        if ~ismember(s, SOC_inTbl)
            warning('[%s] Tbl_%s_ECM 에 SOC%d 변수가 없어 NaN으로 남습니다.', L, L, s);
        end

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

fprintf('\n[SVR] 사용할 입력 피처(%d개):\n', nFeat);
disp(strjoin(feat2RC_names, ', '));

%% ── 타겟(label) 입력 ----------------------------------------------------
fprintf('\n셀 순서(%d개):\n', nC);
disp(strjoin(cell_names, ', '));

% ======================================================================
% [여기부터 입력 구간]
% ======================================================================
QC2_user      = [56.14;55.93;52.55;50.52;52.13;48.53;56.34;55.15;39.89;55.63;55.57;56.86];
QC40_user     = [57.49;57.57;54;52.22;53.45;51.28;57.91;56.51;42.14;57.27;57.18;58.4];
Rcharg_user   = [2.17
1.90
3.50
2.82
2.88
3.38
2.10
1.93
6.41
2.00
2.01
2.09];
Rcharg_80_90_avg_user = nan(nC,1);
R1s_user      = nan(nC,1);

DCIR1s_T20_user    = nan(nC,1);
DCIR10s_T20_user   = [1.60
1.41
2.33
1.88
2.06
1.93
1.35
1.53
3.55
0.82
1.52
1.47];
DCIRdelta_T20_user = nan(nC,1);
Power_T20_user     = [2089.79
2372.03
1427.37
1735.16
1603.14
1677.27
2476.97
2191.48
914.67
4067.20
2196.09
2278.82];
Power_T20_user = Power_T20_user / 1000; % W -> kW
% ======================================================================
% [입력 구간 끝]
% ======================================================================

% 길이 보정(기본 스칼라)
QC2_user               = ensureLength(QC2_user,               nC);
QC40_user              = ensureLength(QC40_user,              nC);
Rcharg_user            = ensureLength(Rcharg_user,            nC);
Rcharg_80_90_avg_user  = ensureLength(Rcharg_80_90_avg_user,  nC);
R1s_user               = ensureLength(R1s_user,               nC);

% 길이 보정(온도별)
DCIR1s_T20_user     = ensureLength(DCIR1s_T20_user,     nC);
DCIR10s_T20_user    = ensureLength(DCIR10s_T20_user,    nC);
DCIRdelta_T20_user  = ensureLength(DCIRdelta_T20_user,  nC);
Power_T20_user      = ensureLength(Power_T20_user,      nC);

%% ── 온도별 데이터 컨테이너 구성 ----------------------------------------
DCIR1s_byT  = struct('T20',DCIR1s_T20_user);
DCIR10s_byT = struct('T20',DCIR10s_T20_user);
DCIRd_byT   = struct('T20',DCIRdelta_T20_user);
Power_byT   = struct('T20',Power_T20_user);

%% ── 타겟 이름 / 벡터 매핑(pool) ----------------------------------------
target_values = containers.Map();
target_values('QC2')              = QC2_user(:);
target_values('QC40')             = QC40_user(:);
target_values('Rcharg')           = Rcharg_user(:);
target_values('Rcharg_80_90_avg') = Rcharg_80_90_avg_user(:);
target_values('R1s')              = R1s_user(:);

for t = TEMP_list
    keyT = sprintf('T%d', t);

    d1  = DCIR1s_byT.(keyT)(:);
    d10 = DCIR10s_byT.(keyT)(:);
    dd  = DCIRd_byT.(keyT)(:);
    pw  = Power_byT.(keyT)(:);

    dDCIR = nan(nC,1);
    for i = 1:nC
        if isfinite(dd(i))
            dDCIR(i) = dd(i);
        elseif isfinite(d10(i)) && isfinite(d1(i))
            dDCIR(i) = d10(i) - d1(i);
        else
            dDCIR(i) = NaN;
        end
    end

    target_values(sprintf('DCIR_1s_T%d', t))           = d1;
    target_values(sprintf('DCIR_10s_T%d', t))          = d10;
    target_values(sprintf('DCIR_delta_10s_1s_T%d', t)) = dDCIR;
    target_values(sprintf('Power_T%d', t))             = pw;
end

%% ── active target list -------------------------------------------------
target_active = {};
if PRED_QC2,          target_active{end+1} = 'QC2';              end
if PRED_QC40,         target_active{end+1} = 'QC40';             end
if PRED_RCHARG,       target_active{end+1} = 'Rcharg';           end
if PRED_RCHARG_8090,  target_active{end+1} = 'Rcharg_80_90_avg'; end
if PRED_R1S,          target_active{end+1} = 'R1s';              end

for t = TEMP_list
    if PRED_DCIR1S_BYTEMP,  target_active{end+1} = sprintf('DCIR_1s_T%d', t); end %#ok<AGROW>
    if PRED_DCIR10S_BYTEMP, target_active{end+1} = sprintf('DCIR_10s_T%d', t); end %#ok<AGROW>
    if PRED_DDELTA_BYTEMP,  target_active{end+1} = sprintf('DCIR_delta_10s_1s_T%d', t); end %#ok<AGROW>
    if PRED_POWER_BYTEMP,   target_active{end+1} = sprintf('Power_T%d', t); end %#ok<AGROW>
end

if isempty(target_active)
    error('활성화된 타겟이 없습니다. PRED_XXX 토글을 확인하세요.');
end

fprintf('\n[SVR] 활성 타겟:\n');
disp(strjoin(target_active, ', '));

%% ── 전체 X / 각 타겟에 대한 SVR(train-only + grid search) -------------
results = struct();
nFeatX  = size(X,2);

for tt = 1:numel(target_active)
    tname = target_active{tt};
    y     = target_values(tname);

    idx_valid = base_valid_X & isfinite(y);
    n_valid   = nnz(idx_valid);

    fprintf('\n============================================================\n');
    fprintf('[SVR] Target = %s, 유효 샘플 수 = %d / %d\n', tname, n_valid, nC);

    if n_valid < nFeatX + 1
        warning('타겟 %s: 유효 샘플이 너무 적어 (n_valid=%d < nFeat+1=%d) 학습 불가. 건너뜁니다.', ...
            tname, n_valid, nFeatX+1);
        continue;
    end

    Xv = X(idx_valid, :);
    yv = y(idx_valid);

    yrng = max(yv) - min(yv);
    if ~isfinite(yrng) || yrng <= 0
        yrng = 1;
    end

    % --------------------------------------------------------------
    % Grid search
    % --------------------------------------------------------------
    gridRows = [];
    rowCnt = 0;

    best_mdl       = [];
    best_yhat      = [];
    best_res       = [];
    best_RMSE      = inf;
    best_MAE       = inf;
    best_R2        = -inf;
    best_Box       = NaN;
    best_EpsRatio  = NaN;
    best_Epsilon   = NaN;

    for ib = 1:numel(BOX_GRID)
        boxC = BOX_GRID(ib);

        for ie = 1:numel(EPS_RATIO_GRID)
            eps_ratio = EPS_RATIO_GRID(ie);
            eps_t     = eps_ratio * yrng;

            if ~isfinite(eps_t) || eps_t <= 0
                eps_t = 0.1;
            end

            mdl_try = fitrsvm(Xv, yv, ...
                'KernelFunction', SVR_KERNEL, ...
                'Standardize',   SVR_STANDARDIZE, ...
                'Epsilon',       eps_t, ...
                'BoxConstraint', boxC);

            y_hat_try = predict(mdl_try, Xv);
            res_try   = yv - y_hat_try;

            RMSE_try = sqrt(mean(res_try.^2));
            MAE_try  = mean(abs(res_try));

            SSres = sum((yv - y_hat_try).^2);
            SStot = sum((yv - mean(yv)).^2);
            if SStot <= eps
                R2_try = NaN;
            else
                R2_try = 1 - SSres / SStot;
            end

            rowCnt = rowCnt + 1;
            gridRows(rowCnt,1) = boxC;      %#ok<SAGROW>
            gridRows(rowCnt,2) = eps_ratio; %#ok<SAGROW>
            gridRows(rowCnt,3) = eps_t;     %#ok<SAGROW>
            gridRows(rowCnt,4) = RMSE_try;  %#ok<SAGROW>
            gridRows(rowCnt,5) = MAE_try;   %#ok<SAGROW>
            gridRows(rowCnt,6) = R2_try;    %#ok<SAGROW>

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
                best_res      = res_try;
                best_RMSE     = RMSE_try;
                best_MAE      = MAE_try;
                best_R2       = R2_try;
                best_Box      = boxC;
                best_EpsRatio = eps_ratio;
                best_Epsilon  = eps_t;
            end
        end
    end

    gridTbl = array2table(gridRows, ...
        'VariableNames', {'BoxConstraint','EpsilonRatio','Epsilon','RMSE','MAE','R2'});
    gridTbl = sortrows(gridTbl, {'RMSE','MAE','R2'}, {'ascend','ascend','descend'});

    fprintf('[Best] %s\n', tname);
    fprintf('  Kernel       = %s\n', SVR_KERNEL);
    fprintf('  BoxConstraint= %.6g\n', best_Box);
    fprintf('  EpsRatio     = %.6g\n', best_EpsRatio);
    fprintf('  Epsilon      = %.6g\n', best_Epsilon);
    fprintf('  R^2          = %.4f\n', best_R2);
    fprintf('  RMSE         = %.4f\n', best_RMSE);
    fprintf('  MAE          = %.4f\n', best_MAE);

    results.(tname).mdl             = best_mdl;
    results.(tname).X               = Xv;
    results.(tname).y_true          = yv;
    results.(tname).y_pred          = best_yhat;
    results.(tname).residuals       = best_res;
    results.(tname).R2              = best_R2;
    results.(tname).RMSE            = best_RMSE;
    results.(tname).MAE             = best_MAE;
    results.(tname).idx_valid       = idx_valid;
    results.(tname).feature_names   = feat2RC_names;
    results.(tname).SOC_use         = SOC_use;
    results.(tname).LOAD_USE_STR    = LOAD_USE_STR;
    results.(tname).svr_kernel      = SVR_KERNEL;
    results.(tname).best_box        = best_Box;
    results.(tname).best_eps_ratio  = best_EpsRatio;
    results.(tname).best_epsilon    = best_Epsilon;
    results.(tname).grid_table      = gridTbl;

    % ---- grid search 결과 csv 저장 ------------------------------------
    writetable(gridTbl, fullfile(save_path, sprintf('SVR_%s_gridsearch_results.csv', tname)));

    % ---- 예측 vs 실제 scatter plot -----------------------------------
    fig1 = figure('Color','w','Name',sprintf('SVR_%s_true_vs_pred',tname));
    scatter(yv, best_yhat, 40, 'k', 'filled'); hold on; grid on;
    minv = min([yv; best_yhat]); maxv = max([yv; best_yhat]);
    plot([minv maxv], [minv maxv], 'r--', 'LineWidth', 1.5);

    xlabel(sprintf('True %s', tname), 'FontSize', 11, 'FontWeight','bold', 'Interpreter','none');
    ylabel(sprintf('Predicted %s', tname), 'FontSize', 11, 'FontWeight','bold', 'Interpreter','none');
    title(sprintf('SVR(%s): %s | Loads: %s | R^2=%.3f, RMSE=%.3f', ...
        SVR_KERNEL, tname, strjoin(load_use, ', '), best_R2, best_RMSE), ...
        'FontSize', 10, 'FontWeight','bold', 'Interpreter','none');

    axis equal; axis tight;

    outfig1 = fullfile(save_path, sprintf('SVR_%s_true_vs_pred.fig', tname));
    outpng1 = fullfile(save_path, sprintf('SVR_%s_true_vs_pred.png', tname));
    savefig(fig1, outfig1);
    exportgraphics(fig1, outpng1, 'Resolution', 220);

    % ---- residual plot ------------------------------------------------
    fig2 = figure('Color','w','Name',sprintf('SVR_%s_residuals',tname));

    subplot(2,1,1);
    plot(best_res, 'o-','LineWidth',1.2);
    grid on;
    xlabel('Sample index (valid rows)', 'Interpreter','none');
    ylabel('Residual', 'Interpreter','none');
    title(sprintf('Residuals (y - yhat) for %s', tname), ...
        'FontWeight','bold', 'Interpreter','none');

    subplot(2,1,2);
    histogram(best_res, 'NumBins', max(5, round(sqrt(numel(best_res)))));
    grid on;
    xlabel('Residual', 'Interpreter','none');
    ylabel('Count', 'Interpreter','none');
    title('Residual histogram','FontWeight','bold', 'Interpreter','none');

    outfig2 = fullfile(save_path, sprintf('SVR_%s_residuals.fig', tname));
    outpng2 = fullfile(save_path, sprintf('SVR_%s_residuals.png', tname));
    savefig(fig2, outfig2);
    exportgraphics(fig2, outpng2, 'Resolution', 220);
end

%% ── 전체 결과 저장 ------------------------------------------------------
save(fullfile(save_path, 'SVR_results_all.mat'), ...
    'results', 'X', 'feat2RC_names', 'cell_names', ...
    'SOC_use', 'pNames_2RC', 'LOAD_USE_STR', 'TEMP_list', ...
    'SVR_KERNEL', 'SVR_STANDARDIZE', ...
    'BOX_GRID', 'EPS_RATIO_GRID');

disp('모든 SVR 결과 저장 완료.');

%% ========================= 보조 함수들 =================================
function loads = parseLoadList(str0)
    % "US06 CITY1", "US06,CITY1", "US06; CITY1" 모두 허용
    if isstring(str0), str0 = char(str0); end
    str0 = strtrim(str0);
    if isempty(str0)
        loads = {};
        return
    end
    parts = regexp(str0, '[,;\s]+', 'split');
    parts = parts(~cellfun(@isempty,parts));
    loads = parts(:).';
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
    % 1) Tbl_Load_ECM.(loadName)
    if isfield(S,'Tbl_Load_ECM') && isstruct(S.Tbl_Load_ECM) && isfield(S.Tbl_Load_ECM, loadName)
        T = S.Tbl_Load_ECM.(loadName);
        return
    end
    % 2) Tbl_<LOAD>_ECM
    varName = sprintf('Tbl_%s_ECM', loadName);
    if isfield(S, varName)
        T = S.(varName);
        return
    end
    error('ECM table not found for load=%s', loadName);
end