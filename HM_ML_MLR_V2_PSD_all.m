%% ======================================================================
%  2RC(부하별 Tbl_<LOAD>_ECM) + PSD features + labels -> MLR (Ridge + zscore)
%  + Exhaustive search over ALL PSD feature subsets
%
%  [FEATURE STRUCTURE]
%   - sample = (cell, load)
%   - X =
%       (A) 해당 load의 2RC 파라미터(5×SOC_use)  -> 항상 포함
%       (B) 해당 load의 PSD feature들             -> exhaustive subset search 대상
%
%  [SEARCH]
%   - ECM feature는 항상 포함
%   - PSD feature 후보들(PSD_FEAT_USE)에 대해 공집합 포함 모든 조합 평가
%   - 각 조합마다 LOAD-level CV score 계산
%   - global best PSD subset 선택
%
%  [CV: LOAD-LEVEL K-FOLD]
%   - train/val 세트에서 "load 단위"로 K-fold
%   - fold validation = 특정 load(들) 전체
%   - fold training   = 나머지 load 전체
%
%  [MODEL]
%   - zscore는 train 기준
%   - Ridge는 fitrlinear(..., Regularization='ridge')
%% ======================================================================
clear; clc; close all;

%% ── 설정(토글) ---------------------------------------------------------
SOC_use = [70];

% 사용할 주행부하 (데이터 풀)
LOAD_USE_STR = "US06 UDDS HWFET WLTP CITY1 CITY2 HW1 HW2";

% LOAD 단위 split
LOAD_TEST_STR     = "US06";
LOAD_TRAINVAL_STR = "";
K_FOLD = 7;

TEMP_list = [20];

% ── 타겟(출력 label) 활성화 토글(기본 스칼라) ---------------------------
PRED_QC2         = true;
PRED_QC40        = true;
PRED_RCHARG      = false;
PRED_RCHARG_8090 = false;
PRED_R1S         = false;

% ── 타겟(출력 label) 활성화 토글(온도별) --------------------------------
PRED_DCIR1S_BYTEMP  = false;
PRED_DCIR10S_BYTEMP = true;
PRED_DDELTA_BYTEMP  = false;
PRED_POWER_BYTEMP   = true;

%% ── PSD feature 설정 ---------------------------------------------------
USE_PSD_FEATURE = true;

psdMatPath = "G:\공유 드라이브\Battery Software Lab\Protocols\2025 현대 NE 고품셀 평가 실험\주행부하_scaledown\PSD\psd_stat_tbl.mat";

% PSD 후보 feature
PSD_FEAT_USE = [ ...
    "E_slow_fast_ratio"
    "f_geo_mean_Hz"
    "f_geo_std_dec"
    "f_fast_geo_mean_Hz"
    "f_slow_geo_mean_Hz"
    ];

PSD_FEAT_TRANSFORM = [ ...
    "log10"
    "log10"
    "none"
    "log10"
    "log10"
    ];

if numel(PSD_FEAT_USE) ~= numel(PSD_FEAT_TRANSFORM)
    error("PSD_FEAT_USE와 PSD_FEAT_TRANSFORM 길이가 같아야 합니다.");
end

%% ── zscore + Ridge 설정 ------------------------------------------------
DO_ZSCORE = true;
RIDGE_LAMBDA_MODE = "fixed";     % "auto" | "fixed"
RIDGE_LAMBDA_FIXED = 0;
RIDGE_LAMBDA_GRID = logspace(-6, 2, 500);

LAMBDA_SCORE_METRIC = "RMSE";    % "RMSE" | "MAE"

%% ── Exhaustive subset search 설정 --------------------------------------
SEARCH_SCORE_METRIC = "RMSE";    % "RMSE" | "MAE"
SAVE_ALL_COMBO_TABLE = true;     % 모든 조합 시트 저장
SAVE_BEST_BY_SIZE_TABLE = true;  % subset size별 best 시트 저장

%% ── 경로/파일 ----------------------------------------------------------
matPath   = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\20degC\2RC_fitting_600s\2RC_results_600s.mat";

save_path = 'G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\ML_MLR';
if ~exist(save_path, 'dir'), mkdir(save_path); end

summary_xlsx_path = fullfile(save_path, 'PSD_subset_results_by_target.xlsx');
if exist(summary_xlsx_path, 'file')
    delete(summary_xlsx_path);
end

%% ── 주행부하 파싱/검증 -------------------------------------------------
loadNames_all = {'US06','UDDS','HWFET','WLTP','CITY1','CITY2','HW1','HW2'};
all_upper = upper(string(loadNames_all));

load_use = upper(parseLoadList(LOAD_USE_STR));
ok = ismember(string(load_use), all_upper);
if any(~ok)
    warning('알 수 없는 주행부하 입력이 있어 제외: %s', strjoin(load_use(~ok), ', '));
end
load_use = load_use(ok);
load_use = toStdCase(load_use, loadNames_all);

if isempty(load_use), error('LOAD_USE_STR에서 유효 부하가 없습니다.'); end
fprintf('>> LOAD_USE: %s\n', strjoin(load_use, ', '));

load_test = upper(parseLoadList(LOAD_TEST_STR));
load_test = load_test(ismember(string(load_test), all_upper));
load_test = toStdCase(load_test, loadNames_all);
fprintf('>> LOAD_TEST: %s\n', strjoin(load_test, ', '));

load_trainval_user = upper(parseLoadList(LOAD_TRAINVAL_STR));
load_trainval_user = load_trainval_user(ismember(string(load_trainval_user), all_upper));
load_trainval_user = toStdCase(load_trainval_user, loadNames_all);

if isempty(load_trainval_user)
    load_trainval = setdiff(load_use, load_test, 'stable');
else
    load_trainval = intersect(load_use, load_trainval_user, 'stable');
    load_trainval = setdiff(load_trainval, load_test, 'stable');
end

if isempty(load_trainval), error('train/val load가 비었습니다. LOAD_TEST_STR/LOAD_TRAINVAL_STR 확인'); end
fprintf('>> LOAD_TRAINVAL: %s\n', strjoin(load_trainval, ', '));

%% ── 2RC 테이블 로드 ----------------------------------------------------
S = load(matPath);
getTblECM = @(loadName) localGetTblECM(S, loadName);

load_use_ok = {};
for i = 1:numel(load_use)
    try
        Ttmp = getTblECM(load_use{i});
        if istable(Ttmp), load_use_ok{end+1} = load_use{i}; end %#ok<AGROW>
    catch
        warning('mat 파일에 %s ECM 테이블이 없어 제외합니다.', load_use{i});
    end
end
load_use = load_use_ok;

load_trainval = intersect(load_trainval, load_use, 'stable');
load_test     = intersect(load_test,     load_use, 'stable');

if isempty(load_use), error('선택한 load 중 mat에 존재하는 테이블이 없음'); end
if isempty(load_trainval), error('train/val load이 mat에 존재하지 않음'); end
if isempty(load_test), warning('test load이 비었습니다. (test 평가 스킵)'); end

fprintf('>> (exists) LOAD_USE     : %s\n', strjoin(load_use, ', '));
fprintf('>> (exists) LOAD_TRAINVAL: %s\n', strjoin(load_trainval, ', '));
fprintf('>> (exists) LOAD_TEST    : %s\n', strjoin(load_test, ', '));

%% ── PSD table 로드 & load 매핑 -----------------------------------------
psd_byLoad = struct();
if USE_PSD_FEATURE
    fprintf(">> PSD matPath used: %s\n", psdMatPath);

    Sp = load(psdMatPath, "psd_stat_tbl");
    if ~isfield(Sp, "psd_stat_tbl")
        error("psd_stat_tbl이 mat에 없습니다: %s", psdMatPath);
    end
    psd_tbl = Sp.psd_stat_tbl;

    missPSD = ~ismember(PSD_FEAT_USE, string(psd_tbl.Properties.VariableNames));
    if any(missPSD)
        error("psd_stat_tbl에 없는 PSD feature: %s", strjoin(PSD_FEAT_USE(missPSD), ", "));
    end

    psd_load = upper(string(psd_tbl.Load));
    psd_load = erase(psd_load, '"');
    psd_load = erase(psd_load, '''');
    psd_load = strtrim(psd_load);

    fprintf(">> PSD unique loads (clean): %s\n", strjoin(unique(psd_load), ", "));

    for li = 1:numel(loadNames_all)
        L = upper(string(loadNames_all{li}));
        m = (psd_load == L);

        if nnz(m) == 0
            warning("PSD table에 %s 행이 없습니다. 해당 load의 PSD feature는 NaN 처리됩니다.", L);
            continue;
        end
        if nnz(m) > 1
            warning("PSD table에 %s 행이 %d개입니다. 첫 행만 사용합니다.", L, nnz(m));
        end
        psd_byLoad.(char(L)) = psd_tbl(find(m,1,'first'), :);
    end
end

%% ── 셀 이름 정합(LOAD_USE 내 공통 셀) -----------------------------------
cell_sets = cell(numel(load_use),1);
for i = 1:numel(load_use)
    T = getTblECM(load_use{i});
    cell_sets{i} = T.Properties.RowNames;
end
cell_names = cell_sets{1};
for i = 2:numel(cell_sets)
    cell_names = intersect(cell_names, cell_sets{i}, 'stable');
end
if isempty(cell_names), error('LOAD_USE 간 공통 셀이 없음'); end
nC = numel(cell_names);
fprintf('>> 공통 셀 개수 (LOAD_USE 기준): %d\n', nC);

%% ── X 구성: sample=(cell,load) -----------------------------------------
pNames_2RC = {'R0','R1','R2','tau1','tau2'};

feat2RC_names = {};
for s = SOC_use(:).'
    for pi = 1:numel(pNames_2RC)
        feat2RC_names{end+1} = sprintf('%s_%d', pNames_2RC{pi}, s); %#ok<AGROW>
    end
end

featPSD_names = {};
if USE_PSD_FEATURE
    for k = 1:numel(PSD_FEAT_USE)
        featPSD_names{end+1} = sprintf('PSD_%s_%s', PSD_FEAT_USE(k), upper(PSD_FEAT_TRANSFORM(k))); %#ok<AGROW>
    end
end

feat_names_all = [feat2RC_names, featPSD_names];
nFeat = numel(feat_names_all);

idx_feat_ecm = 1:numel(feat2RC_names);
idx_feat_psd = numel(feat2RC_names) + (1:numel(featPSD_names));
nPSD = numel(idx_feat_psd);

nL_use = numel(load_use);
nS = nC * nL_use;

X = nan(nS, nFeat);
load_id = strings(nS,1);
cell_id = strings(nS,1);

row = 0;
for li = 1:nL_use
    L = load_use{li};
    TblL = getTblECM(L);
    TblL = TblL(cell_names, :);
    vnames = TblL.Properties.VariableNames;

    psd_vec = [];
    if USE_PSD_FEATURE
        Lkey = upper(string(L));
        if isfield(psd_byLoad, char(Lkey))
            rowPSD = psd_byLoad.(char(Lkey));
            psd_vec = nan(1, numel(PSD_FEAT_USE));

            for k = 1:numel(PSD_FEAT_USE)
                featName = PSD_FEAT_USE(k);
                transMode = lower(string(PSD_FEAT_TRANSFORM(k)));

                val = rowPSD{1, char(featName)};
                psd_vec(k) = applyPSDTransform(val, transMode, featName, Lkey);
            end
        else
            psd_vec = nan(1, numel(PSD_FEAT_USE));
        end
    end

    for ci = 1:nC
        row = row + 1;
        load_id(row) = string(L);
        cell_id(row) = string(cell_names{ci});

        col = 0;

        % ECM feature
        for s = SOC_use(:).'
            for pi = 1:numel(pNames_2RC)
                col = col + 1;
                pname = pNames_2RC{pi};

                if pi <= 3
                    varName = sprintf('SOC%d_%s_mOhm', s, pname);
                else
                    varName = sprintf('SOC%d_%s', s, pname);
                end

                if ismember(varName, vnames)
                    X(row, col) = TblL{ci, varName};
                else
                    X(row, col) = NaN;
                end
            end
        end

        % PSD feature
        if USE_PSD_FEATURE
            X(row, col+1 : col+numel(psd_vec)) = psd_vec;
            col = col + numel(psd_vec);
        end
    end
end

base_valid_X = all(isfinite(X), 2);

fprintf('\n[DATA] Feature names (%d): %s\n', nFeat, strjoin(feat_names_all, ', '));
fprintf('[DATA] Total samples = %d (cells %d × loads %d)\n', nS, nC, nL_use);
fprintf('[DATA] ECM features = %d, PSD candidate features = %d, total PSD subsets = %d\n', ...
    numel(idx_feat_ecm), nPSD, max(1, 2^nPSD));

%% ── 셀 라벨 입력 (nC 기준) ----------------------------------------------
QC2_user  = [56.14;55.93;52.55;50.52;52.13;48.53;56.34;55.15;39.89;55.63;55.57;56.86];
QC40_user = [57.49;57.57;54.00;52.22;53.45;51.28;57.91;56.51;42.14;57.27;57.18;58.40];

Rcharg_user = [2.17
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
8.56
2.09];

Rcharg_80_90_avg_user = nan(nC,1);
R1s_user              = nan(nC,1);

DCIR1s_T20_user    = nan(nC,1);
DCIR10s_T20_user   = [1.48;1.34;1.97;1.91;1.64;1.76;1.35;1.46;3.44;1.45;1.45;1.41];
DCIRdelta_T20_user = nan(nC,1);

Power_T20_user = [2089.79;2372.03;1427.37;1735.16;1603.14;1677.27;2476.97;2191.48;914.67;4067.20;2196.09;2278.82] / 1000;

QC2_user              = ensureLength(QC2_user,              nC);
QC40_user             = ensureLength(QC40_user,             nC);
Rcharg_user           = ensureLength(Rcharg_user,           nC);
Rcharg_80_90_avg_user = ensureLength(Rcharg_80_90_avg_user, nC);
R1s_user              = ensureLength(R1s_user,              nC);

DCIR1s_T20_user     = ensureLength(DCIR1s_T20_user,     nC);
DCIR10s_T20_user    = ensureLength(DCIR10s_T20_user,    nC);
DCIRdelta_T20_user  = ensureLength(DCIRdelta_T20_user,  nC);
Power_T20_user      = ensureLength(Power_T20_user,      nC);

DCIR1s_byT  = struct('T20',DCIR1s_T20_user);
DCIR10s_byT = struct('T20',DCIR10s_T20_user);
DCIRd_byT   = struct('T20',DCIRdelta_T20_user);
Power_byT   = struct('T20',Power_T20_user);

expandY = @(y_cell) repmat(y_cell(:), nL_use, 1);

%% ── 타겟 Map 구성 ------------------------------------------------------
target_values = containers.Map();

target_values('QC2')              = expandY(QC2_user(:));
target_values('QC40')             = expandY(QC40_user(:));
target_values('Rcharg')           = expandY(Rcharg_user(:));
target_values('Rcharg_80_90_avg') = expandY(Rcharg_80_90_avg_user(:));
target_values('R1s')              = expandY(R1s_user(:));

for t = TEMP_list
    keyT = sprintf('T%d', t);

    d1_cell  = DCIR1s_byT.(keyT)(:);
    d10_cell = DCIR10s_byT.(keyT)(:);
    dd_cell  = DCIRd_byT.(keyT)(:);
    pw_cell  = Power_byT.(keyT)(:);

    dDCIR_cell = nan(nC,1);
    for i = 1:nC
        if isfinite(dd_cell(i))
            dDCIR_cell(i) = dd_cell(i);
        elseif isfinite(d10_cell(i)) && isfinite(d1_cell(i))
            dDCIR_cell(i) = d10_cell(i) - d1_cell(i);
        else
            dDCIR_cell(i) = NaN;
        end
    end

    target_values(sprintf('DCIR_1s_T%d', t))           = expandY(d1_cell);
    target_values(sprintf('DCIR_10s_T%d', t))          = expandY(d10_cell);
    target_values(sprintf('DCIR_delta_10s_1s_T%d', t)) = expandY(dDCIR_cell);
    target_values(sprintf('Power_T%d', t))             = expandY(pw_cell);
end

%% ── 활성 타겟 리스트 ----------------------------------------------------
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

if isempty(target_active), error('활성 타겟 없음. PRED_ 토글 확인'); end
fprintf('\n[MLR] Active targets: %s\n', strjoin(target_active, ', '));

%% ── LOAD 기준 split 마스크 ----------------------------------------------
is_trainval_load = ismember(load_id, string(load_trainval));
is_test_load     = ismember(load_id, string(load_test));

idx_trainval_base = base_valid_X & is_trainval_load;
idx_test_base     = base_valid_X & is_test_load;

fprintf('[SPLIT] train/val base samples = %d\n', nnz(idx_trainval_base));
fprintf('[SPLIT] test      base samples = %d\n', nnz(idx_test_base));

%% ── 타겟별: exhaustive subset search + CV + final fit + test ----------
results = struct();

% CV-best summary
summary_cv_target = strings(0,1);
summary_cv_best_subset = strings(0,1);
summary_cv_num_psd = [];
summary_cv_rmse = [];
summary_cv_mae = [];
summary_cv_r2 = [];
summary_cv_test_rmse = [];
summary_cv_test_mae = [];
summary_cv_test_r2 = [];

% Test-best summary
summary_te_target = strings(0,1);
summary_te_best_subset = strings(0,1);
summary_te_num_psd = [];
summary_te_cv_rmse = [];
summary_te_cv_mae = [];
summary_te_cv_r2 = [];
summary_te_rmse = [];
summary_te_mae = [];
summary_te_r2 = [];

for tt = 1:numel(target_active)
    tname = target_active{tt};
    y_all = target_values(tname);

    idx_trainval = idx_trainval_base & isfinite(y_all);
    idx_test     = idx_test_base     & isfinite(y_all);

    X_tv_full = X(idx_trainval, :);
    y_tv = y_all(idx_trainval);
    l_tv = load_id(idx_trainval);

    X_te_full = X(idx_test, :);
    y_te = y_all(idx_test);

    n_tv = size(X_tv_full,1);
    n_te = size(X_te_full,1);

    fprintf('\n[TARGET %s] train/val=%d, test=%d\n', tname, n_tv, n_te);

    %% ---- Exhaustive search over all PSD subsets -----------------------
    if USE_PSD_FEATURE && nPSD > 0
        subset_masks = 0:(2^nPSD - 1);   % 0 = PSD 없음(ECM only)
    else
        subset_masks = 0;
    end

    nComb = numel(subset_masks);
    fprintf('  [EXHAUSTIVE] evaluating %d PSD subsets (including empty set)\n', nComb);

    combo_id_col = nan(nComb,1);
    n_psd_col = nan(nComb,1);
    subset_str_col = strings(nComb,1);

    cv_rmse_col = nan(nComb,1);
    cv_mae_col  = nan(nComb,1);
    cv_r2_col   = nan(nComb,1);

    test_rmse_col = nan(nComb,1);
    test_mae_col  = nan(nComb,1);
    test_r2_col   = nan(nComb,1);

    lambda_col = nan(nComb,1);
    feat_idx_cell = cell(nComb,1);

    best_score_cv = inf;
    best_idx_cv = 1;

    best_score_test = inf;
    best_idx_test = 1;

    for cc = 1:nComb
        mask = subset_masks(cc);
        psd_local_sel = find(bitget(mask, 1:nPSD) == 1);  % local PSD index
        feat_idx_use = [idx_feat_ecm, idx_feat_psd(psd_local_sel)];

        evalRes = localEvaluateFeatureSet_LoadCV( ...
            X_tv_full(:, feat_idx_use), y_tv, l_tv, K_FOLD, ...
            DO_ZSCORE, RIDGE_LAMBDA_MODE, RIDGE_LAMBDA_FIXED, RIDGE_LAMBDA_GRID, ...
            LAMBDA_SCORE_METRIC, SEARCH_SCORE_METRIC);

        combo_id_col(cc) = cc;
        n_psd_col(cc) = numel(psd_local_sel);
        feat_idx_cell{cc} = feat_idx_use;

        if isempty(psd_local_sel)
            subset_str_col(cc) = "(none)";
        else
            subset_str_col(cc) = strjoin(string(featPSD_names(psd_local_sel)), ", ");
        end

        % ----- train/val 전체로 fit 후 test metric 계산 -----
        lambda_use_cc = evalRes.lambda_use;

        X_fit_tv = X_tv_full(:, feat_idx_use);
        X_fit_te = X_te_full(:, feat_idx_use);

        if DO_ZSCORE
            [X_fit_tv_z, mu_cc, sg_cc] = zscoreTrain(X_fit_tv);
            X_fit_te_z = zscoreApply(X_fit_te, mu_cc, sg_cc);
        else
            X_fit_tv_z = X_fit_tv;
            X_fit_te_z = X_fit_te;
        end

        mdl_cc = fitrlinear(X_fit_tv_z, y_tv, ...
            'Learner','leastsquares', ...
            'Regularization','ridge', ...
            'Lambda', lambda_use_cc, ...
            'FitBias', true);

        if size(X_fit_te,1) >= 2
            yhat_te_cc = predict(mdl_cc, X_fit_te_z);
            res_te_cc = y_te - yhat_te_cc;

            test_rmse_cc = sqrt(mean(res_te_cc.^2));
            test_mae_cc  = mean(abs(res_te_cc));
            test_r2_cc   = calcR2(y_te, yhat_te_cc);
        else
            test_rmse_cc = NaN;
            test_mae_cc  = NaN;
            test_r2_cc   = NaN;
        end

        cv_rmse_col(cc) = evalRes.RMSE;
        cv_mae_col(cc)  = evalRes.MAE;
        cv_r2_col(cc)   = evalRes.R2;

        test_rmse_col(cc) = test_rmse_cc;
        test_mae_col(cc)  = test_mae_cc;
        test_r2_col(cc)   = test_r2_cc;

        lambda_col(cc)  = evalRes.lambda_use;

        if SEARCH_SCORE_METRIC == "MAE"
            score_cv_cc = evalRes.MAE;
        else
            score_cv_cc = evalRes.RMSE;
        end

        if isfinite(score_cv_cc) && score_cv_cc < best_score_cv
            best_score_cv = score_cv_cc;
            best_idx_cv = cc;
        end

        if isfinite(test_rmse_cc) && test_rmse_cc < best_score_test
            best_score_test = test_rmse_cc;
            best_idx_test = cc;
        end

        if mod(cc, max(1, floor(nComb/10))) == 0 || cc == nComb
            fprintf('    progress: %d / %d subsets done\n', cc, nComb);
        end
    end

    combo_tbl = table( ...
        combo_id_col, n_psd_col, subset_str_col, ...
        cv_rmse_col, cv_mae_col, cv_r2_col, ...
        test_rmse_col, test_mae_col, test_r2_col, ...
        lambda_col, ...
        'VariableNames', {'ComboID','NumPSD','PSDSubset', ...
        'CV_RMSE','CV_MAE','CV_R2', ...
        'Test_RMSE','Test_MAE','Test_R2', ...
        'Lambda'});

    valid_combo = isfinite(combo_tbl.CV_RMSE);

    if ~any(valid_combo)
        warning('  -> 모든 PSD 조합 평가 실패. target=%s 스킵', tname);
        continue;
    end

    best_row_cv = combo_tbl(best_idx_cv, :);
    best_row_test = combo_tbl(best_idx_test, :);

    best_feat_idx = feat_idx_cell{best_idx_cv};   % 최종 모델은 CV-best 기준
    feat_names_use = feat_names_all(best_feat_idx);
    nFeat_use = numel(best_feat_idx);
    lambda_use = best_row_cv.Lambda;

    fprintf('  [BEST CV SUBSET] NumPSD=%d | PSDSubset=%s | CV_RMSE=%.6f | CV_R2=%.6f | Test_RMSE=%.6f | lambda=%.3g\n', ...
        best_row_cv.NumPSD, best_row_cv.PSDSubset, best_row_cv.CV_RMSE, best_row_cv.CV_R2, best_row_cv.Test_RMSE, lambda_use);

    fprintf('  [BEST TEST SUBSET] NumPSD=%d | PSDSubset=%s | Test_RMSE=%.6f | CV_RMSE=%.6f\n', ...
        best_row_test.NumPSD, best_row_test.PSDSubset, best_row_test.Test_RMSE, best_row_test.CV_RMSE);

    %% ---- best by subset size table -----------------------------------
    best_by_size_tbl = table();
    if SAVE_BEST_BY_SIZE_TABLE
        rows_tmp = [];
        for kk = 0:nPSD
            mk = combo_tbl.NumPSD == kk & isfinite(combo_tbl.CV_RMSE);
            if any(mk)
                tblk = combo_tbl(mk,:);
                [~, ib] = min(tblk.CV_RMSE);
                rows_tmp = [rows_tmp; tblk(ib,:)]; %#ok<AGROW>
            end
        end
        best_by_size_tbl = rows_tmp;
    end

    %% ---- choose feature matrix by CV-best subset ----------------------
    X_tv = X_tv_full(:, best_feat_idx);
    X_te = X_te_full(:, best_feat_idx);

    if n_tv < nFeat_use + 1
        warning('  -> train/val 샘플 부족: %d < %d (nFeat+1). 스킵', n_tv, nFeat_use+1);
        continue;
    end

    %% ---- re-run LOAD-level CV for chosen subset (store yhat_cv) -------
    cvRes = localRunLoadCVWithFixedLambda( ...
        X_tv, y_tv, l_tv, K_FOLD, ...
        DO_ZSCORE, lambda_use);

    yhat_cv   = cvRes.yhat_cv;
    fold_R2   = cvRes.fold_R2;
    fold_RMSE = cvRes.fold_RMSE;
    fold_MAE  = cvRes.fold_MAE;
    cv_loads  = cvRes.cv_loads;
    fold_id_of_load = cvRes.fold_id_of_load;

    cv_RMSE = mean(fold_RMSE, 'omitnan');
    cv_MAE  = mean(fold_MAE,  'omitnan');
    cv_R2   = mean(fold_R2,   'omitnan');

    fprintf('  [LOAD-CV %d-fold | chosen subset] R2=%.4f, RMSE=%.4f, MAE=%.4f\n', ...
        cvRes.K, cv_R2, cv_RMSE, cv_MAE);

    %% ---- Final model --------------------------------------------------
    if DO_ZSCORE
        [Xtv_z, mu_final, sg_final] = zscoreTrain(X_tv);
        Xte_z = zscoreApply(X_te, mu_final, sg_final);
    else
        Xtv_z = X_tv; Xte_z = X_te;
        mu_final = []; sg_final = [];
    end

    mdl = fitrlinear(Xtv_z, y_tv, ...
        'Learner','leastsquares', ...
        'Regularization','ridge', ...
        'Lambda', lambda_use, ...
        'FitBias', true);

    yhat_tv = predict(mdl, Xtv_z);
    res_tv  = y_tv - yhat_tv;
    tv_RMSE = sqrt(mean(res_tv.^2));
    tv_MAE  = mean(abs(res_tv));
    tv_R2   = calcR2(y_tv, yhat_tv);

    %% ---- Test evaluation ----------------------------------------------
    test_metrics = struct('R2',nan,'RMSE',nan,'MAE',nan);
    yhat_te = [];
    res_te  = [];
    if n_te >= 2
        yhat_te = predict(mdl, Xte_z);
        res_te  = y_te - yhat_te;

        test_metrics.RMSE = sqrt(mean(res_te.^2));
        test_metrics.MAE  = mean(abs(res_te));
        test_metrics.R2   = calcR2(y_te, yhat_te);

        fprintf('  [TEST] R2=%.4f, RMSE=%.4f, MAE=%.4f\n', ...
            test_metrics.R2, test_metrics.RMSE, test_metrics.MAE);
    else
        fprintf('  [TEST] test 샘플 부족/없음 -> test 평가 스킵\n');
    end

    %% ---- Save results --------------------------------------------------
    results.(tname).mdl = mdl;

    results.(tname).trainval.idx_global = idx_trainval;
    results.(tname).trainval.X = X_tv;
    results.(tname).trainval.X_z = Xtv_z;
    results.(tname).trainval.y_true = y_tv;
    results.(tname).trainval.y_pred = yhat_tv;
    results.(tname).trainval.residuals = res_tv;
    results.(tname).trainval.R2 = tv_R2;
    results.(tname).trainval.RMSE = tv_RMSE;
    results.(tname).trainval.MAE = tv_MAE;

    results.(tname).cv.K = cvRes.K;
    results.(tname).cv.loads = cv_loads;
    results.(tname).cv.fold_id_of_load = fold_id_of_load;
    results.(tname).cv.y_pred = yhat_cv;
    results.(tname).cv.fold_R2 = fold_R2;
    results.(tname).cv.fold_RMSE = fold_RMSE;
    results.(tname).cv.fold_MAE = fold_MAE;
    results.(tname).cv.R2_mean = cv_R2;
    results.(tname).cv.RMSE_mean = cv_RMSE;
    results.(tname).cv.MAE_mean = cv_MAE;

    results.(tname).test.idx_global = idx_test;
    results.(tname).test.X = X_te;
    results.(tname).test.X_z = Xte_z;
    results.(tname).test.y_true = y_te;
    results.(tname).test.y_pred = yhat_te;
    results.(tname).test.residuals = res_te;
    results.(tname).test.metrics = test_metrics;

    results.(tname).feature_names = feat_names_use;
    results.(tname).selected_feature_idx = best_feat_idx;
    results.(tname).SOC_use = SOC_use;
    results.(tname).LOAD_USE_STR = LOAD_USE_STR;
    results.(tname).LOAD_TRAINVAL = load_trainval;
    results.(tname).LOAD_TEST = load_test;

    results.(tname).zscore.do = DO_ZSCORE;
    results.(tname).zscore.mu = mu_final;
    results.(tname).zscore.sigma = sg_final;
    results.(tname).ridge.lambda = lambda_use;
    results.(tname).ridge.mode = RIDGE_LAMBDA_MODE;

    results.(tname).psd.use = USE_PSD_FEATURE;
    results.(tname).psd.features = PSD_FEAT_USE;
    results.(tname).psd.transforms = PSD_FEAT_TRANSFORM;

    results.(tname).subset_search.metric = SEARCH_SCORE_METRIC;
    results.(tname).subset_search.best_row_cv = best_row_cv;
    results.(tname).subset_search.best_row_test = best_row_test;
    results.(tname).subset_search.best_feat_idx = best_feat_idx;
    results.(tname).subset_search.best_feat_names = feat_names_use;
    results.(tname).subset_search.all_combo_table = combo_tbl;
    results.(tname).subset_search.best_by_size_table = best_by_size_tbl;

    results.(tname).trainval.load_id = l_tv;
    results.(tname).trainval.cell_id = cell_id(idx_trainval);
    results.(tname).test.load_id = load_id(idx_test);
    results.(tname).test.cell_id = cell_id(idx_test);

    %% ---- plot ----------------------------------------------------------
    fig_all = figure('Color','w','Name',sprintf('ALL_%s_true_vs_pred',tname));
    hold on; grid on;

    h_tv = scatter(y_tv, yhat_tv, 35, 'filled', ...
        'Marker','o', 'MarkerFaceColor',[0 0 0], 'MarkerEdgeColor',[0 0 0]);

    h_te = [];
    if n_te >= 2
        h_te = scatter(y_te, yhat_te, 35, 'filled', ...
            'Marker','o', 'MarkerFaceColor',[0.85 0.1 0.1], 'MarkerEdgeColor',[0.85 0.1 0.1]);
    end

    all_true = y_tv;  all_pred = yhat_tv;
    if n_te >= 2
        all_true = [all_true; y_te];
        all_pred = [all_pred; yhat_te];
    end
    minv = min([all_true; all_pred]);
    maxv = max([all_true; all_pred]);
    plot([minv maxv],[minv maxv],'k--','LineWidth',1.2);

    tname_tex = strrep(tname, '_', '\_');
    xlabel(sprintf('True %s', tname_tex), 'Interpreter','tex','FontWeight','bold');
    ylabel(sprintf('Pred %s', tname_tex), 'Interpreter','tex','FontWeight','bold');

    if n_te >= 2
        title(sprintf('%s | CV-best PSD subset | Train/Val: R2=%.3f RMSE=%.3f | CV: R2=%.3f RMSE=%.3f | Test: R2=%.3f RMSE=%.3f', ...
            tname, tv_R2, tv_RMSE, cv_R2, cv_RMSE, test_metrics.R2, test_metrics.RMSE), 'Interpreter','none');
    else
        title(sprintf('%s | CV-best PSD subset | Train/Val: R2=%.3f RMSE=%.3f | CV: R2=%.3f RMSE=%.3f | Test: (none)', ...
            tname, tv_R2, tv_RMSE, cv_R2, cv_RMSE), 'Interpreter','none');
    end

    axis equal; axis tight;

    if isempty(h_te)
        legend(h_tv, {'Train/Val'}, 'Location','best');
    else
        legend([h_tv h_te], {'Train/Val','Test'}, 'Location','best');
    end

    exportgraphics(fig_all, fullfile(save_path, sprintf('ALL_RIDGE_%s_true_vs_pred.png', tname)), 'Resolution', 220);
    savefig(fig_all, fullfile(save_path, sprintf('ALL_RIDGE_%s_true_vs_pred.fig', tname)));

    %% ---- coefficient save ---------------------------------------------
    beta = mdl.Beta;
    bias = mdl.Bias;

    coef_tbl = table(["(Intercept)"; string(feat_names_use(:))], [bias; beta(:)], ...
        'VariableNames', {'Term','Estimate'});
    coef_tbl.Properties.Description = sprintf('Ridge coeffs for %s | lambda=%.3g | zscore=%d | TrainValLoads=%s | TestLoads=%s', ...
        tname, lambda_use, DO_ZSCORE, strjoin(load_trainval, ', '), strjoin(load_test, ', '));

    writetable(coef_tbl, fullfile(save_path, sprintf('RIDGE_coeffs_FINAL_%s.csv', tname)));

    %% ---- save combo tables to one Excel --------------------------------
    if SAVE_ALL_COMBO_TABLE
        writetable(combo_tbl, summary_xlsx_path, 'Sheet', matlab.lang.makeValidName(sprintf('%s_allcombos', tname)));
    end

    if SAVE_BEST_BY_SIZE_TABLE && ~isempty(best_by_size_tbl)
        writetable(best_by_size_tbl, summary_xlsx_path, 'Sheet', matlab.lang.makeValidName(sprintf('%s_best_by_size', tname)));
    end

    %% ---- summary accum -------------------------------------------------
    % CV-best summary
    summary_cv_target(end+1,1) = string(tname); %#ok<AGROW>
    summary_cv_best_subset(end+1,1) = best_row_cv.PSDSubset; %#ok<AGROW>
    summary_cv_num_psd(end+1,1) = best_row_cv.NumPSD; %#ok<AGROW>
    summary_cv_rmse(end+1,1) = best_row_cv.CV_RMSE; %#ok<AGROW>
    summary_cv_mae(end+1,1)  = best_row_cv.CV_MAE; %#ok<AGROW>
    summary_cv_r2(end+1,1)   = best_row_cv.CV_R2; %#ok<AGROW>
    summary_cv_test_rmse(end+1,1) = best_row_cv.Test_RMSE; %#ok<AGROW>
    summary_cv_test_mae(end+1,1)  = best_row_cv.Test_MAE; %#ok<AGROW>
    summary_cv_test_r2(end+1,1)   = best_row_cv.Test_R2; %#ok<AGROW>

    % Test-best summary
    summary_te_target(end+1,1) = string(tname); %#ok<AGROW>
    summary_te_best_subset(end+1,1) = best_row_test.PSDSubset; %#ok<AGROW>
    summary_te_num_psd(end+1,1) = best_row_test.NumPSD; %#ok<AGROW>
    summary_te_cv_rmse(end+1,1) = best_row_test.CV_RMSE; %#ok<AGROW>
    summary_te_cv_mae(end+1,1)  = best_row_test.CV_MAE; %#ok<AGROW>
    summary_te_cv_r2(end+1,1)   = best_row_test.CV_R2; %#ok<AGROW>
    summary_te_rmse(end+1,1) = best_row_test.Test_RMSE; %#ok<AGROW>
    summary_te_mae(end+1,1)  = best_row_test.Test_MAE; %#ok<AGROW>
    summary_te_r2(end+1,1)   = best_row_test.Test_R2; %#ok<AGROW>
end

%% ── summary 테이블 생성 및 저장 ----------------------------------------
cv_best_summary_tbl = table( ...
    summary_cv_target, summary_cv_num_psd, summary_cv_best_subset, ...
    summary_cv_rmse, summary_cv_mae, summary_cv_r2, ...
    summary_cv_test_rmse, summary_cv_test_mae, summary_cv_test_r2, ...
    'VariableNames', {'Target','NumPSD','CV_BestPSDSubset', ...
    'CV_RMSE','CV_MAE','CV_R2', ...
    'Test_RMSE','Test_MAE','Test_R2'});

test_best_summary_tbl = table( ...
    summary_te_target, summary_te_num_psd, summary_te_best_subset, ...
    summary_te_cv_rmse, summary_te_cv_mae, summary_te_cv_r2, ...
    summary_te_rmse, summary_te_mae, summary_te_r2, ...
    'VariableNames', {'Target','NumPSD','Test_BestPSDSubset', ...
    'CV_RMSE','CV_MAE','CV_R2', ...
    'Test_RMSE','Test_MAE','Test_R2'});

disp('=== CV-best PSD subset summary by target ===');
disp(cv_best_summary_tbl);

disp('=== Test-best PSD subset summary by target ===');
disp(test_best_summary_tbl);

writetable(cv_best_summary_tbl, summary_xlsx_path, 'Sheet', 'CV_best_summary');
writetable(test_best_summary_tbl, summary_xlsx_path, 'Sheet', 'Test_best_summary');

%% ── 저장 ---------------------------------------------------------------
save(fullfile(save_path, 'MLR_RIDGE_results_split_LOADCV_withPSD_EXHAUSTIVE.mat'), ...
    'results', 'X', 'feat_names_all', 'cell_names', 'SOC_use', 'pNames_2RC', ...
    'LOAD_USE_STR', 'LOAD_TEST_STR', 'LOAD_TRAINVAL_STR', 'K_FOLD', 'TEMP_list', ...
    'load_id', 'cell_id', 'load_use', 'load_trainval', 'load_test', ...
    'USE_PSD_FEATURE', 'psdMatPath', 'PSD_FEAT_USE', 'PSD_FEAT_TRANSFORM', ...
    'DO_ZSCORE','RIDGE_LAMBDA_MODE','RIDGE_LAMBDA_FIXED','RIDGE_LAMBDA_GRID','LAMBDA_SCORE_METRIC', ...
    'SEARCH_SCORE_METRIC','cv_best_summary_tbl','test_best_summary_tbl','summary_xlsx_path');

disp('완료: ECM 고정 + PSD 모든 조합 완전탐색 + CV/test 기준 요약 + 단일 Excel 저장 완료.');

%% ========================= 보조 함수들 =================================
function loads = parseLoadList(str0)
if isstring(str0), str0 = char(str0); end
str0 = strtrim(str0);
if isempty(str0), loads = {}; return; end
parts = regexp(str0, '[,;\s]+', 'split');
parts = parts(~cellfun(@isempty,parts));
loads = parts(:).';
end

function loads_std = toStdCase(loads_upper_cell, loadNames_all)
all_upper = upper(string(loadNames_all));
loads_std = cell(1,numel(loads_upper_cell));
for i = 1:numel(loads_upper_cell)
    idx = find(all_upper == string(loads_upper_cell{i}), 1, 'first');
    loads_std{i} = loadNames_all{idx};
end
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

function r2 = calcR2(y, yhat)
ssr = sum((y - yhat).^2);
sst = sum((y - mean(y)).^2);
if sst <= eps
    r2 = NaN;
else
    r2 = 1 - ssr/sst;
end
end

function [Xz, mu, sg] = zscoreTrain(X)
mu = mean(X, 1, 'omitnan');
sg = std(X, 0, 1, 'omitnan');
sg(~isfinite(sg) | sg < 1e-12) = 1;
Xz = (X - mu) ./ sg;
end

function Xz = zscoreApply(X, mu, sg)
sg(~isfinite(sg) | sg < 1e-12) = 1;
Xz = (X - mu) ./ sg;
end

function out = applyPSDTransform(x, mode, featName, loadName)
mode = lower(string(mode));

if ~isfinite(x)
    out = NaN;
    return;
end

switch mode
    case "none"
        out = x;

    case "log10"
        if x > 0
            out = log10(x);
        else
            warning("log10 변환 불가: feature=%s, load=%s, value=%.6g -> NaN 처리", ...
                string(featName), string(loadName), x);
            out = NaN;
        end

    case "log1p"
        if x > -1
            out = log1p(x);
        else
            warning("log1p 변환 불가: feature=%s, load=%s, value=%.6g -> NaN 처리", ...
                string(featName), string(loadName), x);
            out = NaN;
        end

    otherwise
        error("알 수 없는 PSD transform mode: %s (feature=%s)", string(mode), string(featName));
end
end

function evalRes = localEvaluateFeatureSet_LoadCV( ...
    X_tv, y_tv, l_tv, K_FOLD, ...
    DO_ZSCORE, RIDGE_LAMBDA_MODE, RIDGE_LAMBDA_FIXED, RIDGE_LAMBDA_GRID, ...
    LAMBDA_SCORE_METRIC, SCORE_METRIC)

nFeat_use = size(X_tv,2);
n_tv = size(X_tv,1);

evalRes = struct('RMSE',inf,'MAE',inf,'R2',NaN,'lambda_use',NaN);

if n_tv < nFeat_use + 1
    return;
end

lambda_use = RIDGE_LAMBDA_FIXED;

% ----- lambda 선택 -----
if RIDGE_LAMBDA_MODE == "auto"
    cv_loads0 = unique(l_tv, 'stable');
    nL0 = numel(cv_loads0);
    K0 = min(K_FOLD, nL0);

    rng(0);
    perm0 = randperm(nL0);
    fold_id0 = zeros(nL0,1);
    for i = 1:nL0
        fold_id0(perm0(i)) = mod(i-1, K0) + 1;
    end

    bestScore0 = inf;
    bestLam0 = RIDGE_LAMBDA_GRID(1);

    for lam = RIDGE_LAMBDA_GRID
        yhat_tmp = nan(size(y_tv));

        for k0 = 1:K0
            val_loads0 = cv_loads0(fold_id0 == k0);
            is_val0 = ismember(l_tv, val_loads0);
            is_tr0  = ~is_val0;

            if nnz(is_val0) < 2 || nnz(is_tr0) < nFeat_use + 1
                continue;
            end

            Xtr0 = X_tv(is_tr0,:);
            ytr0 = y_tv(is_tr0);
            Xva0 = X_tv(is_val0,:);

            if DO_ZSCORE
                [Xtr0z, mu0, sg0] = zscoreTrain(Xtr0);
                Xva0z = zscoreApply(Xva0, mu0, sg0);
            else
                Xtr0z = Xtr0;
                Xva0z = Xva0;
            end

            mdl0 = fitrlinear(Xtr0z, ytr0, ...
                'Learner','leastsquares', ...
                'Regularization','ridge', ...
                'Lambda', lam, ...
                'FitBias', true);

            yhat_tmp(is_val0) = predict(mdl0, Xva0z);
        end

        mOk = isfinite(yhat_tmp);
        if nnz(mOk) < 5
            continue;
        end

        if LAMBDA_SCORE_METRIC == "MAE"
            score0 = mean(abs(y_tv(mOk) - yhat_tmp(mOk)));
        else
            score0 = sqrt(mean((y_tv(mOk) - yhat_tmp(mOk)).^2));
        end

        if score0 < bestScore0
            bestScore0 = score0;
            bestLam0 = lam;
        end
    end

    lambda_use = bestLam0;
end

% ----- 선택된 lambda로 CV 평가 -----
cvRes = localRunLoadCVWithFixedLambda(X_tv, y_tv, l_tv, K_FOLD, DO_ZSCORE, lambda_use);

evalRes.RMSE = mean(cvRes.fold_RMSE, 'omitnan');
evalRes.MAE  = mean(cvRes.fold_MAE,  'omitnan');
evalRes.R2   = mean(cvRes.fold_R2,   'omitnan');
evalRes.lambda_use = lambda_use;

if SCORE_METRIC == "MAE"
    % no-op
else
    % no-op
end
end

function cvRes = localRunLoadCVWithFixedLambda(X_tv, y_tv, l_tv, K_FOLD, DO_ZSCORE, lambda_use)

nFeat_use = size(X_tv,2);
cvRes = struct();

cv_loads = unique(l_tv, 'stable');
nL_tv = numel(cv_loads);
K = min(K_FOLD, nL_tv);

rng(0);
perm = randperm(nL_tv);
fold_id_of_load = zeros(nL_tv,1);
for i = 1:nL_tv
    fold_id_of_load(perm(i)) = mod(i-1, K) + 1;
end

yhat_cv   = nan(size(y_tv));
fold_R2   = nan(K,1);
fold_RMSE = nan(K,1);
fold_MAE  = nan(K,1);

for k = 1:K
    val_loads = cv_loads(fold_id_of_load == k);

    is_val = ismember(l_tv, val_loads);
    is_tr  = ~is_val;

    if nnz(is_val) < 2 || nnz(is_tr) < nFeat_use + 1
        continue;
    end

    Xtr = X_tv(is_tr,:);
    ytr = y_tv(is_tr);
    Xva = X_tv(is_val,:);

    if DO_ZSCORE
        [Xtrz, mu, sg] = zscoreTrain(Xtr);
        Xvaz = zscoreApply(Xva, mu, sg);
    else
        Xtrz = Xtr;
        Xvaz = Xva;
    end

    mdl = fitrlinear(Xtrz, ytr, ...
        'Learner','leastsquares', ...
        'Regularization','ridge', ...
        'Lambda', lambda_use, ...
        'FitBias', true);

    yhat_k = predict(mdl, Xvaz);
    yhat_cv(is_val) = yhat_k;

    res_k = y_tv(is_val) - yhat_k;
    fold_RMSE(k) = sqrt(mean(res_k.^2));
    fold_MAE(k)  = mean(abs(res_k));
    fold_R2(k)   = calcR2(y_tv(is_val), yhat_k);
end

cvRes.K = K;
cvRes.cv_loads = cv_loads;
cvRes.fold_id_of_load = fold_id_of_load;
cvRes.yhat_cv = yhat_cv;
cvRes.fold_R2 = fold_R2;
cvRes.fold_RMSE = fold_RMSE;
cvRes.fold_MAE = fold_MAE;
end