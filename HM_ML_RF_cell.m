%% ======================================================================
%  RF: 2RC(Tbl_Load_ECM) + labels
%   - Cell-based split + Cell-level K-fold CV
%   - Feature load 선택(LOAD_FEAT_STR)
%   - Test cell ID 수동 지정(TEST_CELL_ID_STR)
%   - Random search 튜닝을 cell-level CV로 평가
%   - 성능 평가는 RMSE만 사용
%
%  Based on: HM_ML_SVR_cell.m
% ======================================================================
clear; clc; close all;

% Reproducibility for random search and cell-level CV fold generation
RANDOM_SEED = 42;
rng(RANDOM_SEED);

% (PATCH) Text Interpreter 에러 방지
set(groot,'defaultTextInterpreter','none');
set(groot,'defaultAxesTickLabelInterpreter','none');
set(groot,'defaultLegendInterpreter','none');

%% ── 설정 --------------------------------------------------------------
SOC_use   = [70];
TEMP_list = [20];

% feature에 사용할 주행부하만 선택
LOAD_FEAT_STR = "US06";       % 대표 주행부하 1종 사용

% test 셀은 ID로 직접 지정
TEST_CELL_ID_STR = "02 04";

% CV fold (cell 기준)
K_FOLD = 5;

% ---- 타겟 토글 ----
PRED_QC2         = true;
PRED_QC40        = true;
PRED_RCHARG      = true;
PRED_RCHARG_8090 = false;
PRED_R1S         = false;

PRED_DCIR1S_BYTEMP  = false;
PRED_DCIR10S_BYTEMP = true;
PRED_DDELTA_BYTEMP  = false;
PRED_POWER_BYTEMP   = false;

% ---- RF 튜닝 설정 ----
% TreeBagger 기반 Random Forest regression
TUNE_MODE = "random";     % "fixed" | "random"

% fixed mode
FIXED_NUM_TREES = 200;
FIXED_MIN_LEAF  = 1;
FIXED_NUM_PREDICTORS_TO_SAMPLE = "all";  % "all" 또는 정수

% random search mode
RAND_N_TRIALS = 200;
RAND_NUM_TREES_SET = [50 100 200 300 500];
RAND_MIN_LEAF_SET  = [1 2 3 5];
% feature가 5개일 때 후보: 1,2,3,5,all
RAND_NUM_PRED_SET  = ["1","2","3","5","all"];

% 튜닝 기준: mean-fold RMSE
TUNE_METRIC = "RMSE";

%% ── 경로 --------------------------------------------------------------
matPath   = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\20degC\2RC_fitting_600s\2RC_results_600s.mat";
save_path = "C:\Users\junny\OneDrive\바탕 화면\이범식_머신러닝\term_project\RF_results";
if ~exist(save_path, 'dir'), mkdir(save_path); end

%% ── load 파싱/검증 -----------------------------------------------------
loadNames_all = {'US06','UDDS','HWFET','WLTP','CITY1','CITY2','HW1','HW2'};
all_upper = upper(string(loadNames_all));

load_feat = upper(parseLoadList(LOAD_FEAT_STR));
ok = ismember(string(load_feat), all_upper);
if any(~ok)
    warning('알 수 없는 주행부하 입력 제외: %s', strjoin(load_feat(~ok), ', '));
end
load_feat = load_feat(ok);
load_feat = toStdCase(load_feat, loadNames_all);
if isempty(load_feat), error('LOAD_FEAT_STR 유효 load 없음'); end
fprintf('>> LOAD_FEAT (features): %s\n', strjoin(load_feat, ', '));

%% ── 2RC 테이블 로드 ----------------------------------------------------
S = load(matPath);
getTblECM = @(loadName) localGetTblECM(S, loadName);

% 실제 mat에 존재하는 feature load만 남기기
load_feat_ok = {};
for i = 1:numel(load_feat)
    try
        Ttmp = getTblECM(load_feat{i});
        if istable(Ttmp), load_feat_ok{end+1} = load_feat{i}; end %#ok<AGROW>
    catch
        warning('mat에 %s ECM 테이블 없음 → 제외', load_feat{i});
    end
end
load_feat = load_feat_ok;
if isempty(load_feat), error('LOAD_FEAT 중 mat에 존재하는 테이블이 없음'); end
fprintf('>> (exists) LOAD_FEAT: %s\n', strjoin(load_feat, ', '));

%% ── 공통 셀(RowNames) 교집합 (LOAD_FEAT 내) ----------------------------
cell_sets = cell(numel(load_feat),1);
for i = 1:numel(load_feat)
    T = getTblECM(load_feat{i});
    cell_sets{i} = T.Properties.RowNames;
end
cell_names = cell_sets{1};
for i = 2:numel(cell_sets)
    cell_names = intersect(cell_names, cell_sets{i}, 'stable');
end
if isempty(cell_names), error('LOAD_FEAT 간 공통 셀이 없음'); end
nC = numel(cell_names);
fprintf('>> 공통 셀 개수 (LOAD_FEAT 기준): %d\n', nC);

%% ── X 구성: sample=(cell, load_feat) -----------------------------------
pNames_2RC = {'R0','R1','R2','tau1','tau2'};

feat_names = {};
for s = SOC_use(:).'
    for pi = 1:numel(pNames_2RC)
        feat_names{end+1} = sprintf('%s_%d', pNames_2RC{pi}, s); %#ok<AGROW>
    end
end
nFeat = numel(feat_names);

nL_feat = numel(load_feat);
nS = nC * nL_feat;

X = nan(nS, nFeat);
load_id = strings(nS,1);
cell_id = strings(nS,1);

row = 0;
for li = 1:nL_feat
    L = load_feat{li};
    TblL = getTblECM(L);
    TblL = TblL(cell_names, :);
    vnames = TblL.Properties.VariableNames;

    for ci = 1:nC
        row = row + 1;
        load_id(row) = string(L);
        cell_id(row) = string(cell_names{ci});

        col = 0;
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
    end
end

base_valid_X = all(isfinite(X), 2);
fprintf('[DATA] samples=%d (cells %d × feat-loads %d), features=%d\n', nS, nC, nL_feat, nFeat);

%% ── 라벨 입력 (nC 기준) ------------------------------------------------
QC2_user  = [56.14;55.93;52.55;50.52;52.13;48.53;56.34;55.15;39.89;55.63;55.57;56.86];
QC40_user = [57.49;57.57;54.00;52.22;53.45;51.28;57.91;56.51;42.14;57.27;57.18;58.40];

Rcharg_user = [2.17;1.90;3.50;2.82;2.88;3.38;2.10;1.93;6.41;2.00;2.01;2.09];
Rcharg_80_90_avg_user = nan(nC,1);
R1s_user = nan(nC,1);

DCIR1s_T20_user = nan(nC,1);
DCIR10s_T20_user = [1.48;1.34;1.97;1.91;1.64;1.76;1.35;1.46;3.44;1.45;1.45;1.41];
DCIRdelta_T20_user = nan(nC,1);

Power_T20_user = [2089.79;2372.03;1427.37;1735.16;1603.14;1677.27;2476.97;2191.48;914.67;4067.20;2196.09;2278.82]/1000;

QC2_user              = ensureLength(QC2_user, nC);
QC40_user             = ensureLength(QC40_user, nC);
Rcharg_user           = ensureLength(Rcharg_user, nC);
Rcharg_80_90_avg_user = ensureLength(Rcharg_80_90_avg_user, nC);
R1s_user              = ensureLength(R1s_user, nC);

DCIR1s_T20_user       = ensureLength(DCIR1s_T20_user, nC);
DCIR10s_T20_user      = ensureLength(DCIR10s_T20_user, nC);
DCIRdelta_T20_user    = ensureLength(DCIRdelta_T20_user, nC);
Power_T20_user        = ensureLength(Power_T20_user, nC);

DCIR1s_byT  = struct('T20',DCIR1s_T20_user);
DCIR10s_byT = struct('T20',DCIR10s_T20_user);
DCIRd_byT   = struct('T20',DCIRdelta_T20_user);
Power_byT   = struct('T20',Power_T20_user);

expandY = @(y_cell) repmat(y_cell(:), nL_feat, 1);

target_values = containers.Map();
target_values('QC2')              = expandY(QC2_user);
target_values('QC40')             = expandY(QC40_user);
target_values('Rcharg')           = expandY(Rcharg_user);
target_values('Rcharg_80_90_avg') = expandY(Rcharg_80_90_avg_user);
target_values('R1s')              = expandY(R1s_user);

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

    target_values(sprintf('DCIR_1s_T%d', t))           = expandY(d1);
    target_values(sprintf('DCIR_10s_T%d', t))          = expandY(d10);
    target_values(sprintf('DCIR_delta_10s_1s_T%d', t)) = expandY(dDCIR);
    target_values(sprintf('Power_T%d', t))             = expandY(pw);
end

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
if isempty(target_active), error('활성 타겟 없음'); end
fprintf('>> Active targets: %s\n', strjoin(target_active, ', '));

%% ── Cell-based split: TEST cell IDs ------------------------------------
[test_cells, trainval_cells] = pickTestCellsByID(cell_names, TEST_CELL_ID_STR);
fprintf('>> TEST cells (%d): %s\n', numel(test_cells), strjoin(test_cells, ', '));
fprintf('>> TRAIN/VAL cells (%d)\n', numel(trainval_cells));

if numel(test_cells) < 1
    error('TEST_CELL_ID_STR에서 test 셀이 0개로 선택됨. 파일명/ID 패턴 확인 필요');
end

if numel(trainval_cells) < K_FOLD
    warning('train/val 셀 수(%d) < K_FOLD(%d). K를 줄입니다.', numel(trainval_cells), K_FOLD);
    K_FOLD = numel(trainval_cells);
end

is_trainval_cell = ismember(cell_id, trainval_cells);
is_test_cell_smp = ismember(cell_id, test_cells);

idx_trainval_base = base_valid_X & is_trainval_cell;
idx_test_base     = base_valid_X & is_test_cell_smp;

fprintf('[SPLIT] train/val base samples=%d, test base samples=%d\n', nnz(idx_trainval_base), nnz(idx_test_base));

%% ── 타겟별: RF CELL-CV(mean-fold) + FINAL + TEST -----------------------
results = struct();

for tt = 1:numel(target_active)
    tname = target_active{tt};
    y_all = target_values(tname);

    idx_trainval = idx_trainval_base & isfinite(y_all);
    idx_test     = idx_test_base     & isfinite(y_all);

    X_tv = X(idx_trainval,:);
    y_tv = y_all(idx_trainval);
    c_tv = cell_id(idx_trainval);

    X_te = X(idx_test,:);
    y_te = y_all(idx_test);

    fprintf('\n[TARGET %s] train/val=%d, test=%d\n', tname, size(X_tv,1), size(X_te,1));

    if size(X_tv,1) < 2
        warning('  train/val 샘플 부족 → skip');
        continue;
    end

    cv_cells = unique(c_tv, 'stable');
    nC_tv = numel(cv_cells);
    if nC_tv < 2
        warning('  train/val cell이 1개뿐 → CV 불가, skip');
        continue;
    end

    K = min(K_FOLD, nC_tv);

    % RANDOM_SEED is fixed once at the top of the script.
    perm = randperm(nC_tv);
    fold_id_of_cell = zeros(nC_tv,1);
    for i = 1:nC_tv
        fold_id_of_cell(perm(i)) = mod(i-1, K) + 1;
    end

    % (1) 후보 생성
    if TUNE_MODE == "fixed"
        cand = struct('NumTrees', FIXED_NUM_TREES, ...
                      'MinLeafSize', FIXED_MIN_LEAF, ...
                      'NumPredictorsToSample', FIXED_NUM_PREDICTORS_TO_SAMPLE);
    else
        cand = makeRandomCandidatesRF(RAND_N_TRIALS, RAND_NUM_TREES_SET, RAND_MIN_LEAF_SET, RAND_NUM_PRED_SET, nFeat);
    end

    % (2) 후보별 CELL-CV(mean-fold) 평가
    bestScore = inf;
    bestCand = cand(1);
    bestDetail = struct();

    for ci = 1:numel(cand)
        c0 = cand(ci);

        [cvScore, cvDetail] = evalCellCV_RF_meanfold( ...
            X_tv, y_tv, c_tv, cv_cells, fold_id_of_cell, K, ...
            c0.NumTrees, c0.MinLeafSize, c0.NumPredictorsToSample);

        if ~isfinite(cvScore)
            continue;
        end

        if cvScore < bestScore
            bestScore = cvScore;
            bestCand = c0;
            bestDetail = cvDetail;
        end
    end

    fprintf('  [BEST by CELL-CV(mean)] NumTrees=%d, MinLeaf=%d, NumPredictors=%s, RMSE=%.4f\n', ...
        bestCand.NumTrees, bestCand.MinLeafSize, pred2str(bestCand.NumPredictorsToSample), bestScore);

    % (3) Final 모델 학습
    mdl = trainRF(X_tv, y_tv, bestCand.NumTrees, bestCand.MinLeafSize, bestCand.NumPredictorsToSample);

    yhat_tv = predictRF(mdl, X_tv);
    tv.RMSE = sqrt(mean((y_tv - yhat_tv).^2));
    tv.R2   = calcR2(y_tv, yhat_tv);

    % (4) Test 평가
    te = struct('RMSE',nan,'R2',nan);
    yhat_te = [];
    if size(X_te,1) >= 2
        yhat_te = predictRF(mdl, X_te);
        te.RMSE = sqrt(mean((y_te - yhat_te).^2));
        te.R2   = calcR2(y_te, yhat_te);
        fprintf('  [TEST] R2=%.4f RMSE=%.4f\n', te.R2, te.RMSE);
    else
        fprintf('  [TEST] test 샘플 부족/없음 → 스킵\n');
    end

    % (5) 저장
    results.(tname).best = bestCand;
    results.(tname).cv = bestDetail;
    results.(tname).trainval.metrics = tv;
    results.(tname).test.metrics = te;
    results.(tname).mdl = mdl;

    results.(tname).trainval.y_true = y_tv;
    results.(tname).trainval.y_pred = yhat_tv;
    results.(tname).trainval.cell_id = c_tv;
    results.(tname).trainval.load_id = load_id(idx_trainval);

    results.(tname).test.y_true = y_te;
    results.(tname).test.y_pred = yhat_te;
    results.(tname).test.cell_id = cell_id(idx_test);
    results.(tname).test.load_id = load_id(idx_test);

    % ---- plot: Train/Val + Test ----
    fig_all = figure('Color','w','Name',sprintf('RF_ALL_%s',tname));
    hold on; grid on;

    h_tv = scatter(y_tv, yhat_tv, 30, 'filled', ...
        'Marker','o','MarkerFaceColor',[0 0 0],'MarkerEdgeColor',[0 0 0]);

    h_te = [];
    if numel(yhat_te) >= 2
        h_te = scatter(y_te, yhat_te, 30, 'filled', ...
            'Marker','o','MarkerFaceColor',[0.85 0.1 0.1],'MarkerEdgeColor',[0.85 0.1 0.1]);
    end

    all_true = y_tv; all_pred = yhat_tv;
    if numel(yhat_te) >= 2
        all_true = [all_true; y_te];
        all_pred = [all_pred; yhat_te];
    end
    minv = min([all_true; all_pred]); maxv = max([all_true; all_pred]);
    plot([minv maxv],[minv maxv],'k--','LineWidth',1.2);

    xlabel(sprintf('True %s', tname));
    ylabel(sprintf('Pred %s', tname));

    cvStr = sprintf('CV(mean): RMSE=%.3f', bestDetail.RMSE_mean);

    if numel(yhat_te) >= 2
        title(sprintf('%s | Train/Val: R2=%.3f RMSE=%.3f | %s | Test: R2=%.3f RMSE=%.3f', ...
            tname, tv.R2, tv.RMSE, cvStr, te.R2, te.RMSE));
    else
        title(sprintf('%s | Train/Val: R2=%.3f RMSE=%.3f | %s | Test: (none)', ...
            tname, tv.R2, tv.RMSE, cvStr));
    end

    axis equal; axis tight;

    if isempty(h_te)
        legend(h_tv, {'Train/Val'}, 'Location','best');
    else
        legend([h_tv h_te], {'Train/Val','Test'}, 'Location','best');
    end

    exportgraphics(fig_all, fullfile(save_path, sprintf('RF_ALL_%s.png', tname)), 'Resolution', 220);
    savefig(fig_all, fullfile(save_path, sprintf('RF_ALL_%s.fig', tname)));
end


%% ── Summary table for report/result analysis --------------------------
target_names = fieldnames(results);
Model = strings(0,1); Target = strings(0,1); Hyperparameters = strings(0,1);
Train_RMSE = []; CV_RMSE = []; Test_RMSE = [];
Train_R2 = []; CV_R2 = []; Test_R2 = [];
N_TrainVal = []; N_Test = [];
for ii = 1:numel(target_names)
    tname = target_names{ii};
    rr = results.(tname);
    Model(end+1,1) = "RF";
    Target(end+1,1) = string(tname);
    Hyperparameters(end+1,1) = sprintf('NumTrees=%d, MinLeafSize=%d, NumPredictorsToSample=%s', ...
        rr.best.NumTrees, rr.best.MinLeafSize, pred2str(rr.best.NumPredictorsToSample));
    Train_RMSE(end+1,1) = rr.trainval.metrics.RMSE;
    CV_RMSE(end+1,1) = rr.cv.RMSE_mean;
    Test_RMSE(end+1,1) = rr.test.metrics.RMSE;
    Train_R2(end+1,1) = rr.trainval.metrics.R2;
    CV_R2(end+1,1) = rr.cv.R2_mean;
    Test_R2(end+1,1) = rr.test.metrics.R2;
    N_TrainVal(end+1,1) = numel(rr.trainval.y_true);
    N_Test(end+1,1) = numel(rr.test.y_true);
end
summary_tbl = table(Model, Target, Hyperparameters, Train_RMSE, CV_RMSE, Test_RMSE, Train_R2, CV_R2, Test_R2, N_TrainVal, N_Test);
writetable(summary_tbl, fullfile(save_path, 'RF_summary_RMSE.csv'));

%% ── 저장 --------------------------------------------------------------
save(fullfile(save_path, 'RF_results_cellSplit_CV_meanfold.mat'), ...
    'results', 'X', 'feat_names', 'cell_names', 'SOC_use', 'pNames_2RC', ...
    'LOAD_FEAT_STR', 'TEST_CELL_ID_STR', 'K_FOLD', 'TEMP_list', ...
    'load_id', 'cell_id', 'load_feat', ...
    'TUNE_MODE','RAND_N_TRIALS','RAND_NUM_TREES_SET','RAND_MIN_LEAF_SET','RAND_NUM_PRED_SET', ...
    'RANDOM_SEED','summary_tbl');

disp('완료: RF CELL-split + CELL-level CV(mean-fold) + random search + TEST 평가 저장');

%% ========================= helper functions ============================

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
if sst <= eps, r2 = NaN; else, r2 = 1 - ssr/sst; end
end

function cand = makeRandomCandidatesRF(N, numTreesSet, minLeafSet, numPredSet, nFeat)
cand = repmat(struct('NumTrees',100,'MinLeafSize',1,'NumPredictorsToSample','all'), N, 1);
for i = 1:N
    nt = numTreesSet(randi(numel(numTreesSet)));
    ml = minLeafSet(randi(numel(minLeafSet)));
    npStr = numPredSet(randi(numel(numPredSet)));

    if npStr == "all"
        np = 'all';
    else
        np = str2double(npStr);
        if isnan(np) || np < 1
            np = 'all';
        else
            np = min(round(np), nFeat);
        end
    end

    cand(i).NumTrees = nt;
    cand(i).MinLeafSize = ml;
    cand(i).NumPredictorsToSample = np;
end
end

function [score, detail] = evalCellCV_RF_meanfold(X, y, c, cv_cells, fold_id_of_cell, K, numTrees, minLeaf, numPred)
n = size(X,1);
yhat = nan(n,1);

fold_RMSE = nan(K,1);
fold_R2   = nan(K,1);

for k = 1:K
    val_cells = cv_cells(fold_id_of_cell == k);
    is_val = ismember(c, val_cells);
    is_tr  = ~is_val;

    if nnz(is_val) < 1 || nnz(is_tr) < 2
        continue;
    end

    try
        mdl = trainRF(X(is_tr,:), y(is_tr), numTrees, minLeaf, numPred);
        yhat_k = predictRF(mdl, X(is_val,:));
        yhat(is_val) = yhat_k;

        res = y(is_val) - yhat_k;
        fold_RMSE(k) = sqrt(mean(res.^2));
        fold_R2(k)   = calcR2(y(is_val), yhat_k);
    catch
        continue;
    end
end

detail = struct();
detail.K = K;
detail.NumTrees = numTrees;
detail.MinLeafSize = minLeaf;
detail.NumPredictorsToSample = numPred;
detail.y_pred = yhat;

detail.fold_RMSE = fold_RMSE;
detail.fold_R2   = fold_R2;

detail.RMSE_mean = mean(fold_RMSE,'omitnan');
detail.R2_mean   = mean(fold_R2,'omitnan');

score = detail.RMSE_mean;
end

function mdl = trainRF(X, y, numTrees, minLeaf, numPred)
% TreeBagger regression RF
if ischar(numPred) || isstring(numPred)
    if strcmpi(char(numPred), 'all')
        mdl = TreeBagger(numTrees, X, y, ...
            'Method','regression', ...
            'MinLeafSize', minLeaf, ...
            'OOBPrediction','off');
    else
        np = str2double(numPred);
        mdl = TreeBagger(numTrees, X, y, ...
            'Method','regression', ...
            'MinLeafSize', minLeaf, ...
            'NumPredictorsToSample', np, ...
            'OOBPrediction','off');
    end
else
    mdl = TreeBagger(numTrees, X, y, ...
        'Method','regression', ...
        'MinLeafSize', minLeaf, ...
        'NumPredictorsToSample', numPred, ...
        'OOBPrediction','off');
end
end

function yhat = predictRF(mdl, X)
yhat = predict(mdl, X);
if iscell(yhat)
    yhat = str2double(yhat);
end
yhat = yhat(:);
end

function s = pred2str(x)
if ischar(x) || isstring(x)
    s = char(x);
else
    s = sprintf('%d', x);
end
end

function [test_cells, trainval_cells] = pickTestCellsByID(cell_names, TEST_CELL_ID_STR)
cellStr = string(cell_names(:));

rawParts = regexp(char(TEST_CELL_ID_STR), '[,;\s]+', 'split');
rawParts = rawParts(~cellfun(@isempty, rawParts));
ids = string(rawParts(:));

id_num = str2double(ids);
for i = 1:numel(ids)
    if ~isnan(id_num(i))
        ids(i) = compose("%02d", id_num(i));
    end
end

is_test = false(numel(cellStr),1);

for i = 1:numel(ids)
    id = ids(i);
    pats = [ ...
        "x"+id, ...
        "_"+id+"_", ...
        "-"+id+"-", ...
        "cell"+id, ...
        id ...
    ];

    hit = false(numel(cellStr),1);
    for p = 1:numel(pats)
        hit = hit | contains(lower(cellStr), lower(pats(p)));
    end

    if ~any(hit)
        warning('TEST ID=%s 매칭되는 셀이 없습니다. 예시 cell_names: %s', ...
            id, strjoin(cellStr(1:min(5,end)), ', '));
    end
    is_test = is_test | hit;
end

test_cells = unique(cellStr(is_test), 'stable');
trainval_cells = unique(cellStr(~is_test), 'stable');
end
