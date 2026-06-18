%% SVR / MLR SOH-x prediction
% Train/Val: 7 known driving loads from train cells
% Test     : 1 unseen driving load from held-out test cells
%
% Goal:
%   Train on 10 cells × 7 loads
%   Test  on 2 cells × 1 unseen load
%
% Author: term project version

clear; clc;

%% ===================== USER CONFIG =====================

% 8 driving loads
LOAD_ALL = ["US06","UDDS","HWFET","WLTP","CITY1","CITY2","HW1","HW2"];

% Unseen driving load for test
LOAD_TEST_STR = "US06";

% Train/Val loads = all loads except test load
LOAD_TRAIN = setdiff(LOAD_ALL, LOAD_TEST_STR, 'stable');

% ECM feature names inside each load table
% Change this if you use C1/C2 instead of tau1/tau2
FEAT_NAMES = ["R0_70","R1_70","R2_70","tau1_70","tau2_70"];

% Active SOH-x targets
TARGET_NAMES = ["QC2","QC40","Rcharg","DCIR_10s_T20"];

% Test cells: SOH-unknown cells
TEST_CELLS = [
    "x02_HNE_10degC_1C0_33C_150cyc_2_ch2_260103"
    "x04_HNE___US06_4_ch7_1217"
];

% Label table name in workspace
% This table must contain one row per cell and target columns.
% Example columns:
%   Cell_ID, QC2, QC40, Rcharg, DCIR_10s_T20
LABEL_TABLE_NAME = "Tbl_Label";

% Number of folds for cell-level CV
K_FOLD = 5;

% Random search option for SVR
N_RANDOM_SEARCH = 40;

% SVR hyperparameter search ranges
RAND_C_RANGE_LOG10   = [-3, 2];      % C = 1e-3 ~ 1e2
RAND_EPS_RANGE_LOG10 = [-4, 0];      % epsilon = 1e-4 ~ 1
KERNEL_SET = ["linear","gaussian"];

rng(1);

%% ===================== LOAD LABEL TABLE =====================

if ~evalin('base', sprintf('exist(''%s'', ''var'')', LABEL_TABLE_NAME))
    error("Label table %s does not exist in base workspace.", LABEL_TABLE_NAME);
end

Tbl_Label = evalin('base', LABEL_TABLE_NAME);

CELL_VAR_LABEL = findCellVarName(Tbl_Label);

fprintf(">> TEST LOAD: %s\n", LOAD_TEST_STR);
fprintf(">> TRAIN LOADS (%d): %s\n", numel(LOAD_TRAIN), strjoin(LOAD_TRAIN, ", "));
fprintf(">> TEST cells (%d): %s\n", numel(TEST_CELLS), strjoin(TEST_CELLS, ", "));

%% ===================== BUILD DATASET =====================

% Train/Val data: train cells × 7 train loads
DataTrain = table();

for l = 1:numel(LOAD_TRAIN)
    loadName = LOAD_TRAIN(l);
    Tload = getLoadTable(loadName);
    Ttmp = buildOneLoadDataset(Tload, Tbl_Label, loadName, FEAT_NAMES, TARGET_NAMES);
    DataTrain = [DataTrain; Ttmp]; %#ok<AGROW>
end

% Test data: test cells × 1 unseen load
TtestLoad = getLoadTable(LOAD_TEST_STR);
DataTestAll = buildOneLoadDataset(TtestLoad, Tbl_Label, LOAD_TEST_STR, FEAT_NAMES, TARGET_NAMES);

% Cell split
DataTrain.Cell_ID = string(DataTrain.Cell_ID);
DataTestAll.Cell_ID = string(DataTestAll.Cell_ID);

isTestCellTrainTable = ismember(DataTrain.Cell_ID, TEST_CELLS);
DataTrain(isTestCellTrainTable,:) = [];  % remove held-out cells from train/val

isTestCell = ismember(DataTestAll.Cell_ID, TEST_CELLS);
DataTest = DataTestAll(isTestCell,:);

% Basic check
trainCells = unique(DataTrain.Cell_ID);
testCells  = unique(DataTest.Cell_ID);

fprintf("[DATA] Train/Val samples = %d (%d cells × %d loads expected)\n", ...
    height(DataTrain), numel(trainCells), numel(LOAD_TRAIN));
fprintf("[DATA] Test samples      = %d (%d cells × 1 unseen load)\n", ...
    height(DataTest), numel(testCells));
fprintf("[DATA] Features          = %s\n", strjoin(FEAT_NAMES, ", "));
fprintf("[DATA] Targets           = %s\n", strjoin(TARGET_NAMES, ", "));

if height(DataTest) == 0
    error("No test samples found. Check TEST_CELLS or LOAD_TEST_STR.");
end

%% ===================== FEATURE MATRIX =====================

Xtrain_raw = table2array(DataTrain(:, FEAT_NAMES));
Xtest_raw  = table2array(DataTest(:, FEAT_NAMES));

% Remove rows with NaN features
validTrainX = all(isfinite(Xtrain_raw), 2);
validTestX  = all(isfinite(Xtest_raw), 2);

DataTrain = DataTrain(validTrainX,:);
DataTest  = DataTest(validTestX,:);

Xtrain_raw = table2array(DataTrain(:, FEAT_NAMES));
Xtest_raw  = table2array(DataTest(:, FEAT_NAMES));

% Standardization using train statistics only
muX = mean(Xtrain_raw, 1, 'omitnan');
sigX = std(Xtrain_raw, 0, 1, 'omitnan');
sigX(sigX == 0 | ~isfinite(sigX)) = 1;

Xtrain = (Xtrain_raw - muX) ./ sigX;
Xtest  = (Xtest_raw  - muX) ./ sigX;

%% ===================== CELL-LEVEL CV INDEX =====================

trainCells = unique(DataTrain.Cell_ID);
nCells = numel(trainCells);

if nCells < K_FOLD
    K_FOLD = nCells;
end

cvp = cvpartition(nCells, 'KFold', K_FOLD);

%% ===================== MODEL TRAINING / TEST =====================

Result = table();

for t = 1:numel(TARGET_NAMES)

    targetName = TARGET_NAMES(t);

    ytrain_all = DataTrain.(targetName);
    ytest_all  = DataTest.(targetName);

    validTrain = isfinite(ytrain_all);
    validTest  = isfinite(ytest_all);

    Xtr = Xtrain(validTrain,:);
    ytr = ytrain_all(validTrain);
    cellTr = DataTrain.Cell_ID(validTrain);

    Xte = Xtest(validTest,:);
    yte = ytest_all(validTest);
    cellTe = DataTest.Cell_ID(validTest);

    fprintf("\n[TARGET %s] train/val=%d, test=%d\n", targetName, numel(ytr), numel(yte));

    if numel(yte) < 2
        warning("Target %s has fewer than 2 test samples. R2 may be undefined.", targetName);
    end

    %% ---------- SVR random search with cell-level CV ----------

    bestCVRMSE = inf;
    bestParam = struct();

    for r = 1:N_RANDOM_SEARCH

        Cval = 10^(RAND_C_RANGE_LOG10(1) + rand() * diff(RAND_C_RANGE_LOG10));
        epsVal = 10^(RAND_EPS_RANGE_LOG10(1) + rand() * diff(RAND_EPS_RANGE_LOG10));
        kernel = KERNEL_SET(randi(numel(KERNEL_SET)));

        foldRMSE = nan(K_FOLD,1);

        for k = 1:K_FOLD

            valCells = trainCells(test(cvp,k));
            trCells  = trainCells(training(cvp,k));

            idxTrFold = ismember(cellTr, trCells);
            idxValFold = ismember(cellTr, valCells);

            XtrFold = Xtr(idxTrFold,:);
            ytrFold = ytr(idxTrFold);

            XvalFold = Xtr(idxValFold,:);
            yvalFold = ytr(idxValFold);

            if numel(unique(ytrFold)) < 2 || isempty(yvalFold)
                continue;
            end

            try
                mdl = fitrsvm(XtrFold, ytrFold, ...
                    'KernelFunction', char(kernel), ...
                    'BoxConstraint', Cval, ...
                    'Epsilon', epsVal, ...
                    'KernelScale', 'auto', ...
                    'Standardize', false);

                yvalPred = predict(mdl, XvalFold);
                foldRMSE(k) = calcRMSE(yvalFold, yvalPred);

            catch
                foldRMSE(k) = nan;
            end
        end

        meanCVRMSE = mean(foldRMSE, 'omitnan');

        if isfinite(meanCVRMSE) && meanCVRMSE < bestCVRMSE
            bestCVRMSE = meanCVRMSE;
            bestParam.C = Cval;
            bestParam.Epsilon = epsVal;
            bestParam.Kernel = kernel;
        end
    end

    fprintf("  [SVR BEST] Kernel=%s, C=%.4g, eps=%.4g, CV_RMSE=%.4f\n", ...
        bestParam.Kernel, bestParam.C, bestParam.Epsilon, bestCVRMSE);

    % Train final SVR on all train/val samples
    mdlSVR = fitrsvm(Xtr, ytr, ...
        'KernelFunction', char(bestParam.Kernel), ...
        'BoxConstraint', bestParam.C, ...
        'Epsilon', bestParam.Epsilon, ...
        'KernelScale', 'auto', ...
        'Standardize', false);

    ypredSVR = predict(mdlSVR, Xte);

    svr_R2   = calcR2(yte, ypredSVR);
    svr_RMSE = calcRMSE(yte, ypredSVR);
    svr_MAE  = calcMAE(yte, ypredSVR);

    fprintf("  [SVR TEST] R2=%.4f, RMSE=%.4f, MAE=%.4f\n", svr_R2, svr_RMSE, svr_MAE);

    %% ---------- MLR baseline ----------

    % MLR can be unstable if features are many and samples are few.
    % Here we use fitlm with standardized features.
    mdlMLR = fitlm(Xtr, ytr);
    ypredMLR = predict(mdlMLR, Xte);

    mlr_R2   = calcR2(yte, ypredMLR);
    mlr_RMSE = calcRMSE(yte, ypredMLR);
    mlr_MAE  = calcMAE(yte, ypredMLR);

    fprintf("  [MLR TEST] R2=%.4f, RMSE=%.4f, MAE=%.4f\n", mlr_R2, mlr_RMSE, mlr_MAE);

    %% ---------- Save results ----------

    Result = [Result; table( ...
        targetName, "SVR", string(bestParam.Kernel), bestParam.C, bestParam.Epsilon, ...
        bestCVRMSE, svr_R2, svr_RMSE, svr_MAE, ...
        string(LOAD_TEST_STR), numel(unique(cellTr)), numel(unique(cellTe)), ...
        'VariableNames', {'Target','Model','Kernel','C','Epsilon','CV_RMSE','Test_R2','Test_RMSE','Test_MAE','TestLoad','NTrainCells','NTestCells'})]; %#ok<AGROW>

    Result = [Result; table( ...
        targetName, "MLR", "", nan, nan, ...
        nan, mlr_R2, mlr_RMSE, mlr_MAE, ...
        string(LOAD_TEST_STR), numel(unique(cellTr)), numel(unique(cellTe)), ...
        'VariableNames', {'Target','Model','Kernel','C','Epsilon','CV_RMSE','Test_R2','Test_RMSE','Test_MAE','TestLoad','NTrainCells','NTestCells'})]; %#ok<AGROW>

    %% ---------- Optional: save prediction table ----------

    PredTbl = table();
    PredTbl.Cell_ID = cellTe;
    PredTbl.Load = repmat(string(LOAD_TEST_STR), numel(yte), 1);
    PredTbl.Target = repmat(targetName, numel(yte), 1);
    PredTbl.YTrue = yte;
    PredTbl.YPred_SVR = ypredSVR;
    PredTbl.YPred_MLR = ypredMLR;

    predVarName = sprintf("Pred_%s_TestLoad_%s", targetName, LOAD_TEST_STR);
    assignin('base', predVarName, PredTbl);
end

disp(" ");
disp("===== FINAL RESULT =====");
disp(Result);

assignin('base', 'Result_CellSplit_Train7Loads_Test1UnseenLoad', Result);

fprintf("\n완료: Train/Val = 7 loads from train cells, Test = unseen load from held-out cells.\n");

%% ===================== HELPER FUNCTIONS =====================

function Tload = getLoadTable(loadName)
    % Expected table name examples:
    %   Tbl_US06_ECM
    %   Tbl_UDDS_ECM
    %
    % Modify here if your table naming convention is different.

    candNames = [
        "Tbl_" + loadName + "_ECM"
        "Tbl_" + loadName
        loadName + "_ECM"
    ];

    Tload = [];
    for i = 1:numel(candNames)
        nm = candNames(i);
        if evalin('base', sprintf('exist(''%s'', ''var'')', nm))
            Tload = evalin('base', nm);
            return;
        end
    end

    error("Cannot find ECM table for load %s. Tried: %s", ...
        loadName, strjoin(candNames, ", "));
end

function Tout = buildOneLoadDataset(Tload, Tlabel, loadName, featNames, targetNames)

    cellVarLoad = findCellVarName(Tload);
    cellVarLabel = findCellVarName(Tlabel);

    Tload.Cell_ID_tmp = string(Tload.(cellVarLoad));
    Tlabel.Cell_ID_tmp = string(Tlabel.(cellVarLabel));

    % Keep only necessary columns
    needLoadVars = ["Cell_ID_tmp", featNames];
    needLabelVars = ["Cell_ID_tmp", targetNames];

    missingFeat = setdiff(featNames, string(Tload.Properties.VariableNames));
    if ~isempty(missingFeat)
        error("Missing feature variables in load table %s: %s", ...
            loadName, strjoin(missingFeat, ", "));
    end

    missingTarget = setdiff(targetNames, string(Tlabel.Properties.VariableNames));
    if ~isempty(missingTarget)
        error("Missing target variables in label table: %s", strjoin(missingTarget, ", "));
    end

    A = Tload(:, needLoadVars);
    B = Tlabel(:, needLabelVars);

    Tout = innerjoin(A, B, 'Keys', 'Cell_ID_tmp');

    Tout.Cell_ID = string(Tout.Cell_ID_tmp);
    Tout.Load = repmat(string(loadName), height(Tout), 1);

    Tout.Cell_ID_tmp = [];

    % Move Cell_ID and Load to front
    Tout = movevars(Tout, ["Cell_ID","Load"], 'Before', 1);
end

function cellVar = findCellVarName(T)

    candidates = ["Cell_ID","CellID","cell_id","cellID","Cell","cell","Name","name"];

    vars = string(T.Properties.VariableNames);

    for c = candidates
        if any(vars == c)
            cellVar = char(c);
            return;
        end
    end

    % Fallback: find variable containing "cell"
    idx = contains(lower(vars), "cell");
    if any(idx)
        cellVar = char(vars(find(idx,1)));
        return;
    end

    error("Cannot identify cell ID variable in table.");
end

function r = calcRMSE(y, yhat)
    y = y(:);
    yhat = yhat(:);
    valid = isfinite(y) & isfinite(yhat);
    r = sqrt(mean((y(valid) - yhat(valid)).^2));
end

function m = calcMAE(y, yhat)
    y = y(:);
    yhat = yhat(:);
    valid = isfinite(y) & isfinite(yhat);
    m = mean(abs(y(valid) - yhat(valid)));
end

function r2 = calcR2(y, yhat)
    y = y(:);
    yhat = yhat(:);
    valid = isfinite(y) & isfinite(yhat);
    y = y(valid);
    yhat = yhat(valid);

    if numel(y) < 2
        r2 = nan;
        return;
    end

    ssRes = sum((y - yhat).^2);
    ssTot = sum((y - mean(y)).^2);

    if ssTot == 0
        r2 = nan;
    else
        r2 = 1 - ssRes / ssTot;
    end
end