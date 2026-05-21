%% ======================================================================
%  SVR: 2RC(부하별 Tbl_<LOAD>_ECM) + labels
%  [MODIFIED] feature = R0, R1, R2, C1, C2
%
%  C1 = tau1 / R1_mOhm   [kF]
%  C2 = tau2 / R2_mOhm   [kF]
%
%  - sample = (cell, load)
%  - X = 해당 load의 2RC 파라미터(5×SOC_use) = [R0, R1, R2, C1, C2]
%  - y = 셀 라벨을 load 개수만큼 복제
%
%  [SPLIT]
%   - test load: LOAD_TEST_STR
%   - train/val load: LOAD_TRAINVAL_STR (비우면 = LOAD_USE - LOAD_TEST)
%
%  [CV]  (LOAD-level)
%   - train/val 내부에서 load 단위 K-fold
%   - final model: train/val 전체로 재학습
%   - test: 최종 1회 평가
%
%  [TUNING]
%   - TUNE_MODE="fixed": 지정 hyperparam 1회
%   - TUNE_MODE="random": Random search로 (C, epsilon, kernel scale, kernel) 탐색
%% ======================================================================
clear; clc; close all;

% (A PATCH) Text Interpreter 에러 방지
set(groot,'defaultTextInterpreter','none');
set(groot,'defaultAxesTickLabelInterpreter','none');
set(groot,'defaultLegendInterpreter','none');

%% ── 설정 --------------------------------------------------------------
SOC_use = [70];

LOAD_USE_STR = "US06 UDDS HWFET WLTP CITY1 CITY2 HW1 HW2";

LOAD_TEST_STR     = "US06";
LOAD_TRAINVAL_STR = "";

K_FOLD = 7;
TEMP_list = [20];

% ---- 타겟 토글 ----
PRED_QC2         = true;
PRED_QC40        = true;
PRED_RCHARG      = false;
PRED_RCHARG_8090 = false;
PRED_R1S         = false;

PRED_DCIR1S_BYTEMP  = false;
PRED_DCIR10S_BYTEMP = true;
PRED_DDELTA_BYTEMP  = false;
PRED_POWER_BYTEMP   = true;

% ---- SVR 기본/튜닝 ----
SVR_STANDARDIZE = true;

TUNE_MODE = "random";      % "fixed" | "random"
FIXED_KERNEL      = 'gaussian';
FIXED_C           = 10;
FIXED_EPSILON     = 0.05;
FIXED_KERNELSCALE = 'auto';

RAND_N_TRIALS = 1000;
RAND_KERNEL_SET = {'linear'}; %'linear', 
RAND_C_RANGE_LOG10        = [-3, 1];
RAND_EPS_RANGE_LOG10      = [-4, 0];
RAND_KS_RANGE_LOG10       = [-3, 2];
RAND_USE_AUTO_KERNELSCALE = true;

EARLYSTOP_ENABLE  = false;
EARLYSTOP_PATIENCE = 15;
EARLYSTOP_MIN_IMPROVE = 1e-4;

TUNE_METRIC = "RMSE";   % "RMSE" | "MAE"

%% ── 경로 --------------------------------------------------------------
matPath   = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\20degC\2RC_fitting_600s\2RC_results_600s.mat";
save_path = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\ML_SVR_C";
if ~exist(save_path, 'dir'), mkdir(save_path); end

%% ── load 파싱/검증 -----------------------------------------------------
loadNames_all = {'US06','UDDS','HWFET','WLTP','CITY1','CITY2','HW1','HW2'};
all_upper = upper(string(loadNames_all));

load_use = upper(parseLoadList(LOAD_USE_STR));
load_use = load_use(ismember(string(load_use), all_upper));
load_use = toStdCase(load_use, loadNames_all);
if isempty(load_use), error('LOAD_USE_STR 유효 load 없음'); end
fprintf('>> LOAD_USE: %s\n', strjoin(load_use, ', '));

load_test = upper(parseLoadList(LOAD_TEST_STR));
load_test = load_test(ismember(string(load_test), all_upper));
load_test = toStdCase(load_test, loadNames_all);
fprintf('>> LOAD_TEST (requested): %s\n', strjoin(load_test, ', '));

load_trainval_user = upper(parseLoadList(LOAD_TRAINVAL_STR));
load_trainval_user = load_trainval_user(ismember(string(load_trainval_user), all_upper));
load_trainval_user = toStdCase(load_trainval_user, loadNames_all);

if isempty(load_trainval_user)
    load_trainval = setdiff(load_use, load_test, 'stable');
else
    load_trainval = intersect(load_use, load_trainval_user, 'stable');
    load_trainval = setdiff(load_trainval, load_test, 'stable');
end
if isempty(load_trainval), error('train/val load 비었음. split 설정 확인'); end
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
        warning('mat에 %s ECM 테이블 없음 → 제외', load_use{i});
    end
end
load_use = load_use_ok;

load_trainval = intersect(load_trainval, load_use, 'stable');
load_test     = intersect(load_test,     load_use, 'stable');

if isempty(load_use),      error('LOAD_USE 중 mat에 존재하는 테이블이 없음'); end
if isempty(load_trainval), error('train/val load이 mat에 존재하지 않음'); end
if isempty(load_test),     warning('test load이 비었습니다(평가 스킵 가능)'); end

fprintf('>> (exists) LOAD_USE: %s\n', strjoin(load_use, ', '));
fprintf('>> (exists) TRAINVAL: %s\n', strjoin(load_trainval, ', '));
fprintf('>> (exists) TEST    : %s\n', strjoin(load_test, ', '));

%% ── 공통 셀(RowNames) 교집합 ------------------------------------------
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
fprintf('>> 공통 셀 개수: %d\n', nC);

%% ── X 구성: sample=(cell,load) -----------------------------------------
% feature = [R0, R1, R2, C1, C2]
feat_base_names = {'R0','R1','R2','C1','C2'};

feat_names = {};
for s = SOC_use(:).'
    for pi = 1:numel(feat_base_names)
        feat_names{end+1} = sprintf('%s_%d', feat_base_names{pi}, s); %#ok<AGROW>
    end
end
nFeat = numel(feat_names);

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

    for ci = 1:nC
        row = row + 1;
        load_id(row) = string(L);
        cell_id(row) = string(cell_names{ci});

        col = 0;
        for s = SOC_use(:).'
            vn_R0   = sprintf('SOC%d_R0_mOhm', s);
            vn_R1   = sprintf('SOC%d_R1_mOhm', s);
            vn_R2   = sprintf('SOC%d_R2_mOhm', s);
            vn_tau1 = sprintf('SOC%d_tau1', s);
            vn_tau2 = sprintf('SOC%d_tau2', s);

            if ismember(vn_R0, vnames),   R0v   = TblL{ci, vn_R0};   else, R0v   = NaN; end
            if ismember(vn_R1, vnames),   R1v   = TblL{ci, vn_R1};   else, R1v   = NaN; end
            if ismember(vn_R2, vnames),   R2v   = TblL{ci, vn_R2};   else, R2v   = NaN; end
            if ismember(vn_tau1, vnames), tau1v = TblL{ci, vn_tau1}; else, tau1v = NaN; end
            if ismember(vn_tau2, vnames), tau2v = TblL{ci, vn_tau2}; else, tau2v = NaN; end

            % C1, C2 계산 (kF)
            if isfinite(R1v) && R1v > 0 && isfinite(tau1v)
                C1v = tau1v / R1v;
            else
                C1v = NaN;
            end

            if isfinite(R2v) && R2v > 0 && isfinite(tau2v)
                C2v = tau2v / R2v;
            else
                C2v = NaN;
            end

            vals = [R0v, R1v, R2v, C1v, C2v];

            for k = 1:numel(vals)
                col = col + 1;
                X(row, col) = vals(k);
            end
        end
    end
end

base_valid_X = all(isfinite(X), 2);
fprintf('[DATA] samples=%d (cells %d × loads %d), features=%d\n', nS, nC, nL_use, nFeat);

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

expandY = @(y_cell) repmat(y_cell(:), nL_use, 1);

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

%% ── LOAD split 마스크 --------------------------------------------------
is_trainval_load = ismember(load_id, string(load_trainval));
is_test_load     = ismember(load_id, string(load_test));

idx_trainval_base = base_valid_X & is_trainval_load;
idx_test_base     = base_valid_X & is_test_load;

fprintf('[SPLIT] train/val base samples=%d, test base samples=%d\n', nnz(idx_trainval_base), nnz(idx_test_base));

%% ── 타겟별: (튜닝 포함) LOAD-CV + FINAL + TEST -------------------------
results = struct();

for tt = 1:numel(target_active)
    tname = target_active{tt};
    y_all = target_values(tname);

    idx_trainval = idx_trainval_base & isfinite(y_all);
    idx_test     = idx_test_base     & isfinite(y_all);

    X_tv = X(idx_trainval,:);
    y_tv = y_all(idx_trainval);
    l_tv = load_id(idx_trainval);

    X_te = X(idx_test,:);
    y_te = y_all(idx_test);

    fprintf('\n[TARGET %s] train/val=%d, test=%d\n', tname, size(X_tv,1), size(X_te,1));

    if size(X_tv,1) < nFeat + 1
        warning('  train/val 샘플 부족 → skip');
        continue;
    end

    cv_loads = unique(l_tv, 'stable');
    nL_tv = numel(cv_loads);
    if nL_tv < 2
        warning('  train/val load가 1개뿐 → CV 불가, skip');
        continue;
    end

    K = min(K_FOLD, nL_tv);

    rng(0);
    perm = randperm(nL_tv);
    fold_id_of_load = zeros(nL_tv,1);
    for i = 1:nL_tv
        fold_id_of_load(perm(i)) = mod(i-1, K) + 1;
    end

    cand = [];
    if TUNE_MODE == "fixed"
        cand = struct('Kernel',{FIXED_KERNEL}, ...
                      'C',FIXED_C, ...
                      'Epsilon',FIXED_EPSILON, ...
                      'KernelScale',{FIXED_KERNELSCALE});
    else
        cand = makeRandomCandidates(RAND_N_TRIALS, RAND_KERNEL_SET, ...
            RAND_C_RANGE_LOG10, RAND_EPS_RANGE_LOG10, RAND_KS_RANGE_LOG10, RAND_USE_AUTO_KERNELSCALE);
    end

    bestScore = inf;
    bestCand = cand(1);
    bestDetail = struct();
    noImprove = 0;

    for ci = 1:numel(cand)
        c0 = cand(ci);

        [cvScore, cvDetail] = evalLoadCV_SVR( ...
            X_tv, y_tv, l_tv, cv_loads, fold_id_of_load, K, ...
            c0.Kernel, c0.C, c0.Epsilon, c0.KernelScale, SVR_STANDARDIZE, TUNE_METRIC, nFeat);

        if ~isfinite(cvScore), continue; end

        if cvScore < bestScore - EARLYSTOP_MIN_IMPROVE
            bestScore = cvScore;
            bestCand = c0;
            bestDetail = cvDetail;
            noImprove = 0;
        else
            noImprove = noImprove + 1;
        end

        if EARLYSTOP_ENABLE && TUNE_MODE=="random" && noImprove >= EARLYSTOP_PATIENCE
            fprintf('  [EarlyStop] %d trials no improvement → stop search (best %s=%.4f)\n', ...
                EARLYSTOP_PATIENCE, TUNE_METRIC, bestScore);
            break;
        end
    end

    fprintf('  [BEST by LOAD-CV] Kernel=%s, C=%.4g, eps=%.4g, KS=%s, %s=%.4f\n', ...
        bestCand.Kernel, bestCand.C, bestCand.Epsilon, ks2str(bestCand.KernelScale), TUNE_METRIC, bestScore);

    mdl = fitrsvm(X_tv, y_tv, ...
        'KernelFunction', bestCand.Kernel, ...
        'Standardize', SVR_STANDARDIZE, ...
        'BoxConstraint', bestCand.C, ...
        'Epsilon', bestCand.Epsilon, ...
        'KernelScale', bestCand.KernelScale);

    yhat_tv = predict(mdl, X_tv);
    res_tv = y_tv - yhat_tv;

    tv.RMSE = sqrt(mean(res_tv.^2));
    tv.MAE  = mean(abs(res_tv));
    tv.R2   = calcR2(y_tv, yhat_tv);

    te = struct('RMSE',nan,'MAE',nan,'R2',nan);
    yhat_te = [];
    res_te  = [];

    if size(X_te,1) >= 2
        yhat_te = predict(mdl, X_te);
        res_te = y_te - yhat_te;
        te.RMSE = sqrt(mean(res_te.^2));
        te.MAE  = mean(abs(res_te));
        te.R2   = calcR2(y_te, yhat_te);
        fprintf('  [TEST] R2=%.4f RMSE=%.4f MAE=%.4f\n', te.R2, te.RMSE, te.MAE);
    else
        fprintf('  [TEST] test 샘플 부족/없음 → 스킵\n');
    end

    results.(tname).best = bestCand;
    results.(tname).cv = bestDetail;
    results.(tname).trainval.metrics = tv;
    results.(tname).test.metrics = te;

    results.(tname).mdl = mdl;
    results.(tname).trainval.y_true = y_tv;
    results.(tname).trainval.y_pred = yhat_tv;
    results.(tname).trainval.load_id = l_tv;

    results.(tname).test.y_true = y_te;
    results.(tname).test.y_pred = yhat_te;
    results.(tname).test.load_id = load_id(idx_test);

    fig_all = figure('Color','w','Name',sprintf('SVR_ALL_%s',tname));
    hold on; grid on;

    h_tv = scatter(y_tv, yhat_tv, 30, 'filled', ...
        'Marker', 'o', ...
        'MarkerFaceColor', [0 0 0], ...
        'MarkerEdgeColor', [0 0 0]);

    h_te = [];
    if numel(yhat_te) >= 2
        h_te = scatter(y_te, yhat_te, 30, 'filled', ...
            'Marker', 'o', ...
            'MarkerFaceColor', [0.85 0.1 0.1], ...
            'MarkerEdgeColor', [0.85 0.1 0.1]);
    end

    all_true = y_tv;
    all_pred = yhat_tv;
    if numel(yhat_te) >= 2
        all_true = [all_true; y_te];
        all_pred = [all_pred; yhat_te];
    end
    minv = min([all_true; all_pred]);
    maxv = max([all_true; all_pred]);
    plot([minv maxv],[minv maxv],'k--','LineWidth',1.2);

    xlabel(sprintf('True %s', tname));
    ylabel(sprintf('Pred %s', tname));

    if numel(yhat_te) >= 2
        title(sprintf('%s | Train/Val: R2=%.3f RMSE=%.3f | Test: R2=%.3f RMSE=%.3f', ...
            tname, tv.R2, tv.RMSE, te.R2, te.RMSE));
    else
        title(sprintf('%s | Train/Val: R2=%.3f RMSE=%.3f | Test: (none)', ...
            tname, tv.R2, tv.RMSE));
    end

    axis equal;
    axis tight;

    if isempty(h_te)
        legend(h_tv, {'Train/Val'}, 'Location','best');
    else
        legend([h_tv h_te], {'Train/Val','Test'}, 'Location','best');
    end

    exportgraphics(fig_all, fullfile(save_path, sprintf('SVR_ALL_%s.png', tname)), 'Resolution', 220);
    savefig(fig_all, fullfile(save_path, sprintf('SVR_ALL_%s.fig', tname)));
end

%% ── 저장 --------------------------------------------------------------
save(fullfile(save_path, 'SVR_results_split_LOADCV_Cfeat.mat'), ...
    'results', 'X', 'feat_names', 'cell_names', 'SOC_use', ...
    'LOAD_USE_STR', 'LOAD_TEST_STR', 'LOAD_TRAINVAL_STR', 'K_FOLD', 'TEMP_list', ...
    'load_id', 'cell_id', 'load_use', 'load_trainval', 'load_test', ...
    'TUNE_MODE','RAND_N_TRIALS','RAND_KERNEL_SET','TUNE_METRIC', ...
    'SVR_STANDARDIZE','EARLYSTOP_ENABLE','EARLYSTOP_PATIENCE','EARLYSTOP_MIN_IMPROVE');

disp('완료: SVR C1/C2 feature 버전 저장');

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

function s = ks2str(ks)
    if ischar(ks) || isstring(ks)
        s = char(ks);
    else
        s = sprintf('%.4g', ks);
    end
end

function cand = makeRandomCandidates(N, kernelSet, logC, logEps, logKS, allowAutoKS)
    cand = repmat(struct('Kernel','gaussian','C',1,'Epsilon',0.1,'KernelScale','auto'), N, 1);
    for i = 1:N
        ker = kernelSet{randi(numel(kernelSet))};
        C   = 10^(logC(1) + (logC(2)-logC(1))*rand());
        eps = 10^(logEps(1) + (logEps(2)-logEps(1))*rand());

        if strcmpi(ker,'gaussian')
            if allowAutoKS && rand() < 0.3
                KS = 'auto';
            else
                KS = 10^(logKS(1) + (logKS(2)-logKS(1))*rand());
            end
        else
            KS = 'auto';
        end

        cand(i).Kernel = ker;
        cand(i).C = C;
        cand(i).Epsilon = eps;
        cand(i).KernelScale = KS;
    end
end

function [score, detail] = evalLoadCV_SVR(X, y, l, cv_loads, fold_id_of_load, K, kernel, C, eps0, ks, doStd, metric, nFeat)
    n = size(X,1);
    yhat = nan(n,1);
    fold_RMSE = nan(K,1);
    fold_MAE  = nan(K,1);
    fold_R2   = nan(K,1);

    for k = 1:K
        val_loads = cv_loads(fold_id_of_load == k);
        is_val = ismember(l, val_loads);
        is_tr  = ~is_val;

        if nnz(is_val) < 2 || nnz(is_tr) < nFeat + 1
            continue;
        end

        try
            mdl = fitrsvm(X(is_tr,:), y(is_tr), ...
                'KernelFunction', kernel, ...
                'Standardize', doStd, ...
                'BoxConstraint', C, ...
                'Epsilon', eps0, ...
                'KernelScale', ks);

            yhat_k = predict(mdl, X(is_val,:));
            yhat(is_val) = yhat_k;

            res = y(is_val) - yhat_k;
            fold_RMSE(k) = sqrt(mean(res.^2));
            fold_MAE(k)  = mean(abs(res));
            fold_R2(k)   = calcR2(y(is_val), yhat_k);
        catch
            continue;
        end
    end

    detail = struct();
    detail.K = K;
    detail.kernel = kernel;
    detail.C = C;
    detail.epsilon = eps0;
    detail.kernelscale = ks;
    detail.y_pred = yhat;
    detail.fold_RMSE = fold_RMSE;
    detail.fold_MAE  = fold_MAE;
    detail.fold_R2   = fold_R2;
    detail.RMSE_mean = mean(fold_RMSE,'omitnan');
    detail.MAE_mean  = mean(fold_MAE,'omitnan');
    detail.R2_mean   = mean(fold_R2,'omitnan');

    if metric == "MAE"
        score = detail.MAE_mean;
    else
        score = detail.RMSE_mean;
    end
end