%% ======================================================================
%  2RC(선택 부하들: Tbl_<LOAD>_ECM) + labels → MLR (Cell-based split + Cell-level K-fold CV)
%
%  [WHAT'S NEW vs 기존]
%   1) test 셀을 사용자가 "ID"로 직접 지정 (예: "01 13")
%   2) split 기준을 load가 아니라 cell 기준으로 수행 (leakage 방지)
%   3) feature에 포함할 주행부하를 사용자가 선택 (예: "US06", "US06 UDDS")
%   4) CV도 cell 단위로 K-fold (기본 5-fold)
%
%  [SAMPLE STRUCTURE]   ★★★ (이번 버전 핵심 변경)
%   - sample = cell   (셀 1개 = 샘플 1개)
%   - X = 선택한 여러 load의 2RC 파라미터를 "옆으로 concat"
%         예) LOAD_FEAT_STR="US06 UDDS", SOC_use=[70]이면
%             X = [US06_R0_70..US06_tau2_70, UDDS_R0_70..UDDS_tau2_70] (총 10개)
%   - y = 셀 라벨 (복제 없음)
%
%  [DATA SPLIT]
%   - test: 사용자가 지정한 TEST_CELL_ID_STR에 매칭되는 "cell(파일)" 전체
%   - train/val: 나머지 cell 전체
%   - CV: train/val cell들을 K-fold로 분할 (fold마다 cell 통째로 val)
%
%  NOTE
%   - Tbl_Load_ECM 구조체가 mat 파일에 있어야 함.
%     (너가 만든 2RC_results_600s.mat 또는 2RC_results_XXXs.mat)
% ======================================================================
clear; clc; close all;

%% ── 설정(토글) ---------------------------------------------------------
SOC_use = [70];                 % 예: [70] 또는 [70 50]
TEMP_list = [20];               % 온도별 타겟 키 생성용

% feature에 사용할 주행부하 선택 (여기서만 바꾸면 됨)
LOAD_FEAT_STR = "US06 UDDS";    % 예: "US06" 또는 "US06 UDDS WLTP"

% (중요) test 셀은 ID로 직접 지정
TEST_CELL_ID_STR = "04";     % 예: "01 13" / "1 13" / "01,13" / "01; 13"

% CV fold (cell 기준)
K_FOLD = 5;

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

%% ── 경로/파일 ----------------------------------------------------------
matPath   = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\20degC\2RC_fitting_600s\2RC_results_600s.mat";

save_path = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\ML_MLR_cellSplit_concatLoads";
if ~exist(save_path, 'dir'), mkdir(save_path); end

%% ── 주행부하 파싱/검증 -------------------------------------------------
loadNames_all = {'US06','UDDS','HWFET','WLTP','CITY1','CITY2','HW1','HW2'};
all_upper = upper(string(loadNames_all));

load_feat = upper(parseLoadList(LOAD_FEAT_STR));
ok = ismember(string(load_feat), all_upper);
if any(~ok)
    warning('알 수 없는 주행부하 입력이 있어 제외: %s', strjoin(load_feat(~ok), ', '));
end
load_feat = load_feat(ok);
load_feat = toStdCase(load_feat, loadNames_all);

if isempty(load_feat), error('LOAD_FEAT_STR에서 유효 부하가 없습니다.'); end
fprintf('>> LOAD_FEAT (features): %s\n', strjoin(load_feat, ', '));

%% ── 2RC 테이블 로드 ----------------------------------------------------
S = load(matPath);
getTblECM = @(loadName) localGetTblECM(S, loadName);

% 실제 mat에 존재하는 feature-load만 남기기
load_feat_ok = {};
for i = 1:numel(load_feat)
    try
        Ttmp = getTblECM(load_feat{i});
        if istable(Ttmp), load_feat_ok{end+1} = load_feat{i}; end %#ok<AGROW>
    catch
        warning('mat 파일에 %s ECM 테이블이 없어 제외합니다.', load_feat{i});
    end
end
load_feat = load_feat_ok;

if isempty(load_feat), error('선택한 feature load 중 mat에 존재하는 테이블이 없음'); end
fprintf('>> (exists) LOAD_FEAT: %s\n', strjoin(load_feat, ', '));

%% ── 셀 이름 정합(LOAD_FEAT 내 공통 셀) -----------------------------------
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

%% ── X 구성: sample = cell (load들을 옆으로 concat) -----------------------
pNames_2RC = {'R0','R1','R2','tau1','tau2'};

% feature name: <LOAD>_<param>_<SOC>
feat_names = {};
for li = 1:numel(load_feat)
    L = load_feat{li};
    for s = SOC_use(:).'
        for pi = 1:numel(pNames_2RC)
            feat_names{end+1} = sprintf('%s_%s_%d', L, pNames_2RC{pi}, s); %#ok<AGROW>
        end
    end
end
nFeat = numel(feat_names);

X = nan(nC, nFeat);
cell_id = string(cell_names(:));   % sample=cell

col = 0;
for li = 1:numel(load_feat)
    L = load_feat{li};
    TblL = getTblECM(L);
    TblL = TblL(cell_names, :);  % 공통 셀 순서로 정렬
    vnames = TblL.Properties.VariableNames;

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
                X(:, col) = TblL{:, varName};
            else
                X(:, col) = NaN;
            end
        end
    end
end

base_valid_X = all(isfinite(X), 2);

fprintf('\n[DATA] Feature names (%d): %s\n', nFeat, strjoin(feat_names, ', '));
fprintf('[DATA] Total samples = %d (cells %d)\n', nC, nC);

%% ── 셀 라벨 입력 (nC 기준) ----------------------------------------------
% ====== 여기 값은 네가 채우는 구간 ======
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
% ======================================

% 길이 보정
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

%% ── 타겟 Map 구성 (y 길이 = nC) ----------------------------------------
target_values = containers.Map();

target_values('QC2')              = QC2_user(:);
target_values('QC40')             = QC40_user(:);
target_values('Rcharg')           = Rcharg_user(:);
target_values('Rcharg_80_90_avg') = Rcharg_80_90_avg_user(:);
target_values('R1s')              = R1s_user(:);

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

    target_values(sprintf('DCIR_1s_T%d', t))           = d1_cell;
    target_values(sprintf('DCIR_10s_T%d', t))          = d10_cell;
    target_values(sprintf('DCIR_delta_10s_1s_T%d', t)) = dDCIR_cell;
    target_values(sprintf('Power_T%d', t))             = pw_cell;
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

%% ── Cell-based split: TEST cell IDs (user 지정) -------------------------
[test_cells, trainval_cells] = pickTestCellsByID(cell_names, TEST_CELL_ID_STR);

fprintf('>> TEST cells (%d): %s\n', numel(test_cells), strjoin(test_cells, ', '));
fprintf('>> TRAIN/VAL cells (%d)\n', numel(trainval_cells));

if numel(test_cells) < 1
    error('지정한 TEST_CELL_ID_STR에서 test 셀이 0개로 선택됨. 파일명/ID 패턴 확인 필요');
end

if numel(trainval_cells) < K_FOLD
    warning('train/val 셀 수(%d) < K_FOLD(%d). K를 줄입니다.', numel(trainval_cells), K_FOLD);
    K_FOLD = numel(trainval_cells);
end

is_trainval_cell = ismember(cell_id, string(trainval_cells));
is_test_cell     = ismember(cell_id, string(test_cells));

idx_trainval_base = base_valid_X & is_trainval_cell;
idx_test_base     = base_valid_X & is_test_cell;

fprintf('[SPLIT] train/val base samples = %d\n', nnz(idx_trainval_base));
fprintf('[SPLIT] test      base samples = %d\n', nnz(idx_test_base));

%% ── 타겟별: Cell-level CV(train/val) + 최종학습 + test 평가 -------------
results = struct();

for tt = 1:numel(target_active)
    tname = target_active{tt};
    y_all = target_values(tname);

    idx_trainval = idx_trainval_base & isfinite(y_all);
    idx_test     = idx_test_base     & isfinite(y_all);

    X_tv = X(idx_trainval, :);
    y_tv = y_all(idx_trainval);
    c_tv = cell_id(idx_trainval);      % cell ID (string)

    X_te = X(idx_test, :);
    y_te = y_all(idx_test);

    n_tv = size(X_tv,1);
    n_te = size(X_te,1);

    fprintf('\n[TARGET %s] train/val=%d, test=%d\n', tname, n_tv, n_te);

    if n_tv < nFeat + 1
        warning('  -> train/val 샘플 부족: %d < %d (nFeat+1). 스킵', n_tv, nFeat+1);
        continue;
    end

    %% ---- Cell-level K-fold CV ----------------------------------------
    cv_cells = unique(c_tv, 'stable'); % 이젠 c_tv가 이미 cell 단위 row라 거의 동일하지만 유지
    nC_tv = numel(cv_cells);
    K = min(K_FOLD, nC_tv);

    rng(0);
    perm = randperm(nC_tv);
    fold_id_of_cell = zeros(nC_tv,1);
    for i = 1:nC_tv
        fold_id_of_cell(perm(i)) = mod(i-1, K) + 1;
    end

    fold_R2   = nan(K,1);
    fold_RMSE = nan(K,1);
    fold_MAE  = nan(K,1);

    for k = 1:K
        val_cells_k = cv_cells(fold_id_of_cell == k);

        is_val = ismember(c_tv, val_cells_k);
        is_tr  = ~is_val;

        if nnz(is_val) < 2 || nnz(is_tr) < nFeat + 1
            warning('  fold %d: train/val 샘플 부족 (tr=%d, val=%d). fold 스킵', k, nnz(is_tr), nnz(is_val));
            continue;
        end

        mdl_k  = fitlm(X_tv(is_tr,:), y_tv(is_tr));
        yhat_k = predict(mdl_k, X_tv(is_val,:));

        res_k = y_tv(is_val) - yhat_k;
        fold_RMSE(k) = sqrt(mean(res_k.^2));
        fold_MAE(k)  = mean(abs(res_k));
        fold_R2(k)   = 1 - sum(res_k.^2) / sum( (y_tv(is_val)-mean(y_tv(is_val))).^2 );
    end

    cv_RMSE = mean(fold_RMSE, 'omitnan');
    cv_MAE  = mean(fold_MAE,  'omitnan');
    cv_R2   = mean(fold_R2,   'omitnan');

    fprintf('  [CELL-CV %d-fold] R2=%.4f, RMSE=%.4f, MAE=%.4f\n', K, cv_R2, cv_RMSE, cv_MAE);

    %% ---- Final model: fit on ALL train/val ----------------------------
    mdl = fitlm(X_tv, y_tv);

    yhat_tv = predict(mdl, X_tv);
    res_tv  = y_tv - yhat_tv;
    tv_RMSE = sqrt(mean(res_tv.^2));
    tv_MAE  = mean(abs(res_tv));
    tv_R2   = mdl.Rsquared.Ordinary;

    %% ---- Test evaluation ----------------------------------------------
    test_metrics = struct('R2',nan,'RMSE',nan,'MAE',nan);
    yhat_te = [];
    res_te  = [];
    if n_te >= 2
        yhat_te = predict(mdl, X_te);
        res_te  = y_te - yhat_te;

        test_metrics.RMSE = sqrt(mean(res_te.^2));
        test_metrics.MAE  = mean(abs(res_te));
        test_metrics.R2   = 1 - sum(res_te.^2) / sum( (y_te-mean(y_te)).^2 );

        fprintf('  [TEST] R2=%.4f, RMSE=%.4f, MAE=%.4f\n', ...
            test_metrics.R2, test_metrics.RMSE, test_metrics.MAE);
    else
        fprintf('  [TEST] test 샘플 부족/없음 → test 평가 스킵\n');
    end

    %% ---- Save results struct -----------------------------------------
    results.(tname).mdl = mdl;

    results.(tname).trainval.idx_global = idx_trainval;
    results.(tname).trainval.X = X_tv;
    results.(tname).trainval.y_true = y_tv;
    results.(tname).trainval.y_pred = yhat_tv;
    results.(tname).trainval.residuals = res_tv;
    results.(tname).trainval.R2 = tv_R2;
    results.(tname).trainval.RMSE = tv_RMSE;
    results.(tname).trainval.MAE = tv_MAE;

    results.(tname).cv.K = K;
    results.(tname).cv.cells = cv_cells;
    results.(tname).cv.fold_id_of_cell = fold_id_of_cell;
    results.(tname).cv.fold_R2 = fold_R2;
    results.(tname).cv.fold_RMSE = fold_RMSE;
    results.(tname).cv.fold_MAE = fold_MAE;
    results.(tname).cv.R2_mean = cv_R2;
    results.(tname).cv.RMSE_mean = cv_RMSE;
    results.(tname).cv.MAE_mean = cv_MAE;

    results.(tname).test.idx_global = idx_test;
    results.(tname).test.X = X_te;
    results.(tname).test.y_true = y_te;
    results.(tname).test.y_pred = yhat_te;
    results.(tname).test.residuals = res_te;
    results.(tname).test.metrics = test_metrics;

    results.(tname).feature_names = feat_names;
    results.(tname).SOC_use = SOC_use;
    results.(tname).LOAD_FEAT_STR = LOAD_FEAT_STR;
    results.(tname).TEST_CELL_ID_STR = TEST_CELL_ID_STR;
    results.(tname).K_FOLD = K_FOLD;

    results.(tname).trainval.cell_id = c_tv;
    results.(tname).test.cell_id = cell_id(idx_test);

    %% ---- plot: Final Train/Val + Test in one figure -------------------
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
        title(sprintf('%s | Train/Val: R2=%.3f RMSE=%.3f | CV: R2=%.3f RMSE=%.3f | Test: R2=%.3f RMSE=%.3f', ...
            tname, tv_R2, tv_RMSE, cv_R2, cv_RMSE, test_metrics.R2, test_metrics.RMSE), 'Interpreter','none');
    else
        title(sprintf('%s | Train/Val: R2=%.3f RMSE=%.3f | CV: R2=%.3f RMSE=%.3f | Test: (none)', ...
            tname, tv_R2, tv_RMSE, cv_R2, cv_RMSE), 'Interpreter','none');
    end

    axis equal; axis tight;

    if isempty(h_te)
        legend(h_tv, {'Train/Val'}, 'Location','best');
    else
        legend([h_tv h_te], {'Train/Val','Test'}, 'Location','best');
    end

    exportgraphics(fig_all, fullfile(save_path, sprintf('ALL_%s_true_vs_pred.png', tname)), 'Resolution', 220);
    savefig(fig_all, fullfile(save_path, sprintf('ALL_%s_true_vs_pred.fig', tname)));

    % 계수 저장
    coef_tbl = mdl.Coefficients;
    coef_tbl.Properties.Description = sprintf('Final MLR coeffs for %s | LOAD_FEAT=%s | TEST_ID=%s', ...
        tname, LOAD_FEAT_STR, TEST_CELL_ID_STR);
    writetable(coef_tbl, fullfile(save_path, sprintf('MLR_coeffs_FINAL_%s.csv', tname)), 'WriteRowNames', true);
end

%% ── 저장 ---------------------------------------------------------------
save(fullfile(save_path, 'MLR_results_cellSplit_concatLoads.mat'), ...
    'results', 'X', 'feat_names', 'cell_names', 'SOC_use', 'pNames_2RC', ...
    'LOAD_FEAT_STR', 'TEST_CELL_ID_STR', 'K_FOLD', 'TEMP_list', ...
    'cell_id', 'load_feat');

disp('완료: CELL split + (concat loads as features) + CELL-level K-fold CV + test 평가 결과 저장.');

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

test_cells = cellStr(is_test);
trainval_cells = cellStr(~is_test);

test_cells = unique(test_cells, 'stable');
trainval_cells = unique(trainval_cells, 'stable');
end