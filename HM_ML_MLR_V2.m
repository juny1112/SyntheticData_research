%% ======================================================================
%  2RC(부하별 Tbl_<LOAD>_ECM) + labels → MLR
%
%  [FEATURE STRUCTURE]
%   - sample = (cell, load)
%   - X = 해당 load의 2RC 파라미터(5×SOC_use)
%   - y = 셀 라벨을 load 개수만큼 복제
%
%  [DATA SPLIT BY LOAD]
%   - test load는 사용자가 지정 (LOAD_TEST_STR)
%   - train/val load도 사용자가 지정 가능 (LOAD_TRAINVAL_STR)
%     (비우면 = LOAD_USE - LOAD_TEST)
%
%  [CV: LOAD-LEVEL K-FOLD]
%   - train/val 세트에서 "load 단위"로 K-fold
%     fold validation = 특정 load(들) 전체
%     fold training   = 나머지 load 전체
%   - 최종 모델은 train/val 전체로 재학습 후 test 평가
%
%  NOTE:
%   - train/val load 개수가 7개면 "val=load 1개씩" 형태로 7-fold(LOLO-CV) 가능
% ======================================================================
clear; clc; close all;

%% ── 설정(토글) ---------------------------------------------------------
SOC_use = [70];

% 사용할 주행부하 (데이터 풀)
LOAD_USE_STR = "US06 UDDS HWFET WLTP CITY1 CITY2 HW1 HW2";  % 필요시 수정

% ★★★ LOAD 단위 split을 사용자 지정
LOAD_TEST_STR     = "US06";     % 예: "HW2" 또는 "US06 HW2"
LOAD_TRAINVAL_STR = "";         % 비워두면 자동 = LOAD_USE - LOAD_TEST
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

%% ── 경로/파일 ----------------------------------------------------------
matPath   = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\20degC\2RC_fitting_600s\2RC_results_600s.mat";

save_path = 'G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\ML_MLR';
if ~exist(save_path, 'dir'), mkdir(save_path); end

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

% 실제 mat에 존재하는 것만 남기기 (use set 기준)
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

% train/val/test도 존재 테이블 기준으로 재정리
load_trainval = intersect(load_trainval, load_use, 'stable');
load_test     = intersect(load_test,     load_use, 'stable');

if isempty(load_use), error('선택한 load 중 mat에 존재하는 테이블이 없음'); end
if isempty(load_trainval), error('train/val load이 mat에 존재하지 않음'); end
if isempty(load_test), warning('test load이 비었습니다. (test 평가 스킵)'); end

fprintf('>> (exists) LOAD_USE     : %s\n', strjoin(load_use, ', '));
fprintf('>> (exists) LOAD_TRAINVAL: %s\n', strjoin(load_trainval, ', '));
fprintf('>> (exists) LOAD_TEST    : %s\n', strjoin(load_test, ', '));

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
nFeat = numel(feat2RC_names);

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

fprintf('\n[DATA] Feature names (%d): %s\n', nFeat, strjoin(feat2RC_names, ', '));
fprintf('[DATA] Total samples = %d (cells %d × loads %d)\n', nS, nC, nL_use);

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
% DCIR10s_T20_user   = [1.60;1.41;2.33;1.88;2.06;1.93;1.35;1.53;3.55;0.82;1.52;1.47];
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

% 셀 라벨을 (cell×load)로 확장 (LOAD_USE 기준)
expandY = @(y_cell) repmat(y_cell(:), nL_use, 1);

%% ── 타겟 Map 구성 (모든 y 길이=nS) -------------------------------------
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

%% ── 타겟별: LOAD-level CV(train/val) + 최종학습 + test 평가 -------------
results = struct();

for tt = 1:numel(target_active)
    tname = target_active{tt};
    y_all = target_values(tname);

    % 타겟별 유효 인덱스 (X,y finite + load mask)
    idx_trainval = idx_trainval_base & isfinite(y_all);
    idx_test     = idx_test_base     & isfinite(y_all);

    X_tv = X(idx_trainval, :);
    y_tv = y_all(idx_trainval);
    l_tv = load_id(idx_trainval);

    X_te = X(idx_test, :);
    y_te = y_all(idx_test);

    n_tv = size(X_tv,1);
    n_te = size(X_te,1);

    fprintf('\n[TARGET %s] train/val=%d, test=%d\n', tname, n_tv, n_te);

    if n_tv < nFeat + 1
        warning('  -> train/val 샘플 부족: %d < %d (nFeat+1). 스킵', n_tv, nFeat+1);
        continue;
    end

    %% ---- LOAD-level K-fold CV ----------------------------------------
    % fold 단위는 "train/val에 포함된 load"
    cv_loads = unique(l_tv, 'stable'); % string array (유효샘플 기준)
    nL_tv = numel(cv_loads);

    K = min(K_FOLD, nL_tv);  % load 개수보다 fold가 클 수 없음

    % loads를 K개 fold에 랜덤 분배 (재현성 원하면 rng 고정)
    rng(0);
    perm = randperm(nL_tv);
    fold_id_of_load = zeros(nL_tv,1);
    for i = 1:nL_tv
        fold_id_of_load(perm(i)) = mod(i-1, K) + 1;  % 1..K
    end

    yhat_cv   = nan(n_tv,1);
    fold_R2   = nan(K,1);
    fold_RMSE = nan(K,1);
    fold_MAE  = nan(K,1);

    for k = 1:K
        val_loads = cv_loads(fold_id_of_load == k);

        is_val = ismember(l_tv, val_loads);
        is_tr  = ~is_val;

        if nnz(is_val) < 2 || nnz(is_tr) < nFeat + 1
            warning('  fold %d: train/val 샘플 부족 (tr=%d, val=%d). fold 스킵', k, nnz(is_tr), nnz(is_val));
            continue;
        end

        mdl_k  = fitlm(X_tv(is_tr,:), y_tv(is_tr));
        yhat_k = predict(mdl_k, X_tv(is_val,:));
        yhat_cv(is_val) = yhat_k;

        res_k = y_tv(is_val) - yhat_k;
        fold_RMSE(k) = sqrt(mean(res_k.^2));
        fold_MAE(k)  = mean(abs(res_k));
        fold_R2(k)   = 1 - sum(res_k.^2) / sum( (y_tv(is_val)-mean(y_tv(is_val))).^2 );
    end

    cv_RMSE = mean(fold_RMSE, 'omitnan');
    cv_MAE  = mean(fold_MAE,  'omitnan');
    cv_R2   = mean(fold_R2,   'omitnan');

    fprintf('  [LOAD-CV %d-fold] R2=%.4f, RMSE=%.4f, MAE=%.4f\n', K, cv_R2, cv_RMSE, cv_MAE);

    %% ---- Final model: fit on ALL train/val ----------------------------
    mdl = fitlm(X_tv, y_tv);

    % train/val fit 성능(참고용)
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
    results.(tname).test.y_true = y_te;
    results.(tname).test.y_pred = yhat_te;
    results.(tname).test.residuals = res_te;
    results.(tname).test.metrics = test_metrics;

    results.(tname).feature_names = feat2RC_names;
    results.(tname).SOC_use = SOC_use;
    results.(tname).LOAD_USE_STR = LOAD_USE_STR;
    results.(tname).LOAD_TRAINVAL = load_trainval;
    results.(tname).LOAD_TEST = load_test;

    % 메타(유효 샘플에 대해서만)
    results.(tname).trainval.load_id = l_tv;
    results.(tname).trainval.cell_id = cell_id(idx_trainval);
    results.(tname).test.load_id = load_id(idx_test);
    results.(tname).test.cell_id = cell_id(idx_test);


    %% ---- plot: Final Train/Val + Test in one figure -----------------------
    fig_all = figure('Color','w','Name',sprintf('ALL_%s_true_vs_pred',tname));
    hold on; grid on;

    % train/val (final model prediction on train/val)
    h_tv = scatter(y_tv, yhat_tv, 35, 'filled', ...
        'Marker','o', ...
        'MarkerFaceColor',[0 0 0], ...
        'MarkerEdgeColor',[0 0 0]);

    % test (final model prediction on test)
    h_te = [];
    if n_te >= 2
        h_te = scatter(y_te, yhat_te, 35, 'filled', ...
            'Marker','o', ...
            'MarkerFaceColor',[0.85 0.1 0.1], ...  % test만 색 다르게
            'MarkerEdgeColor',[0.85 0.1 0.1]);
    end

    % 1:1 line (축 범위는 tv+te 합쳐서)
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

    % 제목에 train/test 성능 같이 표기
    if n_te >= 2
        title(sprintf('%s | Train/Val: R2=%.3f RMSE=%.3f | Test: R2=%.3f RMSE=%.3f', ...
            tname, tv_R2, tv_RMSE, test_metrics.R2, test_metrics.RMSE), 'Interpreter','none');
    else
        title(sprintf('%s | Train/Val: R2=%.3f RMSE=%.3f | Test: (none)', ...
            tname, tv_R2, tv_RMSE), 'Interpreter','none');
    end

    axis equal; axis tight;

    % legend
    if isempty(h_te)
        legend(h_tv, {'Train/Val'}, 'Location','best');
    else
        legend([h_tv h_te], {'Train/Val','Test'}, 'Location','best');
    end

    exportgraphics(fig_all, fullfile(save_path, sprintf('ALL_%s_true_vs_pred.png', tname)), 'Resolution', 220);
    savefig(fig_all, fullfile(save_path, sprintf('ALL_%s_true_vs_pred.fig', tname)));

    % 계수 저장
    coef_tbl = mdl.Coefficients;
    coef_tbl.Properties.Description = sprintf('Final MLR coeffs for %s | TrainValLoads=%s | TestLoads=%s', ...
        tname, strjoin(load_trainval, ', '), strjoin(load_test, ', '));
    writetable(coef_tbl, fullfile(save_path, sprintf('MLR_coeffs_FINAL_%s.csv', tname)), 'WriteRowNames', true);
end

%% ── 저장 ---------------------------------------------------------------
save(fullfile(save_path, 'MLR_results_split_LOADCV.mat'), ...
    'results', 'X', 'feat2RC_names', 'cell_names', 'SOC_use', 'pNames_2RC', ...
    'LOAD_USE_STR', 'LOAD_TEST_STR', 'LOAD_TRAINVAL_STR', 'K_FOLD', 'TEMP_list', ...
    'load_id', 'cell_id', 'load_use', 'load_trainval', 'load_test');

disp('완료: LOAD split + train/val LOAD-level K-fold CV + test 평가 결과 저장.');

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
