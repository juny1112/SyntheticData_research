%% ======================================================================
%  2RC(부하별 Tbl_<LOAD>_ECM) + labels → MLR
%
%  [FEATURE STRUCTURE]
%   - sample = (cell, load)
%   - X = 해당 load의 2RC 파라미터(5×SOC_use)
%   - y = 셀 라벨을 load 개수만큼 복제
%
%  [MODE: TRAIN ALL + EVAL US06 SLICE]
%   - Train: LOAD_USE 전체 (US06 포함 8종) 모든 유효 샘플로 1회 학습
%   - CV: OFF
%   - Eval: US06 load 샘플만 슬라이스하여 성능 출력
%     (주의: US06도 Train에 포함되므로 hold-out test가 아님 / in-sample slice 평가)
% ======================================================================
clear; clc; close all;

%% ── 설정(토글) ---------------------------------------------------------
SOC_use = [70];

% 사용할 주행부하 (데이터 풀)
LOAD_USE_STR = "US06 UDDS HWFET WLTP CITY1 CITY2 HW1 HW2";  % 필요시 수정

% 평가할 부하(슬라이스 평가)
EVAL_LOAD_STR = "US06";  % <- 여기만 바꾸면 됨

% CV OFF (본 스크립트에서는 사용하지 않음)
K_FOLD = NaN;

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

eval_load = upper(parseLoadList(EVAL_LOAD_STR));
eval_load = eval_load(ismember(string(eval_load), all_upper));
eval_load = toStdCase(eval_load, loadNames_all);
if isempty(eval_load)
    error('EVAL_LOAD_STR에서 유효 부하가 없습니다.');
end
fprintf('>> EVAL_LOAD: %s\n', strjoin(eval_load, ', '));

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

% eval_load도 존재 테이블 기준으로 재정리
eval_load = intersect(eval_load, load_use, 'stable');

if isempty(load_use), error('선택한 load 중 mat에 존재하는 테이블이 없음'); end
if isempty(eval_load), error('EVAL_LOAD가 mat에 존재하지 않음'); end

fprintf('>> (exists) LOAD_USE : %s\n', strjoin(load_use, ', '));
fprintf('>> (exists) EVAL_LOAD: %s\n', strjoin(eval_load, ', '));

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
fprintf('[DATA] Valid X samples = %d\n', nnz(base_valid_X));

%% ── 셀 라벨 입력 (nC 기준) ----------------------------------------------
% ====== 여기 값은 네가 채우는 구간 ======
QC2_user  = [56.14;55.93;52.55;50.52;52.13;48.53;56.34;55.15;39.89;55.63;55.57;56.86];
QC40_user = [57.49;57.57;54.00;52.22;53.45;51.28;57.91;56.51;42.14;57.27;57.18;58.40];

% NOTE: 아래 Rcharg_user는 길이가 nC와 안 맞아 보이면 ensureLength가 보정해줌
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

%% ── 학습/평가 마스크 ----------------------------------------------------
% 학습: 전체 유효 샘플 (8종, US06 포함)
idx_train_base = base_valid_X;

% 평가: US06만 슬라이스 (US06도 train에 포함됨 → in-sample slice)
idx_eval_base  = base_valid_X & ismember(load_id, string(eval_load));

fprintf('[SPLIT] train(all loads) base samples = %d\n', nnz(idx_train_base));
fprintf('[SPLIT] eval(%s only) base samples  = %d\n', strjoin(eval_load, ','), nnz(idx_eval_base));

%% ── 타겟별: (CV 없음) 전체학습 + US06 슬라이스 평가 ----------------------
results = struct();

for tt = 1:numel(target_active)
    tname = target_active{tt};
    y_all = target_values(tname);

    % 타겟별 유효 인덱스 (X,y finite + mask)
    idx_train = idx_train_base & isfinite(y_all);
    idx_eval  = idx_eval_base  & isfinite(y_all);

    X_tr = X(idx_train, :);
    y_tr = y_all(idx_train);
    l_tr = load_id(idx_train);

    X_ev = X(idx_eval, :);
    y_ev = y_all(idx_eval);

    n_tr = size(X_tr,1);
    n_ev = size(X_ev,1);

    fprintf('\n[TARGET %s] train=%d (ALL loads), eval(%s)=%d\n', ...
        tname, n_tr, strjoin(eval_load, ','), n_ev);

    if n_tr < nFeat + 1
        warning('  -> train 샘플 부족: %d < %d (nFeat+1). 스킵', n_tr, nFeat+1);
        continue;
    end

    %% ---- Final model: fit on ALL training data ------------------------
    mdl = fitlm(X_tr, y_tr);

    % train fit 성능(참고용)
    yhat_tr = predict(mdl, X_tr);
    res_tr  = y_tr - yhat_tr;
    tr_RMSE = sqrt(mean(res_tr.^2));
    tr_MAE  = mean(abs(res_tr));
    tr_R2   = mdl.Rsquared.Ordinary;

    %% ---- Eval on US06 slice (in-sample) -------------------------------
    eval_metrics = struct('R2',nan,'RMSE',nan,'MAE',nan);
    yhat_ev = [];
    res_ev  = [];
    if n_ev >= 2
        yhat_ev = predict(mdl, X_ev);
        res_ev  = y_ev - yhat_ev;

        eval_metrics.RMSE = sqrt(mean(res_ev.^2));
        eval_metrics.MAE  = mean(abs(res_ev));
        eval_metrics.R2   = 1 - sum(res_ev.^2) / sum( (y_ev-mean(y_ev)).^2 );

        fprintf('  [EVAL %s (in-sample slice)] R2=%.4f, RMSE=%.4f, MAE=%.4f\n', ...
            strjoin(eval_load, ','), eval_metrics.R2, eval_metrics.RMSE, eval_metrics.MAE);
    else
        fprintf('  [EVAL] eval 샘플 부족/없음 → 평가 스킵\n');
    end

    %% ---- Save results struct -----------------------------------------
    results.(tname).mdl = mdl;

    results.(tname).train.idx_global = idx_train;
    results.(tname).train.X = X_tr;
    results.(tname).train.y_true = y_tr;
    results.(tname).train.y_pred = yhat_tr;
    results.(tname).train.residuals = res_tr;
    results.(tname).train.R2 = tr_R2;
    results.(tname).train.RMSE = tr_RMSE;
    results.(tname).train.MAE = tr_MAE;

    % CV는 OFF → 빈 값으로 저장 (구조 유지)
    results.(tname).cv.K = NaN;
    results.(tname).cv.loads = unique(l_tr,'stable');
    results.(tname).cv.fold_id_of_load = [];
    results.(tname).cv.y_pred = nan(n_tr,1);
    results.(tname).cv.fold_R2 = [];
    results.(tname).cv.fold_RMSE = [];
    results.(tname).cv.fold_MAE = [];
    results.(tname).cv.R2_mean = NaN;
    results.(tname).cv.RMSE_mean = NaN;
    results.(tname).cv.MAE_mean = NaN;

    results.(tname).eval.idx_global = idx_eval;
    results.(tname).eval.X = X_ev;
    results.(tname).eval.y_true = y_ev;
    results.(tname).eval.y_pred = yhat_ev;
    results.(tname).eval.residuals = res_ev;
    results.(tname).eval.metrics = eval_metrics;

    results.(tname).feature_names = feat2RC_names;
    results.(tname).SOC_use = SOC_use;
    results.(tname).LOAD_USE_STR = LOAD_USE_STR;
    results.(tname).EVAL_LOAD_STR = EVAL_LOAD_STR;

    % 메타(유효 샘플에 대해서만)
    results.(tname).train.load_id = l_tr;
    results.(tname).train.cell_id = cell_id(idx_train);
    results.(tname).eval.load_id  = load_id(idx_eval);
    results.(tname).eval.cell_id  = cell_id(idx_eval);

    %% ---- plot: Train + Eval(US06 slice) in one figure -----------------
    fig_all = figure('Color','w','Name',sprintf('ALL_%s_true_vs_pred',tname));
    hold on; grid on;

    % train (final model prediction on train)
    h_tr = scatter(y_tr, yhat_tr, 35, 'filled', ...
        'Marker','o', ...
        'MarkerFaceColor',[0 0 0], ...
        'MarkerEdgeColor',[0 0 0]);

    % eval (US06 slice)
    h_ev = [];
    if n_ev >= 2
        h_ev = scatter(y_ev, yhat_ev, 35, 'filled', ...
            'Marker','o', ...
            'MarkerFaceColor',[0.85 0.1 0.1], ...
            'MarkerEdgeColor',[0.85 0.1 0.1]);
    end

    % 1:1 line (축 범위는 tr+ev 합쳐서)
    all_true = y_tr;  all_pred = yhat_tr;
    if n_ev >= 2
        all_true = [all_true; y_ev];
        all_pred = [all_pred; yhat_ev];
    end
    minv = min([all_true; all_pred]);
    maxv = max([all_true; all_pred]);
    plot([minv maxv],[minv maxv],'k--','LineWidth',1.2);

    tname_tex = strrep(tname, '_', '\_');
    xlabel(sprintf('True %s', tname_tex), 'Interpreter','tex','FontWeight','bold');
    ylabel(sprintf('Pred %s', tname_tex), 'Interpreter','tex','FontWeight','bold');

    % 제목에 train/eval 성능 같이 표기
    if n_ev >= 2
        title(sprintf('%s | Train(all loads): R2=%.3f RMSE=%.3f | Eval(%s in-sample): R2=%.3f RMSE=%.3f', ...
            tname, tr_R2, tr_RMSE, strjoin(eval_load, ','), eval_metrics.R2, eval_metrics.RMSE), 'Interpreter','none');
    else
        title(sprintf('%s | Train(all loads): R2=%.3f RMSE=%.3f | Eval(%s): (none)', ...
            tname, tr_R2, tr_RMSE, strjoin(eval_load, ',')), 'Interpreter','none');
    end

    axis equal; axis tight;

    if isempty(h_ev)
        legend(h_tr, {'Train(all loads)'}, 'Location','best');
    else
        legend([h_tr h_ev], {'Train(all loads)', sprintf('Eval(%s in-sample)', strjoin(eval_load, ','))}, 'Location','best');
    end

    exportgraphics(fig_all, fullfile(save_path, sprintf('ALL_%s_true_vs_pred.png', tname)), 'Resolution', 220);
    savefig(fig_all, fullfile(save_path, sprintf('ALL_%s_true_vs_pred.fig', tname)));

    % 계수 저장
    coef_tbl = mdl.Coefficients;
    coef_tbl.Properties.Description = sprintf('Final MLR coeffs for %s | TrainLoads=%s | EvalLoads=%s (in-sample slice)', ...
        tname, strjoin(load_use, ', '), strjoin(eval_load, ', '));
    writetable(coef_tbl, fullfile(save_path, sprintf('MLR_coeffs_FINAL_%s.csv', tname)), 'WriteRowNames', true);
end

%% ── 저장 ---------------------------------------------------------------
save(fullfile(save_path, 'MLR_results_TRAINALL_EVALLOAD_NOCV.mat'), ...
    'results', 'X', 'feat2RC_names', 'cell_names', 'SOC_use', 'pNames_2RC', ...
    'LOAD_USE_STR', 'EVAL_LOAD_STR', 'K_FOLD', 'TEMP_list', ...
    'load_id', 'cell_id', 'load_use', 'eval_load');

disp('완료: Train=ALL loads(US06 포함) + CV OFF + Eval=US06(in-sample slice) 결과 저장.');

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