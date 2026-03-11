%% ======================================================================
%  SVR: 2RC(부하별 Tbl_<LOAD>_ECM) + labels
%
%  [FEATURE STRUCTURE]
%   - sample = (cell, load)
%   - X = 해당 load의 2RC 파라미터(5×SOC_use)
%   - y = 셀 라벨을 load 개수만큼 복제
%
%  [MODE: TRAIN ALL + EVAL US06 SLICE | FIXED HPARAM from US06-only CELL-CV]
%   - Train: LOAD_USE 전체(US06 포함 8종) 모든 유효 샘플로 1회 학습
%   - CV: OFF
%   - Eval: US06 load 샘플만 슬라이스하여 성능 출력
%     (주의: US06도 Train에 포함되므로 hold-out test가 아님 / in-sample slice 평가)
%
%  [FIXED SVR HPARAMS]
%   - hyperparams are fixed per target, taken from US06-only CELL-CV best.
% ======================================================================
clear; clc; close all;

% (PATCH) Text Interpreter 에러 방지
set(groot,'defaultTextInterpreter','none');
set(groot,'defaultAxesTickLabelInterpreter','none');
set(groot,'defaultLegendInterpreter','none');

%% ── 설정 --------------------------------------------------------------
SOC_use = [70];

% 전체 load 풀 (학습에 전부 포함)
LOAD_USE_STR = "US06 UDDS HWFET WLTP CITY1 CITY2 HW1 HW2";

% 평가할 부하(슬라이스 평가)
EVAL_LOAD_STR = "US06";

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

% ---- SVR 공통 옵션 ----
SVR_STANDARDIZE = true;   % SVR은 거의 항상 true 추천
SVR_KERNEL      = 'gaussian';

%% ── 경로 --------------------------------------------------------------
matPath   = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\20degC\2RC_fitting_600s\2RC_results_600s.mat";
save_path = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\ML_SVR";
if ~exist(save_path, 'dir'), mkdir(save_path); end

%% ── load 파싱/검증 -----------------------------------------------------
loadNames_all = {'US06','UDDS','HWFET','WLTP','CITY1','CITY2','HW1','HW2'};
all_upper = upper(string(loadNames_all));

load_use = upper(parseLoadList(LOAD_USE_STR));
load_use = load_use(ismember(string(load_use), all_upper));
load_use = toStdCase(load_use, loadNames_all);
if isempty(load_use), error('LOAD_USE_STR 유효 load 없음'); end
fprintf('>> LOAD_USE: %s\n', strjoin(load_use, ', '));

eval_load = upper(parseLoadList(EVAL_LOAD_STR));
eval_load = eval_load(ismember(string(eval_load), all_upper));
eval_load = toStdCase(eval_load, loadNames_all);
if isempty(eval_load), error('EVAL_LOAD_STR 유효 load 없음'); end
fprintf('>> EVAL_LOAD: %s\n', strjoin(eval_load, ', '));

%% ── 2RC 테이블 로드 ----------------------------------------------------
S = load(matPath);
getTblECM = @(loadName) localGetTblECM(S, loadName);

% 실제 mat에 존재하는 load만 남기기
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
eval_load = intersect(eval_load, load_use, 'stable');

if isempty(load_use),  error('LOAD_USE 중 mat에 존재하는 테이블이 없음'); end
if isempty(eval_load), error('EVAL_LOAD가 mat에 존재하지 않음'); end

fprintf('>> (exists) LOAD_USE : %s\n', strjoin(load_use, ', '));
fprintf('>> (exists) EVAL_LOAD: %s\n', strjoin(eval_load, ', '));

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
pNames_2RC = {'R0','R1','R2','tau1','tau2'};

feat_names = {};
for s = SOC_use(:).'
    for pi = 1:numel(pNames_2RC)
        feat_names{end+1} = sprintf('%s_%d', pNames_2RC{pi}, s); %#ok<AGROW>
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
fprintf('[DATA] samples=%d (cells %d × loads %d), features=%d, validX=%d\n', ...
    nS, nC, nL_use, nFeat, nnz(base_valid_X));

%% ── 라벨 입력 (nC 기준) ------------------------------------------------
% ====== 네 값 그대로 ======
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
R1s_user = nan(nC,1);

DCIR1s_T20_user = nan(nC,1);
DCIR10s_T20_user = [1.48;1.34;1.97;1.91;1.64;1.76;1.35;1.46;3.44;1.45;1.45;1.41];
DCIRdelta_T20_user = nan(nC,1);

Power_T20_user = [2089.79;2372.03;1427.37;1735.16;1603.14;1677.27;2476.97;2191.48;914.67;4067.20;2196.09;2278.82]/1000;
% ===============================

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

% 셀 라벨을 (cell×load)로 확장 (X row order 정합)
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

% active 타겟 리스트
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

%% ── 학습/평가 마스크 (CV OFF) ------------------------------------------
idx_train_base = base_valid_X; % train = 전체 유효 샘플
idx_eval_base  = base_valid_X & ismember(load_id, string(eval_load)); % eval = 특정 load 슬라이스

fprintf('[SPLIT] train(all loads) base samples=%d\n', nnz(idx_train_base));
fprintf('[SPLIT] eval(%s only) base samples=%d\n', strjoin(eval_load, ','), nnz(idx_eval_base));

%% ── 타겟별: 전체학습(1회) + eval load 슬라이스 평가 ----------------------
results = struct();

for tt = 1:numel(target_active)
    tname = target_active{tt};
    y_all = target_values(tname);

    idx_train = idx_train_base & isfinite(y_all);
    idx_eval  = idx_eval_base  & isfinite(y_all);

    X_tr = X(idx_train,:);
    y_tr = y_all(idx_train);
    l_tr = load_id(idx_train);

    X_ev = X(idx_eval,:);
    y_ev = y_all(idx_eval);

    fprintf('\n[TARGET %s] train=%d (ALL loads), eval(%s)=%d\n', ...
        tname, size(X_tr,1), strjoin(eval_load, ','), size(X_ev,1));

    if size(X_tr,1) < nFeat + 1
        warning('  train 샘플 부족 → skip');
        continue;
    end

    %% ---- (FIXED) US06-only CELL-CV best hyperparams per target --------
    % QC2 & QC40 share same best
    switch string(tname)
        case {"QC2","QC40"}
            C0   = 6.356;
            eps0 = 0.009779;
            ks0  = 5.153;
        case "DCIR_10s_T20"
            C0   = 3.817;
            eps0 = 0.02138;
            ks0  = 18.82;
        case "Power_T20"
            C0   = 5.857;
            eps0 = 0.04204;
            ks0  = 15.79;
        otherwise
            % 만약 다른 타겟을 켰다면(예: Rcharg 등), 기본값을 하나 지정
            C0   = 6.356;
            eps0 = 0.009779;
            ks0  = 'auto';
    end

    %% ---- SVR fit (fixed 1회) -----------------------------------------
    mdl = fitrsvm(X_tr, y_tr, ...
        'KernelFunction', SVR_KERNEL, ...
        'Standardize', SVR_STANDARDIZE, ...
        'BoxConstraint', C0, ...
        'Epsilon', eps0, ...
        'KernelScale', ks0);

    % train fit 성능(참고용)
    yhat_tr = predict(mdl, X_tr);
    tr.RMSE = sqrt(mean((y_tr - yhat_tr).^2));
    tr.MAE  = mean(abs(y_tr - yhat_tr));
    tr.R2   = calcR2(y_tr, yhat_tr);

    % eval slice 성능
    ev = struct('RMSE',nan,'MAE',nan,'R2',nan);
    yhat_ev = [];
    if size(X_ev,1) >= 2
        yhat_ev = predict(mdl, X_ev);
        ev.RMSE = sqrt(mean((y_ev - yhat_ev).^2));
        ev.MAE  = mean(abs(y_ev - yhat_ev));
        ev.R2   = calcR2(y_ev, yhat_ev);

        fprintf('  [EVAL %s (in-sample slice)] R2=%.4f RMSE=%.4f MAE=%.4f\n', ...
            strjoin(eval_load, ','), ev.R2, ev.RMSE, ev.MAE);
    else
        fprintf('  [EVAL] eval 샘플 부족/없음 → 스킵\n');
    end

    %% ---- 저장 ---------------------------------------------------------
    results.(tname).mdl = mdl;
    results.(tname).fixed.Kernel      = SVR_KERNEL;
    results.(tname).fixed.Standardize = SVR_STANDARDIZE;
    results.(tname).fixed.C           = C0;
    results.(tname).fixed.Epsilon     = eps0;
    results.(tname).fixed.KernelScale = ks0;

    results.(tname).train.metrics = tr;
    results.(tname).eval.metrics  = ev;

    results.(tname).train.y_true  = y_tr;
    results.(tname).train.y_pred  = yhat_tr;
    results.(tname).train.load_id = l_tr;
    results.(tname).train.cell_id = cell_id(idx_train);

    results.(tname).eval.y_true   = y_ev;
    results.(tname).eval.y_pred   = yhat_ev;
    results.(tname).eval.load_id  = load_id(idx_eval);
    results.(tname).eval.cell_id  = cell_id(idx_eval);

    results.(tname).feature_names = feat_names;
    results.(tname).SOC_use       = SOC_use;
    results.(tname).LOAD_USE_STR  = LOAD_USE_STR;
    results.(tname).EVAL_LOAD_STR = EVAL_LOAD_STR;

    %% ---- plot: Train + Eval in one figure ----------------------------
    fig_all = figure('Color','w','Name',sprintf('SVR_TRAINALL_EVAL_%s',tname));
    hold on; grid on;

    h_tr = scatter(y_tr, yhat_tr, 30, 'filled', ...
        'Marker','o', 'MarkerFaceColor',[0 0 0], 'MarkerEdgeColor',[0 0 0]);

    h_ev = [];
    if numel(yhat_ev) >= 2
        h_ev = scatter(y_ev, yhat_ev, 30, 'filled', ...
            'Marker','o', 'MarkerFaceColor',[0.85 0.1 0.1], 'MarkerEdgeColor',[0.85 0.1 0.1]);
    end

    all_true = y_tr; all_pred = yhat_tr;
    if numel(yhat_ev) >= 2
        all_true = [all_true; y_ev];
        all_pred = [all_pred; yhat_ev];
    end
    minv = min([all_true; all_pred]);
    maxv = max([all_true; all_pred]);
    plot([minv maxv],[minv maxv],'k--','LineWidth',1.2);

    xlabel(sprintf('True %s', tname));
    ylabel(sprintf('Pred %s', tname));

    if numel(yhat_ev) >= 2
        title(sprintf('%s | Train(all loads): R2=%.3f RMSE=%.3f | Eval(%s in-sample): R2=%.3f RMSE=%.3f', ...
            tname, tr.R2, tr.RMSE, strjoin(eval_load, ','), ev.R2, ev.RMSE));
    else
        title(sprintf('%s | Train(all loads): R2=%.3f RMSE=%.3f | Eval(%s): (none)', ...
            tname, tr.R2, tr.RMSE, strjoin(eval_load, ',')));
    end

    axis equal; axis tight;

    if isempty(h_ev)
        legend(h_tr, {'Train(all loads)'}, 'Location','best');
    else
        legend([h_tr h_ev], {'Train(all loads)', sprintf('Eval(%s in-sample)', strjoin(eval_load, ','))}, 'Location','best');
    end

    exportgraphics(fig_all, fullfile(save_path, sprintf('SVR_TRAINALL_EVAL_%s.png', tname)), 'Resolution', 220);
    savefig(fig_all, fullfile(save_path, sprintf('SVR_TRAINALL_EVAL_%s.fig', tname)));
end

%% ── 저장 --------------------------------------------------------------
save(fullfile(save_path, 'SVR_results_TRAINALL_EVALLOAD_NOCV_FIXED_US06ONLY.mat'), ...
    'results', 'X', 'feat_names', 'cell_names', 'SOC_use', 'pNames_2RC', ...
    'LOAD_USE_STR', 'EVAL_LOAD_STR', 'TEMP_list', ...
    'load_id', 'cell_id', 'load_use', 'eval_load', ...
    'SVR_STANDARDIZE','SVR_KERNEL');

disp('완료: SVR Train=ALL loads(US06 포함) + CV OFF + Eval=US06(in-sample slice) + FIXED(US06-only best) 저장');

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