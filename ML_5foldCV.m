%% ======================================================================
%  2RC(Tbl_ECM_mean) → MLR (다중선형회귀, 5-fold Cross Validation)
%  - 입력:
%       2RC_results.mat (Tbl_ECM_mean 포함)
%          * Tbl_ECM_mean.RowNames : 셀 이름
%          * Tbl_ECM_mean.VarNames: SOC90_R0_mOhm, SOC90_R1_mOhm, ..., SOC30_tau2
%       사용자 스칼라(QC/2, QC/40, R1s, DCIR_1s, DCIR_10s, ΔDCIR, Rcharg)
%
%  - MLR 설정:
%       * 입력 X : 선택 SOC의 ECM 파라미터 (R0,R1,R2,tau1,tau2)
%       * 출력 y : 사용자 스칼라 (토글로 선택)
%       * 5-fold cross validation 사용 (cvpartition으로 수동 구현)
%       * 회귀계수는 전체 데이터로 학습한 full 모델 기준 (fitlm 기본값)
%
%  - 출력( save_path ):
%       X/Y 저장, 각 타겟별 회귀계수, 5-fold CV 예측결과, 성능지표, 그림 등
% ======================================================================
clear; clc; close all;

%% ── 설정(토글) ---------------------------------------------------------
% 입력 ECM에서 사용할 SOC
SOC_use = [70];   % 예: [90 70 50 30], [90 50], [70] 등

% 타겟(출력) 활성화 토글
PRED_QC2        = true;
PRED_QC40       = true;
PRED_R1S        = false;
PRED_DCIR1S     = false;
PRED_DCIR10S    = false;   % DCIR10s 데이터 없으면 false 유지
PRED_DELTA      = false;   % ΔDCIR
PRED_RCHARG     = false;

%% ── 경로/파일 ----------------------------------------------------------
% 2RC_results.mat (Tbl_ECM_mean 사용)
matPath   = 'G:\공유 드라이브\BSL_Data4\HNE_agedcell_8_processed\SIM_parsed\2RC_fitting\2RC_results.mat';

% 결과 저장 경로 (MLR)
save_path = 'C:\Users\junny\Downloads';
if ~exist(save_path, 'dir'), mkdir(save_path); end

%% ── Tbl_ECM_mean 로드 & 2RC 파라미터 구성 -----------------------------
S = load(matPath, 'Tbl_ECM_mean');
if ~isfield(S,'Tbl_ECM_mean')
    error('Tbl_ECM_mean 이 %s 에서 발견되지 않았습니다.', matPath);
end
Tbl_ECM_mean = S.Tbl_ECM_mean;

% 셀 이름
cell_names = Tbl_ECM_mean.Properties.RowNames;
if isstring(cell_names)
    cell_names = cellstr(cell_names);
elseif ischar(cell_names)
    cell_names = cellstr(cell_names);
end
nC = numel(cell_names);

% SOC 리스트를 VarNames 에서 자동 추출 (SOC90_..., SOC50_... 이런 형태)
vnames = Tbl_ECM_mean.Properties.VariableNames;
soc_list = [];
for i = 1:numel(vnames)
    tok = regexp(vnames{i}, '^SOC(\d+)_', 'tokens', 'once');
    if ~isempty(tok)
        soc_list(end+1) = str2double(tok{1}); %#ok<AGROW>
    end
end
SOC_2RC = unique(soc_list, 'stable');  % 예: [90 70 50 30]
if isempty(SOC_2RC)
    error('Tbl_ECM_mean VarNames 에 "SOCxx_..." 형식의 변수를 찾지 못했습니다.');
end

% 표준 파라미터 이름
pNames_2RC = {'R0','R1','R2','tau1','tau2'};
nSOC_all   = numel(SOC_2RC);

% P2RC 초기화: 각 필드 [nC × nSOC_all]
P2RC = struct();
for k = 1:numel(pNames_2RC)
    P2RC.(pNames_2RC{k}) = nan(nC, nSOC_all);
end

% Tbl_ECM_mean → P2RC 채우기
for sIdx = 1:nSOC_all
    soc = SOC_2RC(sIdx);    % 예: 90, 70, 50, 30
    for pi = 1:numel(pNames_2RC)
        pname = pNames_2RC{pi};
        if pi <= 3
            % R0,R1,R2 : mΩ
            varName = sprintf('SOC%d_%s_mOhm', soc, pname);
        else
            % tau1,tau2 : s
            varName = sprintf('SOC%d_%s', soc, pname);
        end

        if ~ismember(varName, vnames)
            warning('Tbl_ECM_mean 에 변수 %s 가 없습니다. (SOC=%d, param=%s)', varName, soc, pname);
        else
            P2RC.(pname)(:, sIdx) = Tbl_ECM_mean{:, varName};
        end
    end
end

%% ── SOC 선택/정합 ------------------------------------------------------
SOC_use = SOC_use(:).';
SOC_use = SOC_use(ismember(SOC_use, SOC_2RC));     % Tbl_ECM_mean에 존재하는 SOC만
if isempty(SOC_use)
    error('선택한 SOC(%s)가 Tbl_ECM_mean 데이터에 존재하지 않습니다. (SOC_2RC=%s)', ...
        mat2str(SOC_use), mat2str(SOC_2RC));
end
idxSOC_2RC = arrayfun(@(s)find(SOC_2RC==s,1), SOC_use);
nSOC = numel(SOC_use);

% SOC별 파라미터명 구성 (예: R0_90, R1_90, ..., tau2_50 ...)
soc_param_names_2RC = {};
for s = SOC_use
    soc_param_names_2RC = [soc_param_names_2RC, strcat(pNames_2RC, ['_' num2str(s)])]; %#ok<AGROW>
end
nFeat = numel(soc_param_names_2RC);

%% ── 입력 X (ECM 파라미터) 행렬 생성 -----------------------------------
X = nan(nC, nFeat);
for i = 1:nC
    c = 1;
    for jj = 1:nSOC
        sIdx = idxSOC_2RC(jj);
        for k = 1:numel(pNames_2RC)
            X(i,c) = P2RC.(pNames_2RC{k})(i, sIdx);
            c = c + 1;
        end
    end
end
base_valid_X = all(isfinite(X), 2);   % X에서 NaN 없는 행

%% ── 타겟 스칼라 입력(셀 순서=Tbl_ECM_mean Row 순서) -------------------
fprintf('셀 순서(%d개):\n', nC);
disp(strjoin(cell_names, ', '));

% ⚠ 여기 스칼라 값들은 cell_names 순서와 맞아야 함
QC2_user      = [56.14
55.93
52.55
50.52
52.13
48.53
56.34
55.15
39.89
55.63
55.57
30.16
56.86];

QC40_user     = [57.49
57.57
54
52.22
53.45
51.28
57.91
56.51
42.14
57.27
57.18
43.92
58.4];

DCIR1s_user   = [0.99
0.95
1.37
1.31
1.47
1.29
1.04
0.98
2.36
1.02
0.98
2.53
1.02];

% DCIR10s 가 없으면 NaN으로 두고, ΔDCIR에서만 사용 가능하게 처리
DCIR10s_user  = nan(nC,1);

Rcharg_user   = [2.72
2.43
3.51
3.21
3.35
3.50
2.47
2.51
5.10
2.63
2.61
3.43
2.63];

R1s_user      = [0.98
0.94
1.33
1.30
1.45
1.28
1.02
0.96
2.36
0.97
0.96
2.53
1.00];

% ΔDCIR (= DCIR_10s - DCIR_1s) 직접 입력
DCIRdelta_user = [0.28
0.26
0.40
0.37
0.33
0.37
0.27
0.26
0.81
0.23
0.26
0.79
0.26];

% 길이 보정
QC2_user       = ensureLength(QC2_user,       nC);
QC40_user      = ensureLength(QC40_user,      nC);
DCIR1s_user    = ensureLength(DCIR1s_user,    nC);
DCIR10s_user   = ensureLength(DCIR10s_user,   nC);
Rcharg_user    = ensureLength(Rcharg_user,    nC);
R1s_user       = ensureLength(R1s_user,       nC);
DCIRdelta_user = ensureLength(DCIRdelta_user, nC);

% ΔDCIR 최종 벡터 (입력 우선, 없으면 10s-1s)
dDCIR_user = nan(nC,1);
for i = 1:nC
    if isfinite(DCIRdelta_user(i))
        dDCIR_user(i) = DCIRdelta_user(i);
    elseif isfinite(DCIR10s_user(i)) && isfinite(DCIR1s_user(i))
        dDCIR_user(i) = DCIR10s_user(i) - DCIR1s_user(i);
    else
        dDCIR_user(i) = NaN;
    end
end

%% ── 타겟 이름 / 벡터 매핑 ----------------------------------------------
target_pool_names = {'QC2','QC40','R1s','DCIR_1s','DCIR_10s','DCIR_delta_10s_1s','Rcharg'};
target_values = containers.Map( ...
    target_pool_names, ...
    {QC2_user(:), ...
     QC40_user(:), ...
     R1s_user(:), ...
     DCIR1s_user(:), ...
     DCIR10s_user(:), ...
     dDCIR_user(:), ...
     Rcharg_user(:)} );

% 토글 기반 active target list
target_active = {};
if PRED_QC2,     target_active{end+1} = 'QC2';               end
if PRED_QC40,    target_active{end+1} = 'QC40';              end
if PRED_R1S,     target_active{end+1} = 'R1s';               end
if PRED_DCIR1S,  target_active{end+1} = 'DCIR_1s';           end
if PRED_DCIR10S, target_active{end+1} = 'DCIR_10s';          end
if PRED_DELTA,   target_active{end+1} = 'DCIR_delta_10s_1s'; end
if PRED_RCHARG,  target_active{end+1} = 'Rcharg';            end

if isempty(target_active)
    error('활성화된 타겟이 없습니다. PRED_XXX 토글을 확인하세요.');
end

fprintf('\n[MLR] 사용할 입력 피처(%d개):\n', nFeat);
disp(strjoin(soc_param_names_2RC, ', '));

fprintf('[MLR] 활성 타겟:\n');
disp(strjoin(target_active, ', '));

%% ── 전체 X / 각 타겟에 대한 5-fold CV MLR -----------------------------
results = struct();
nFeatX  = size(X,2);

for t = 1:numel(target_active)
    tname = target_active{t};
    y     = target_values(tname);

    % 유효 행: X, y 둘 다 finite
    idx_valid = base_valid_X & isfinite(y);
    n_valid   = nnz(idx_valid);

    fprintf('\n[MLR] Target = %s, 유효 샘플 수 = %d / %d\n', tname, n_valid, nC);

    if n_valid < nFeatX + 1
        warning('타겟 %s: 유효 샘플이 너무 적어 (n_valid=%d < nFeat+1=%d) 회귀 불가. 건너뜁니다.', ...
            tname, n_valid, nFeatX+1);
        continue;
    end
    if n_valid < 5
        warning('타겟 %s: 5-fold CV 를 수행하기에 샘플 수가 부족합니다 (n_valid=%d < 5). 건너뜁니다.', ...
            tname, n_valid);
        continue;
    end

    Xv = X(idx_valid, :);
    yv = y(idx_valid);

    % ---- full 모델 (전체 데이터 학습, 계수/해석용) ----
    mdl_full = fitlm(Xv, yv);   % 기본: 절편 포함, 모든 피처 사용, OLS

    % ---- 5-fold cross validation (cvpartition으로 수동 구현) ----
    K = 5;
    cvp = cvpartition(n_valid, 'KFold', K);

    y_hat_cv  = nan(n_valid, 1);   % CV 예측값 (각 샘플에 하나씩)
    fold_RMSE = nan(K, 1);        % fold별 RMSE

    for k = 1:K
        idxTrain = training(cvp, k);  % 학습 인덱스 (logical)
        idxTest  = test(cvp, k);      % 검증 인덱스 (logical)

        % 각 fold마다 학습
        mdl_k = fitlm(Xv(idxTrain, :), yv(idxTrain));

        % 해당 fold의 검증 샘플에 대한 예측
        y_hat_cv(idxTest) = predict(mdl_k, Xv(idxTest, :));

        % fold별 RMSE
        res_k        = yv(idxTest) - y_hat_cv(idxTest);
        fold_RMSE(k) = sqrt(mean(res_k.^2));
    end

    % ---- CV 기반 성능 지표 ----
    res_cv = yv - y_hat_cv;

    RMSE_cv = sqrt(mean(res_cv.^2));
    MAE_cv  = mean(abs(res_cv));

    % R^2 (CV 기반): 1 - SSE/SST
    SSE = sum(res_cv.^2);
    SST = sum((yv - mean(yv)).^2);
    R2_cv = 1 - SSE / SST;

    fprintf('  [5-fold CV] R^2   = %.4f\n', R2_cv);
    fprintf('  [5-fold CV] RMSE  = %.4f\n', RMSE_cv);
    fprintf('  [5-fold CV] MAE   = %.4f\n', MAE_cv);

    % TeX-safe 타겟 이름 (언더스코어 escape)
    tname_tex = strrep(tname, '_', '\_');

    % 결과 구조체 저장
    results.(tname).mdl_full        = mdl_full;
    results.(tname).X               = Xv;
    results.(tname).y_true          = yv;
    results.(tname).y_pred_cv       = y_hat_cv;
    results.(tname).residuals_cv    = res_cv;
    results.(tname).R2_cv           = R2_cv;
    results.(tname).RMSE_cv         = RMSE_cv;
    results.(tname).MAE_cv          = MAE_cv;
    results.(tname).fold_RMSE       = fold_RMSE;
    results.(tname).idx_valid       = idx_valid;
    results.(tname).feature_names   = soc_param_names_2RC;

    %% ---- 계수 테이블 CSV 저장 (full 모델 기준) -------------------------
    coef_tbl = mdl_full.Coefficients;   % RowNames: (Intercept), x1,x2,... / Name: Estimate, SE, tStat, pValue
    coef_tbl.Properties.Description = sprintf('MLR coefficients (full model) for target %s', tname);
    coef_csv = fullfile(save_path, sprintf('MLR_coeffs_%s.csv', tname));
    writetable(coef_tbl, coef_csv, 'WriteRowNames', true);
    fprintf('  -> 계수 테이블 저장: %s\n', coef_csv);

    %% ---- 예측 vs 실제 scatter plot (5-fold CV 결과) --------------------
    fig1 = figure('Color','w','Name',sprintf('MLR_%s_true_vs_pred_CV',tname));
    scatter(yv, y_hat_cv, 40, 'k', 'filled'); hold on; grid on;
    minv = min([yv; y_hat_cv]); maxv = max([yv; y_hat_cv]);
    plot([minv maxv], [minv maxv], 'r--', 'LineWidth', 1.5);   % y = x line
    xlabel(sprintf('True %s', tname_tex), 'FontSize', 11, 'FontWeight','bold', 'Interpreter','tex');
    ylabel(sprintf('Predicted %s (5-fold CV)', tname_tex), 'FontSize', 11, 'FontWeight','bold', 'Interpreter','tex');
    title(sprintf('MLR (5-fold CV): %s (R^2=%.3f, RMSE=%.3f)', tname_tex, R2_cv, RMSE_cv), ...
        'FontSize', 12, 'FontWeight','bold', 'Interpreter','tex');
    axis equal; axis tight;

    outfig1 = fullfile(save_path, sprintf('MLR_%s_true_vs_pred_5foldCV.fig', tname));
    outpng1 = fullfile(save_path, sprintf('MLR_%s_true_vs_pred_5foldCV.png', tname));
    savefig(fig1, outfig1);
    exportgraphics(fig1, outpng1, 'Resolution', 220);

    %% ---- 잔차(residual) plot (5-fold CV 기준) --------------------------
    fig2 = figure('Color','w','Name',sprintf('MLR_%s_residuals_CV',tname));

    subplot(2,1,1);
    plot(res_cv, 'o-','LineWidth',1.2);
    grid on;
    xlabel('Sample index (valid rows)');
    ylabel('Residual (CV)');
    title(sprintf('Residuals (y - \\hat{y}_{CV}) for %s (5-fold CV)', tname_tex), ...
        'FontWeight','bold', 'Interpreter','tex');

    subplot(2,1,2);
    histogram(res_cv, 'NumBins', max(5, round(sqrt(numel(res_cv)))));
    grid on;
    xlabel('Residual (CV)');
    ylabel('Count');
    title('Residual histogram (5-fold CV)','FontWeight','bold');

    outfig2 = fullfile(save_path, sprintf('MLR_%s_residuals_5foldCV.fig', tname));
    outpng2 = fullfile(save_path, sprintf('MLR_%s_residuals_5foldCV.png', tname));
    savefig(fig2, outfig2);
    exportgraphics(fig2, outpng2, 'Resolution', 220);
end

%% ── 전체 결과 .mat 저장 ------------------------------------------------
save(fullfile(save_path, 'MLR_results_all_5foldCV.mat'), ...
    'results', 'X', 'soc_param_names_2RC', 'cell_names', ...
    'SOC_use', 'pNames_2RC');

disp('모든 MLR (5-fold CV) 결과 저장 완료.');

%% ========================= 보조 함수들 =================================
function v = ensureLength(v, n)
    v = v(:).';
    if numel(v) < n
        v = [v, nan(1, n - numel(v))];
    elseif numel(v) > n
        v = v(1:n);
    end
end
