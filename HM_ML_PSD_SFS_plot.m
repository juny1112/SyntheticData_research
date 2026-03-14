%% ======================================================================
%  plot_PSD_subset_best_by_size_MLR_SVR.m
%
%  목적:
%   - MLR(Ridge exhaustive) / SVR(exhaustive) 결과 .mat에서
%     target(라벨)별로 "PSD subset size(0~5)에서 CV-best 조합" 성능을 plot
%
%  입력(필수):
%   - MLR_RIDGE_results_split_LOADCV_withPSD_EXHAUSTIVE.mat
%   - SVR_results_split_LOADCV_withPSD_EXHAUSTIVE.mat
%
%  기대 구조:
%   - results.(target).subset_search.best_by_size_table (table)
%     columns: NumPSD, CV_RMSE, CV_MAE, CV_R2, Test_RMSE, Test_MAE, Test_R2, PSDSubset ...
%
%  출력:
%   - (A) target별 figure: MLR vs SVR overlay (CV + optional Test)
%   - (B) 전체 target 4개를 한 figure(2x2)로도 가능 (옵션)
%% ======================================================================
clear; clc; close all;

%% ====== USER CONFIG ====================================================
% 결과 MAT 경로 (본인 PC 경로로 수정)
MLR_MAT = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\ML_MLR\MLR_RIDGE_results_split_LOADCV_withPSD_EXHAUSTIVE.mat";
SVR_MAT = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\ML_SVR\SVR_results_split_LOADCV_withPSD_EXHAUSTIVE.mat";

% 라벨(타겟) 4개
TARGETS = { "QC2", "QC40", "DCIR_10s_T20", "Power_T20" };

% plot metric
METRIC = "RMSE";   % "RMSE" | "MAE" | "R2"

% test curve 표시 여부
SHOW_TEST = true;

% subset size 표시 최대치 (너는 0~5라 했으니 5)
MAX_N_PSD = 5;

% 파일 저장
SAVE_FIG = true;
SAVE_DIR = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\ML_PLOTS\PSD_subset_best_by_size";
if SAVE_FIG && ~exist(SAVE_DIR,'dir'), mkdir(SAVE_DIR); end

% (옵션) 4개 타겟을 한 figure에 2x2 subplot으로도 만들지
MAKE_GRID_FIG = true;

% (옵션) 점 옆에 PSDSubset 문자열 표시(길면 지저분해질 수 있음)
ANNOTATE_SUBSET = false;  % true면 점 옆에 subset명 붙임

%% ====== LOAD RESULTS ===================================================
S1 = load(MLR_MAT, "results");
S2 = load(SVR_MAT, "results");
resMLR = S1.results;
resSVR = S2.results;

%% ====== PER-TARGET FIGURES ============================================
for ti = 1:numel(TARGETS)
    tname = char(TARGETS{ti});

    tbl_mlr = localGetBestBySize(resMLR, tname);
    tbl_svr = localGetBestBySize(resSVR, tname);

    if isempty(tbl_mlr) && isempty(tbl_svr)
        warning("[SKIP] target=%s : best_by_size_table not found in both MLR/SVR.", tname);
        continue;
    end

    [x_mlr, ycv_mlr, yte_mlr, s_mlr] = localCurveFromTbl(tbl_mlr, METRIC, MAX_N_PSD);
    [x_svr, ycv_svr, yte_svr, s_svr] = localCurveFromTbl(tbl_svr, METRIC, MAX_N_PSD);

    fig = figure('Color','w', 'Name', sprintf('PSD subset size vs %s | %s', METRIC, tname));
    hold on; grid on;

    xlabel('NumPSD (subset size)');
    ylabel(sprintf('CV_%s', METRIC));
    title(sprintf('CV-best PSD subset by size | target=%s | metric=%s', tname, METRIC), 'Interpreter','none');

    legH = gobjects(0); legL = strings(0);

    % ----- MLR -----
    if ~isempty(x_mlr)
        h = plot(x_mlr, ycv_mlr, '-o', 'LineWidth', 1.8, 'MarkerSize', 6);
        legH(end+1) = h; legL(end+1) = "MLR (CV-best)";

        if SHOW_TEST && ~isempty(yte_mlr)
            h2 = plot(x_mlr, yte_mlr, '--o', 'LineWidth', 1.2, 'MarkerSize', 5);
            legH(end+1) = h2; legL(end+1) = "MLR (Test)";
        end

        if ANNOTATE_SUBSET
            localAnnotate(x_mlr, ycv_mlr, s_mlr, "MLR");
        end
    else
        warning("[INFO] target=%s : MLR best_by_size_table missing.", tname);
    end

    % ----- SVR -----
    if ~isempty(x_svr)
        h = plot(x_svr, ycv_svr, '-s', 'LineWidth', 1.8, 'MarkerSize', 6);
        legH(end+1) = h; legL(end+1) = "SVR (CV-best)";

        if SHOW_TEST && ~isempty(yte_svr)
            h2 = plot(x_svr, yte_svr, '--s', 'LineWidth', 1.2, 'MarkerSize', 5);
            legH(end+1) = h2; legL(end+1) = "SVR (Test)";
        end

        if ANNOTATE_SUBSET
            localAnnotate(x_svr, ycv_svr, s_svr, "SVR");
        end
    else
        warning("[INFO] target=%s : SVR best_by_size_table missing.", tname);
    end

    legend(legH, legL, 'Location','best');
    xlim([0 MAX_N_PSD]);

    if SAVE_FIG
        outP = fullfile(SAVE_DIR, sprintf('%s_%s_PSDsubset_by_size.png', tname, METRIC));
        exportgraphics(fig, outP, 'Resolution', 220);
        savefig(fig, fullfile(SAVE_DIR, sprintf('%s_%s_PSDsubset_by_size.fig', tname, METRIC)));
    end
end

%% ====== GRID FIGURE (2x2) =============================================
if MAKE_GRID_FIG
    figG = figure('Color','w', 'Name', sprintf('GRID PSD subset size vs %s', METRIC));
    tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

    for ti = 1:numel(TARGETS)
        tname = char(TARGETS{ti});
        nexttile; hold on; grid on;

        tbl_mlr = localGetBestBySize(resMLR, tname);
        tbl_svr = localGetBestBySize(resSVR, tname);

        [x_mlr, ycv_mlr, yte_mlr] = localCurveFromTbl(tbl_mlr, METRIC, MAX_N_PSD);
        [x_svr, ycv_svr, yte_svr] = localCurveFromTbl(tbl_svr, METRIC, MAX_N_PSD);

        title(tname, 'Interpreter','none');
        xlabel('NumPSD'); ylabel(sprintf('CV_%s', METRIC));
        xlim([0 MAX_N_PSD]);

        if ~isempty(x_mlr)
            plot(x_mlr, ycv_mlr, '-o', 'LineWidth', 1.5, 'MarkerSize', 5);
            if SHOW_TEST && ~isempty(yte_mlr), plot(x_mlr, yte_mlr, '--o', 'LineWidth', 1.0, 'MarkerSize', 4); end
        end
        if ~isempty(x_svr)
            plot(x_svr, ycv_svr, '-s', 'LineWidth', 1.5, 'MarkerSize', 5);
            if SHOW_TEST && ~isempty(yte_svr), plot(x_svr, yte_svr, '--s', 'LineWidth', 1.0, 'MarkerSize', 4); end
        end
    end

    % 공통 legend는 마지막 tile에만 간단히 표시(겹치면 안 예뻐서)
    lg = legend( ...
        { "MLR (CV-best)", "MLR (Test)", "SVR (CV-best)", "SVR (Test)" }, ...
        'Location','southoutside', 'Orientation','horizontal');
    lg.Layout.Tile = 'south';

    if SAVE_FIG
        outP = fullfile(SAVE_DIR, sprintf('GRID_%s_PSDsubset_by_size.png', METRIC));
        exportgraphics(figG, outP, 'Resolution', 220);
        savefig(figG, fullfile(SAVE_DIR, sprintf('GRID_%s_PSDsubset_by_size.fig', METRIC)));
    end
end

disp('[done] plotting complete.');

%% ======================================================================
%  LOCAL FUNCTIONS
%% ======================================================================

function tbl = localGetBestBySize(resultsStruct, tname)
    tbl = [];
    if isempty(resultsStruct) || ~isstruct(resultsStruct), return; end
    if ~isfield(resultsStruct, tname), return; end

    R = resultsStruct.(tname);

    if isfield(R,'subset_search') && isfield(R.subset_search,'best_by_size_table') ...
            && istable(R.subset_search.best_by_size_table) ...
            && height(R.subset_search.best_by_size_table) > 0
        tbl = R.subset_search.best_by_size_table;
    end
end

function [x, ycv, yte, subsetStr] = localCurveFromTbl(tbl, metric, maxN)
    x = []; ycv = []; yte = []; subsetStr = strings(0,1);
    if isempty(tbl) || ~istable(tbl) || height(tbl)==0, return; end
    if ~ismember('NumPSD', tbl.Properties.VariableNames), return; end

    tbl = sortrows(tbl, 'NumPSD','ascend');
    m = tbl.NumPSD <= maxN;
    tbl = tbl(m,:);

    x = tbl.NumPSD;

    switch upper(char(metric))
        case 'RMSE'
            cvVar = 'CV_RMSE'; teVar = 'Test_RMSE';
        case 'MAE'
            cvVar = 'CV_MAE';  teVar = 'Test_MAE';
        case 'R2'
            cvVar = 'CV_R2';   teVar = 'Test_R2';
        otherwise
            error("Unknown METRIC=%s (use RMSE/MAE/R2)", string(metric));
    end

    if ismember(cvVar, tbl.Properties.VariableNames)
        ycv = tbl.(cvVar);
    else
        ycv = nan(size(x));
    end

    if ismember(teVar, tbl.Properties.VariableNames)
        yte = tbl.(teVar);
    else
        yte = [];
    end

    if ismember('PSDSubset', tbl.Properties.VariableNames)
        subsetStr = string(tbl.PSDSubset);
    else
        subsetStr = strings(numel(x),1);
    end
end

function localAnnotate(x, y, subsetStr, tag)
    for i = 1:numel(x)
        s = subsetStr(i);
        if strlength(s) > 40
            s = extractBefore(s, 40) + "...";
        end
        text(x(i), y(i), sprintf('  %s:%s', tag, s), 'FontSize', 8, 'Interpreter','none');
    end
end