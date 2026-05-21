clear; clc; close all;

%% ===== 설정 =====
SOC_use = 70;
TEMP_load = 20;     % ECM feature temperature
TEMP_label = 20;    % DCIR10s temperature
LOAD_use = 'US06';

baseDir_2RC = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed";
save_path = 'G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\Correlation_Anlaysis';
if ~exist(save_path, 'dir'), mkdir(save_path); end

%% ===== 2RC mat 로드 =====
matPath = fullfile(baseDir_2RC, sprintf('%ddegC', TEMP_load), '2RC_fitting_600s', '2RC_results_600s.mat');
if ~isfile(matPath)
    error('2RC mat 파일이 없습니다: %s', matPath);
end
S = load(matPath);

Tbl = localGetTblECM(S, LOAD_use);
raw_names = Tbl.Properties.RowNames;

ids_raw = extractCellIDs(raw_names);
[ids_unique, idx_keep] = uniqueCellIDsFirst(ids_raw);

Tbl = Tbl(idx_keep, :);
cell_ids = ids_unique(:);
nC = numel(cell_ids);

fprintf('>> 공통 셀 개수: %d\n', nC);
fprintf('>> 공통 셀 ID: ');
disp(cell_ids(:).');

%% ===== 사용자 label 데이터 입력 =====
QC2_user = [56.14;55.93;52.55;50.52;52.13;48.53;56.34;55.15;39.89;55.63;55.57;56.86];
QC40_user = [57.49;57.57;54;52.22;53.45;51.28;57.91;56.51;42.14;57.27;57.18;58.4];
Rcharg_user = [2.17;1.90;3.50;2.82;2.88;3.38;2.10;1.93;6.41;2.00;2.01;2.09];

DCIR10s_T20_user = [1.60;1.41;2.33;1.88;2.06;1.93;1.35;1.53;3.55;0.82;1.52;1.47];

QC2_user = ensureLength(QC2_user, nC);
QC40_user = ensureLength(QC40_user, nC);
Rcharg_user = ensureLength(Rcharg_user, nC);
DCIR10s_T20_user = ensureLength(DCIR10s_T20_user, nC);

%% ===== ECM 5개 파라미터 추출 =====
pNames = {'R0','R1','R2','tau1','tau2'};
X_ecm = nan(nC, numel(pNames));

for pi = 1:numel(pNames)
    pname = pNames{pi};

    if pi <= 3
        varName = sprintf('SOC%d_%s_mOhm', SOC_use, pname);
    else
        varName = sprintf('SOC%d_%s', SOC_use, pname);
    end

    if ~ismember(varName, Tbl.Properties.VariableNames)
        error('변수 %s 를 Tbl에서 찾지 못했습니다.', varName);
    end

    X_ecm(:, pi) = Tbl{:, varName};
end

%% ===== Label 행렬 구성 =====
Y_lab = [QC2_user, QC40_user, DCIR10s_T20_user, Rcharg_user];

ecm_display_names = {'R_0','R_1','R_2','\tau_1','\tau_2'};
label_display_names = {'Q_{C/2}','Q_{C/40}','DCIR_{10s}','R_{charge}'};

%% ===== Pearson correlation 계산 =====
R_custom = nan(size(Y_lab,2), size(X_ecm,2));
P_custom = nan(size(Y_lab,2), size(X_ecm,2));

for i = 1:size(Y_lab,2)
    for j = 1:size(X_ecm,2)
        x = X_ecm(:, j);
        y = Y_lab(:, i);
        v = isfinite(x) & isfinite(y);

        if nnz(v) >= 3 && std(x(v)) > 0 && std(y(v)) > 0
            [r, p] = corr(x(v), y(v), 'Type', 'Pearson');
            R_custom(i,j) = r;
            P_custom(i,j) = p;
        end
    end
end

%% ===== Colormap =====
anchors = [ ...
    0.00 0.20 0.80;
    1.00 1.00 1.00;
    0.80 0.10 0.10];
xs = [-1 0 1];
cmap = interp1(xs, anchors, linspace(-1,1,256), 'linear', 'extrap');

%% ===== Heatmap 그리기 =====
f = figure('Color','w','Position',[200 180 760 520]);
imagesc(R_custom, [-1 1]);
axis tight;
colormap(cmap);

cb = colorbar;
cb.Label.String = 'Pearson r';
set(cb, 'Ticks', -1:0.5:1, 'TickLabels', compose('%.1f', -1:0.5:1));

ax = gca;
set(ax, ...
    'XTick', 1:numel(ecm_display_names), ...
    'YTick', 1:numel(label_display_names), ...
    'XTickLabel', ecm_display_names, ...
    'YTickLabel', label_display_names, ...
    'TickLabelInterpreter', 'tex', ...
    'FontSize', 12, ...
    'FontWeight', 'bold', ...
    'LineWidth', 1.0);

xlabel('ECM Parameters', 'FontSize', 14, 'FontWeight', 'bold');
ylabel('SOH/SOH-x Labels', 'FontSize', 14, 'FontWeight', 'bold');
title('Correlation (Pearson) Heatmap', 'FontSize', 18, 'FontWeight', 'bold');

hold on;
[nr, nc] = size(R_custom);

for x = 0.5:1:nc+0.5
    plot([x x], [0.5 nr+0.5], 'w-', 'LineWidth', 1.0);
end
for y = 0.5:1:nr+0.5
    plot([0.5 nc+0.5], [y y], 'w-', 'LineWidth', 1.0);
end

for i = 1:nr
    for j = 1:nc
        r = R_custom(i,j);
        if isnan(r), continue; end

        idx = max(1, min(256, 1 + round((r+1)/2*255)));
        cc = cmap(idx,:);
        yiq = 0.299*cc(1) + 0.587*cc(2) + 0.114*cc(3);

        if yiq < 0.5
            tcol = [1 1 1];
        else
            tcol = [0 0 0];
        end

        text(j, i, sprintf('%.2f', r), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'FontSize', 11, ...
            'FontWeight', 'bold', ...
            'Color', tcol);
    end
end

box on;
set(gca, 'Layer', 'top');

%% ===== 저장 =====
savefig(f, fullfile(save_path, 'custom_corr_heatmap_ECM_vs_labels.fig'));
exportgraphics(f, fullfile(save_path, 'custom_corr_heatmap_ECM_vs_labels.png'), 'Resolution', 300);

T_custom_r = array2table(R_custom, ...
    'VariableNames', matlab.lang.makeValidName({'R0','R1','R2','tau1','tau2'}), ...
    'RowNames', matlab.lang.makeValidName({'QC2','QC40','DCIR10s','Rcharge'}));
T_custom_p = array2table(P_custom, ...
    'VariableNames', matlab.lang.makeValidName({'R0','R1','R2','tau1','tau2'}), ...
    'RowNames', matlab.lang.makeValidName({'QC2','QC40','DCIR10s','Rcharge'}));

writetable(T_custom_r, fullfile(save_path, 'custom_corr_heatmap_r.csv'), 'WriteRowNames', true);
writetable(T_custom_p, fullfile(save_path, 'custom_corr_heatmap_p.csv'), 'WriteRowNames', true);

disp('custom_corr_heatmap_ECM_vs_labels 저장 완료');

%% ===== 보조 함수 =====
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

function ids = extractCellIDs(names_in)
    names_in = cellstr(string(names_in));
    ids = nan(numel(names_in),1);

    for k = 1:numel(names_in)
        s = strtrim(names_in{k});

        tokx = regexp(s, '^[xX]\s*0*(\d+)', 'tokens', 'once');
        if ~isempty(tokx)
            ids(k) = str2double(tokx{1});
            continue;
        end

        tok1 = regexp(s, '\d+', 'match', 'once');
        if ~isempty(tok1)
            ids(k) = str2double(tok1);
        end
    end
end

function [ids_unique, idx_keep] = uniqueCellIDsFirst(ids)
    valid = ~isnan(ids);
    ids_valid = ids(valid);
    idx_valid = find(valid);

    [ids_unique, ia] = unique(ids_valid, 'stable');
    idx_keep = idx_valid(ia);
end