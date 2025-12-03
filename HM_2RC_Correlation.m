%% ======================================================================
%  2RC(Tbl_ECM_mean) + 스칼라 → 피처 구성 → 상관분석 (Pearson)
%  - 입력:
%       2RC_results.mat (Tbl_ECM_mean 포함)
%          * Tbl_ECM_mean.RowNames : 셀 이름
%          * Tbl_ECM_mean.VarNames: SOC90_R0_mOhm, SOC90_R1_mOhm, ..., SOC30_tau2
%       사용자 스칼라(QC/2, QC/40, R1s, DCIR_1s, DCIR_10s, ΔDCIR, Rcharg,
%                    Rcharg_80_90_avg)
%  - 출력( save_path ):
%    features_allvars_full.csv / features_allvars_active.csv
%    corr_pearson.csv / pval_pearson.csv / corr_pearson.mat
%    corrmatrix_pair_kde.(fig|png), corr_heatmap_r_p_ci.(fig|png)
% ======================================================================
clear; clc; close all;

%% ── 설정(토글) ---------------------------------------------------------
SOC_use = [70];           % 분석에 포함할 SOC들 (예: [90 70 50 30], [90 50] 등)
DRAW_PAIRPLOT = true;     % Pairplot 그리기 여부

% 스칼라 활성화 토글
USE_DELTA          = true;   % ΔDCIR(=DCIR_10s - DCIR_1s 또는 직접 입력)
USE_RAW_DCIR10S    = false;  % DCIR_10s
USE_RAW_DCIR1S     = true;   % DCIR_1s
USE_RCHARG         = true;   % Rcharg
USE_R1S            = true;   % R1s
USE_RCHARG_8090    = true;   % Rcharg_80_90_avg  (추가)

%% ── 경로/파일 ----------------------------------------------------------
% 2RC_results.mat (Tbl_ECM_mean 사용)
matPath   = 'G:\공유 드라이브\BSL_Data4\HNE_agedcell_8_processed\SIM_parsed\2RC_fitting\2RC_results.mat';

% 결과 저장 경로
save_path = 'G:\공유 드라이브\BSL_Data4\HNE_agedcell_8_processed\SIM_parsed\2RC_fitting\Correlation_Anlaysis';
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
    soc = SOC_2RC(sIdx);          % 예: 90, 70, 50, 30
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

%% ── 사용자 스칼라 입력(셀 순서=Tbl_ECM_mean Row 순서) ------------------
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

% ΔDCIR (= DCIR_10s - DCIR_1s) 를 직접 넣고 싶으면 여기 입력
DCIRdelta_user = [0.28
0.26
0.4
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

% 추가: Rcharg_80_90_avg (셀 순서 동일, [nC x 1] 값으로 채워넣기)
% 일단 NaN으로 초기화 -> 실제 값으로 교체해서 사용
Rcharg_80_90_avg_user = [2.75
2.41
4.43
3.56
3.51
4.25
2.64
2.38
8.36
2.52
2.49
8.45
2.72];
% 예시로 직접 입력하려면 아래처럼 교체 가능
% Rcharg_80_90_avg_user = [ ... nC개 값 ... ];

% 길이 보정
QC2_user            = ensureLength(QC2_user,            nC);
QC40_user           = ensureLength(QC40_user,           nC);
DCIR1s_user         = ensureLength(DCIR1s_user,         nC);
DCIR10s_user        = ensureLength(DCIR10s_user,        nC);
Rcharg_user         = ensureLength(Rcharg_user,         nC);
R1s_user            = ensureLength(R1s_user,            nC);
DCIRdelta_user      = ensureLength(DCIRdelta_user,      nC);
Rcharg_80_90_avg_user = ensureLength(Rcharg_80_90_avg_user, nC);

%% ── DCIR 사용 가능 여부 체크 (보호 로직) -------------------------------
%  - DCIR_10s도 없고 ΔDCIR도 다 NaN이면: 두 피처 다 끄기
%  - DCIR_10s만 없고 ΔDCIR는 있으면: 10s만 끄고 ΔDCIR는 그대로 사용
if all(~isfinite(DCIR10s_user)) && all(~isfinite(DCIRdelta_user))
    warning('DCIR_10s와 ΔDCIR 값이 모두 NaN 입니다. USE_RAW_DCIR10S / USE_DELTA 를 자동으로 false로 변경합니다.');
    USE_RAW_DCIR10S = false;
    USE_DELTA       = false;
elseif all(~isfinite(DCIR10s_user))
    warning('DCIR_10s 값이 모두 NaN 입니다. USE_RAW_DCIR10S 만 false로 변경하고, ΔDCIR(직접 입력값)은 그대로 사용합니다.');
    USE_RAW_DCIR10S = false;
    % USE_DELTA는 그대로 둠 (직접 넣은 ΔDCIR 사용)
end

%% ── 피처명 구성 --------------------------------------------------------
scalar_pool_names   = {'QC2','QC40','DCIR_1s','DCIR_10s', ...
                       'DCIR_delta_10s_1s','Rcharg','R1s', ...
                       'Rcharg_80_90_avg'};    % ★ 추가

% active: 순서는 여기서 조정
active_scalar_names = {'QC2','QC40'};
if USE_RAW_DCIR1S,      active_scalar_names{end+1} = 'DCIR_1s';           end
if USE_RAW_DCIR10S,     active_scalar_names{end+1} = 'DCIR_10s';          end
if USE_DELTA,           active_scalar_names{end+1} = 'DCIR_delta_10s_1s'; end
if USE_RCHARG,          active_scalar_names{end+1} = 'Rcharg';            end
if USE_R1S,             active_scalar_names{end+1} = 'R1s';               end
if USE_RCHARG_8090,     active_scalar_names{end+1} = 'Rcharg_80_90_avg';  end

feat_full_names   = [soc_param_names_2RC, scalar_pool_names];
feat_active_names = [soc_param_names_2RC, active_scalar_names];

%% ── 피처 행렬 생성 (2RC + 스칼라) -------------------------------------
feat_full   = nan(nC, numel(feat_full_names));
feat_active = nan(nC, numel(feat_active_names));

for i = 1:nC
    % 2RC SOC별 값 (선택된 SOC_use만)
    socvals_2RC = nan(1, numel(soc_param_names_2RC));
    c = 1;
    for jj = 1:nSOC
        sIdx = idxSOC_2RC(jj);     % SOC_2RC 기준 인덱스
        for k = 1:numel(pNames_2RC)
            socvals_2RC(c) = P2RC.(pNames_2RC{k})(i, sIdx);
            c = c + 1;
        end
    end

    % ΔDCIR: 우선순위
    %  1) DCIRdelta_user(i) 가 finite면 그 값 사용
    %  2) 아니면 DCIR10s_user, DCIR1s_user 둘 다 finite면 10s - 1s
    %  3) 아니면 NaN
    if isfinite(DCIRdelta_user(i))
        dDCIR = DCIRdelta_user(i);
    elseif isfinite(DCIR10s_user(i)) && isfinite(DCIR1s_user(i))
        dDCIR = DCIR10s_user(i) - DCIR1s_user(i);
    else
        dDCIR = NaN;
    end

    % 스칼라 Map (키 순서 = scalar_pool_names 순서)
    scalar_vals = containers.Map( ...
        scalar_pool_names, ...
        {QC2_user(i), ...                 % 'QC2'
         QC40_user(i), ...                % 'QC40'
         DCIR1s_user(i), ...              % 'DCIR_1s'
         DCIR10s_user(i), ...             % 'DCIR_10s'
         dDCIR, ...                       % 'DCIR_delta_10s_1s'
         Rcharg_user(i), ...              % 'Rcharg'
         R1s_user(i), ...                 % 'R1s'
         Rcharg_80_90_avg_user(i)} ...   % 'Rcharg_80_90_avg'
    );

    % 전체/활성 피처 행
    row_full   = [socvals_2RC, cellfun(@(nm) scalar_vals(nm), scalar_pool_names)];
    row_active = [socvals_2RC, cellfun(@(nm) scalar_vals(nm), active_scalar_names)];

    feat_full(i,:)   = row_full;
    feat_active(i,:) = row_active;
end

%% ── 유효 행 필터(분석은 활성 기준) ------------------------------------
row_valid_active   = all(~isnan(feat_active),2);
feat_active_valid  = feat_active(row_valid_active, :);
cells_active_valid = cell_names(row_valid_active);

%% ── CSV 저장 -----------------------------------------------------------
T_full   = array2table(feat_full,   'VariableNames', feat_full_names,   'RowNames', cell_names);
T_active = array2table(feat_active_valid, 'VariableNames', feat_active_names, 'RowNames', cells_active_valid);

csv_full   = fullfile(save_path, 'features_allvars_full.csv');
csv_active = fullfile(save_path, 'features_allvars_active.csv');
writetable(T_full,   csv_full,   'WriteRowNames', true);
writetable(T_active, csv_active, 'WriteRowNames', true);
disp(['features_allvars_full.csv 저장: ' csv_full]);
disp(['features_allvars_active.csv 저장: ' csv_active]);

%% ── 상관분석/그림 (활성 집합) ------------------------------------------
X = feat_active_valid;
vnames_active = feat_active_names;

% Pairplot
outfig1 = fullfile(save_path, 'corrmatrix_pair_kde.fig');
outpng1 = fullfile(save_path, 'corrmatrix_pair_kde.png');

p = size(X,2);
if DRAW_PAIRPLOT && p >= 2
    f1 = figure('Color','w','Position',[60 40 1300 950]);
    [~, AX] = plotmatrix(X); % 축만 생성
    darkBlue  = [0 0.447 0.741];
    lightBlue = [0.35 0.65 1.00];

    for ii = 1:p
        for jj = 1:p
            cla(AX(ii,jj));
            set(AX(ii,jj),'Box','on','FontSize',9);
            if ii == jj
                xdata = X(:,ii); xdata = xdata(isfinite(xdata));
                if numel(xdata) >= 2
                    hold(AX(ii,jj),'on');
                    try
                        [f, xi] = ksdensity(xdata);
                        fill(AX(ii,jj), xi, f, lightBlue, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
                        plot(AX(ii,jj), xi, f, '-', 'Color', darkBlue, 'LineWidth', 2.0);
                    catch
                    end
                    hold(AX(ii,jj),'off');
                    if ~isempty(xdata), xlim(AX(ii,jj), [min(xdata) max(xdata)]); end
                end
            else
                x = X(:,jj); y = X(:,ii);
                v = isfinite(x) & isfinite(y);
                if any(v)
                    hold(AX(ii,jj),'on');
                    scatter(AX(ii,jj), x(v), y(v), 15, 'k', 'filled');
                end
                n = nnz(v);
                if n >= 2 && std(x(v))>0 && std(y(v))>0
                    C = corrcoef(x(v), y(v));
                    r = C(1,2);
                    text(AX(ii,jj), 0.05, 0.92, sprintf('r = %.2f', r), ...
                        'Units','normalized','FontSize',10,'FontWeight','bold', ...
                        'BackgroundColor','w','Margin',1);
                end
                if n >= 3 && std(x(v))>0
                    c  = polyfit(x(v), y(v), 1);
                    xf = linspace(min(x(v)), max(x(v)), 100);
                    yf = polyval(c, xf);
                    plot(AX(ii,jj), xf, yf, '-', 'Color', darkBlue, 'LineWidth', 1.7);
                end
                hold(AX(ii,jj),'off');
            end
        end
    end

    vlabels = arrayfun(@prettyVarLabel, vnames_active, 'UniformOutput', false);
    for j = 1:p
        xlabel(AX(p,j), vlabels{j}, 'Interpreter','tex','FontSize',10,'FontWeight','bold');
        ylabel(AX(j,1), vlabels{j}, 'Interpreter','tex','FontSize',10,'FontWeight','bold');
        title (AX(j,j), vlabels{j}, 'Interpreter','tex','FontSize',10,'FontWeight','bold');
    end
    try, sgtitle('Correlation Matrix (Pearson, pairwise)'); end
    savefig(f1, outfig1);
    exportgraphics(f1, outpng1, 'Resolution', 220);
end

% Heatmap (r, p, 95% CI) —— BWR(Blue-White-Red) 컬러맵 적용
outfig2 = fullfile(save_path, 'corr_heatmap_r_p_ci.fig');
outpng2 = fullfile(save_path, 'corr_heatmap_r_p_ci.png');

[Rmat, Pmat] = corr(X, 'Type','Pearson', 'Rows','pairwise');
pN = size(Rmat,1);

Npair  = zeros(pN); RL = nan(pN); RU = nan(pN);
zcrit = icdf('Normal',0.975,0,1);  % 1.96
for i = 1:pN
    for j = 1:pN
        v = isfinite(X(:,i)) & isfinite(X(:,j));
        Npair(i,j) = nnz(v);
        if i==j && Npair(i,j)>=2, RL(i,j)=1; RU(i,j)=1; continue; end
        r = Rmat(i,j); n = Npair(i,j);
        if n>=4 && ~isnan(r) && abs(r)<1
            z  = atanh(r); se = 1/sqrt(n-3);
            RL(i,j) = tanh(z - zcrit*se);
            RU(i,j) = tanh(z + zcrit*se);
        end
    end
end

% Blue(-1) – White(0) – Red(+1) 선형 컬러맵
anchors = [ ...
    0.00 0.20 0.80;   % blue at -1
    1.00 1.00 1.00;   % white at 0
    0.80 0.10 0.10];  % red  at +1
xs   = [-1 0 1];
cmap = interp1(xs, anchors, linspace(-1,1,256), 'linear','extrap');

f2 = figure('Color','w','Position',[60 40 1200 950]);
imagesc(Rmat, [-1 1]); axis image;
colormap(cmap);
cb = colorbar; cb.Label.String = 'Pearson r';
set(cb,'Ticks',-1:0.5:1,'TickLabels',compose('%.1f',-1:0.5:1));

vlabels = arrayfun(@prettyVarLabel, vnames_active, 'UniformOutput', false);
ax = gca;
lbls = cellfun(@(c) sprintf('%s\\newline%s', c{:}), vlabels, 'UniformOutput', false);
set(ax,'XTick',1:pN,'YTick',1:pN, ...
       'TickLabelInterpreter','tex','FontSize',10,'FontWeight','bold');
set(ax,'XTickLabel',lbls,'YTickLabel',lbls);
xtickangle(ax,45);
hold on;

for k = 0.5:1:pN+0.5
    plot([0.5 pN+0.5],[k k],'w-','LineWidth',0.5);
    plot([k k],[0.5 pN+0.5],'w-','LineWidth',0.5);
end

for i = 1:pN
    for j = 1:pN
        r = Rmat(i,j); if isnan(r), continue; end
        pval = Pmat(i,j); lo = RL(i,j); hi = RU(i,j);

        idx = max(1, min(256, 1+round((r+1)/2*255)));
        cc  = cmap(idx,:); YIQ = 0.299*cc(1)+0.587*cc(2)+0.114*cc(3);
        tcol = [0 0 0]; if YIQ < 0.5, tcol = [1 1 1]; end

        txt = sprintf('r=%.2f\np=%.3g\nCI[%.2f, %.2f]', r, pval, lo, hi);
        text(j, i, txt, 'HorizontalAlignment','center', ...
            'VerticalAlignment','middle','FontSize',9, ...
            'FontWeight','bold','Color',tcol);

        if isfinite(r) && abs(r) > 0.9 && i~=j
            ydir   = get(gca,'YDir');
            margin = 0.01;
            if strcmpi(ydir,'reverse'), y = i - 0.5 + margin;
            else,                       y = i + 0.5 - margin; end
            text(j, y, '*', 'Color',[1 1 0], 'FontSize',12, 'FontWeight','bold', ...
                'HorizontalAlignment','center','VerticalAlignment','top');
        end
    end
end
title('2RC - Correlation heatmap (r, p, 95% CI; * if |r|>0.9)');
savefig(f2, outfig2);
exportgraphics(f2, outpng2, 'Resolution', 220);

%% ── 상관 행렬 저장 -----------------------------------------------------
[R_pearson, P_pearson] = deal(Rmat, Pmat);
Tcorr = array2table(R_pearson, 'VariableNames', vnames_active, 'RowNames', vnames_active);
Tpval = array2table(P_pearson,  'VariableNames', vnames_active, 'RowNames', vnames_active);

writetable(Tcorr, fullfile(save_path, 'corr_pearson.csv'), 'WriteRowNames', true);
writetable(Tpval,  fullfile(save_path, 'pval_pearson.csv'), 'WriteRowNames', true);
save(fullfile(save_path, 'corr_pearson.mat'), 'R_pearson', 'P_pearson', 'vnames_active', 'cells_active_valid');

disp('corr_pearson.csv / pval_pearson.csv / corr_pearson.mat 저장 완료');

%% ========================= 보조 함수들 =================================
function v = ensureLength(v, n)
    v = v(:).';
    if numel(v) < n
        v = [v, nan(1, n - numel(v))];
    elseif numel(v) > n
        v = v(1:n);
    end
end

function lab = prettyVarLabel(name0)
    name = char(name0);
    switch name
        case 'QC2',                 lab = {'Q_{C/2}'}; return;
        case 'QC40',                lab = {'Q_{C/40}'}; return;
        case 'R1s',                 lab = {'R_{1s}'}; return;
        case 'Rcharg',              lab = {'R_{charg}'}; return;
        case 'Rcharg_80_90_avg',    lab = {'R_{charg,80-90}','(avg)'}; return;
        case 'DCIR_1s',             lab = {'DCIR (1 s)'}; return;
        case 'DCIR_10s',            lab = {'DCIR (10 s)'}; return;
        case 'DCIR_delta_10s_1s',   lab = {'\DeltaDCIR (10-1 s)'}; return;
    end
    socLine = '';
    tok = regexp(name,'_(\d+)$','tokens','once');
    if ~isempty(tok)
        v = str2double(tok{1});
        if ismember(v, [90 70 50 30])
            socLine = sprintf('(SOC %d%%)', v);
            name = regexprep(name,'_(\d+)$','');
        end
    end
    switch string(name)
        case "R0",   base = 'R_{0}';
        case "R1",   base = 'R_{1}';
        case "R2",   base = 'R_{2}';
        case "tau1", base = '\tau_{1}';
        case "tau2", base = '\tau_{2}';
        otherwise,   base = strrep(name0,'_','\_');
    end
    if isempty(socLine), lab = {base};
    else, lab = {base; socLine};
    end
end
