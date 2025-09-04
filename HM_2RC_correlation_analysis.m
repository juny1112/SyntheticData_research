%% ======================================================================
% 2RC 요약 → SOC90/50 Mean + 용량(QC/2, QC/40) → 12변수 상관분석
% - 입력: 2RC_results.mat (변수: all_summary)
% - 출력:
%   corr_pearson_12x12.csv : 상관계수
%   pval_pearson_12x12.csv : p-value
%   features_12vars.csv    : 최종 피처 테이블(행=셀, 열=12변수)
% ======================================================================
clear; clc; close all;

%% 경로 설정 -------------------------------------------------------------
results_mat = fullfile('G:\공유 드라이브\BSL_Data4\HNE_SOC_moving_cutoff_5_processed\SIM_parsed', ...
    '2RC_fitting','2RC_results.mat');
if ~isfile(results_mat)
    error('파일을 찾을 수 없습니다: %s', results_mat);
end
S = load(results_mat, 'all_summary');
if ~isfield(S,'all_summary')
    error('all_summary 변수가 없습니다: %s', results_mat);
end
all_summary = S.all_summary;
cell_names = fieldnames(all_summary);

save_path = 'G:\공유 드라이브\BSL_Data4\HNE_SOC_moving_cutoff_5_processed\SIM_parsed\2RC_fitting';

% 열 이름 고정
pNames = {'R0','R1','R2','tau1','tau2'};

% 셀 순서대로 QC/2, QC/40 값 입력
QC2_user  = [56.92, 45.37, 50.47, 48.85, 55.73, 52.09];
QC40_user = [58.94, 47.97, 52.15, 51.50, 57.29, 53.39];

%% 각 셀에서 SOC90/SOC50 Mean 파라미터 추출 ----------------------------
nC = numel(cell_names);
feat = nan(nC, 12); % [R0_90 ... tau2_90 | R0_50 ... tau2_50 | QC2 | QC40]
feat_names = [ ...
    strcat(pNames,'_90'), ...
    strcat(pNames,'_50'), ...
    {'QC2','QC40'}];

valid_row_90 = "SOC90_Mean";
valid_row_50 = "SOC50_Mean";

for i = 1:nC
    cname = cell_names{i};
    T = all_summary.(cname);
    if isempty(T) || height(T)~=12
        warning('(%s) 요약 테이블 형식이 예상과 다릅니다. 건너뜁니다.', cname);
        continue;
    end
    v90 = nan(1,5);
    v50 = nan(1,5);
    for k = 1:5
        colk = pNames{k};
        v90(k) = getVal(T, valid_row_90, colk);
        v50(k) = getVal(T, valid_row_50, colk);
    end
    feat(i, 1:5)  = v90; % SOC90 mean
    feat(i, 6:10) = v50; % SOC50 mean
    % QC2, QC40 는 입력 받음
end

%% --- 용량 입력 -------------------------------
% 셀 순서 확인용 출력
fprintf('셀 순서(%d개):\n', nC);
disp(strjoin(cell_names, ', '));
% 용량 값 주입
feat(:,11) = QC2_user;
feat(:,12) = QC40_user;

%% 유효 행 추리기(12개 변수 모두가 NaN이 아닌 행) -----------------------
row_valid = all(~isnan(feat), 2);
if nnz(row_valid) < 3
    warning('유효 샘플이 적습니다 (n=%d). 결과 해석에 주의하세요.', nnz(row_valid));
end
feat_valid  = feat(row_valid, :);
cells_valid = cell_names(row_valid);

%% 테이블로 정리 및 저장 -----------------------------------------------
Tfeat = array2table(feat_valid, 'VariableNames', feat_names, 'RowNames', cells_valid);
csv_out = fullfile(save_path, 'features_12vars.csv');
writetable(Tfeat, csv_out, 'WriteRowNames', true);
disp(['features_12vars.csv 저장 완료: ' csv_out]);

%% ── Pairplot: Histogram + KDE(대각), Scatter + r + 회귀선(비대각) ──
outfig1 = fullfile(save_path, 'corrmatrix_pair_kde.fig');
outpng1 = fullfile(save_path, 'corrmatrix_pair_kde.png');

% 유효 변수만 사용(결측 아닌 표본 ≥2)
col_ok = sum(~isnan(feat_valid), 1) >= 2;
X = feat_valid(:, col_ok);
vnames_raw = feat_names(col_ok);
vlabels = arrayfun(@prettyVarLabel, vnames_raw, 'UniformOutput', false);
p = size(X,2);

f1 = figure('Color','w','Position',[60 40 1300 950]);
[~, AX] = plotmatrix(X); % 축(grid)만 생성하기 위해 호출

% 색상 정의
darkBlue = [0 0.447 0.741];
lightBlue = [0.35 0.65 1.00];

for ii = 1:p
    for jj = 1:p
        cla(AX(ii,jj));
        set(AX(ii,jj),'Box','on','FontSize',9);
        if ii == jj
            % ── 대각: KDE 영역
            xdata = X(:,ii);
            xdata = xdata(isfinite(xdata));
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
            % ── 비대각: 산점도 + r + 회귀선
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
                c = polyfit(x(v), y(v), 1);
                xf = linspace(min(x(v)), max(x(v)), 100);
                yf = polyval(c, xf);
                plot(AX(ii,jj), xf, yf, '-', 'Color', darkBlue, 'LineWidth', 1.7);
            end
            hold(AX(ii,jj),'off');
        end
    end
end

% 라벨
for j = 1:p
    xlabel(AX(p,j), vlabels{j}, 'Interpreter','tex','FontSize',10,'FontWeight','bold');
    ylabel(AX(j,1), vlabels{j}, 'Interpreter','tex','FontSize',10,'FontWeight','bold');
    title (AX(j,j), vlabels{j}, 'Interpreter','tex','FontSize',10,'FontWeight','bold');
end
try, sgtitle('Correlation Matrix (Pearson, pairwise)'); end
savefig(f1, outfig1);
exportgraphics(f1, outpng1, 'Resolution', 220);

%% ── Heatmap: r + p + 95% CI (Blue→Cyan→Green, no yellow) + red star on top-right ──
outfig2 = fullfile(save_path, 'corr_heatmap_r_p_ci.fig');
outpng2 = fullfile(save_path, 'corr_heatmap_r_p_ci.png');

% 상관/유의확률
[Rmat, Pmat] = corr(X, 'Type','Pearson', 'Rows','pairwise');
pN = size(Rmat,1);

% 95% CI (Fisher z, n>=4)
N = zeros(pN); RL = nan(pN); RU = nan(pN);
zcrit = icdf('Normal',0.975,0,1);  % 1.96
for i = 1:pN
    for j = 1:pN
        v = isfinite(X(:,i)) & isfinite(X(:,j));
        N(i,j) = nnz(v);
        if i==j && N(i,j)>=2, RL(i,j)=1; RU(i,j)=1; continue; end
        r = Rmat(i,j); n = N(i,j);
        if n>=4 && ~isnan(r) && abs(r)<1
            z  = atanh(r); se = 1/sqrt(n-3);
            RL(i,j) = tanh(z - zcrit*se);
            RU(i,j) = tanh(z + zcrit*se);
        end
    end
end

% 블루→청록→그린 (노랑 제거)
anchors = [ ...
    0.031 0.188 0.420;  % deep blue (−1)
    0.031 0.317 0.611;  % blue
    0.031 0.443 0.682;  % light blue
    0.129 0.553 0.553;  % cyan (0 근처)
    0.251 0.682 0.361;  % green
    0.454 0.768 0.274]; % light green (+1)
cmap = interp1(linspace(-1,1,size(anchors,1)), anchors, linspace(-1,1,256), 'pchip');

% 그림
f2 = figure('Color','w','Position',[60 40 1200 950]);
imagesc(Rmat, [-1 1]); axis image;
colormap(cmap); cb = colorbar; cb.Label.String = 'Pearson r';

% 축 라벨: pairplot과 동일(vlabels 사용)
ax = gca;
lbls = cellfun(@(c) sprintf('%s\\newline%s', c{:}), vlabels, 'UniformOutput', false);

set(ax,'XTick',1:pN,'YTick',1:pN, ...
       'TickLabelInterpreter','tex','FontSize',10,'FontWeight','bold');
set(ax,'XTickLabel',lbls,'YTickLabel',lbls);
xtickangle(ax,45);
hold on;

% 옅은 흰 그리드
for k = 0.5:1:pN+0.5
    plot([0.5 pN+0.5],[k k],'w-','LineWidth',0.5);
    plot([k k],[0.5 pN+0.5],'w-','LineWidth',0.5);
end

% 셀 텍스트 + 빨간 별(우상단)
for i = 1:pN
    for j = 1:pN
        r = Rmat(i,j); if isnan(r), continue; end
        p = Pmat(i,j); lo = RL(i,j); hi = RU(i,j);

        % 배경 대비 글자색 자동(검/흰)
        idx = max(1, min(256, 1+round((r+1)/2*255)));
        cc  = cmap(idx,:); YIQ = 0.299*cc(1)+0.587*cc(2)+0.114*cc(3);
        tcol = [0 0 0]; if YIQ < 0.5, tcol = [1 1 1]; end

        % 본문(중앙)
        txt = sprintf('r=%.2f\np=%.3g\nCI[%.2f, %.2f]', r, p, lo, hi);
        text(j, i, txt, 'HorizontalAlignment','center', ...
            'VerticalAlignment','middle','FontSize',9, ...
            'FontWeight','bold','Color',tcol);

        % 별(정중앙-상단), 윗변에서 margin만큼 아래
        if isfinite(r) && abs(r) > 0.9 && i~=j
            ydir   = get(gca,'YDir');   % imagesc 기본: 'reverse'
            margin = 0.01;              % 셀 높이 기준으로 윗변에서 내려오는 거리(0.12~0.18로 취향 조절)
            if strcmpi(ydir,'reverse')
                y = i - 0.5 + margin;   % 윗변(i-0.5)에서 아래로 margin
            else
                y = i + 0.5 - margin;   % 윗변(i+0.5)에서 아래로 margin
            end
            text(j, y, '*', 'Color',[0.85 0 0], 'FontSize',12, 'FontWeight','bold', ...
                'HorizontalAlignment','center','VerticalAlignment','top');
        end
    end
end
title('2RC - Correlation Matrix heatmap  (r, p, 95%CI; * if |r|>0.9)');

savefig(f2, outfig2);
exportgraphics(f2, outpng2, 'Resolution', 220);


%% 상관 분석(피어슨) ----------------------------------------------------
[R_pearson, P_pearson] = corr(feat_valid, 'Type','Pearson', 'Rows','pairwise');

% MAT로 저장 (라벨도 같이 저장)
mat_out = fullfile(save_path, 'corr_pearson_12x12.mat');
save(mat_out, 'R_pearson', 'P_pearson', 'feat_names', 'cells_valid');

disp(['MAT 저장 완료: ' mat_out]);

fprintf('\n상관계수(일부):\n');
disp(array2table(R_pearson, 'VariableNames', feat_names, 'RowNames', feat_names));
fprintf('\n유의확률 p-value(일부):\n');
disp(array2table(P_pearson, 'VariableNames', feat_names, 'RowNames', feat_names));

%% -------- 보조 함수 ---------------------------------------------------
function y = getVal(T, rowName, colName)
    if ~ismember(rowName, string(T.Properties.RowNames)), y = NaN; return; end
    if ~ismember(colName, string(T.Properties.VariableNames)), y = NaN; return; end
    y = T{rowName, colName}; if isempty(y), y = NaN; end
end

function lab = prettyVarLabel(name0)
    name = char(name0);
    if strcmp(name,'QC2')
        lab = {'Q_{C/2}'}; return;
    elseif strcmp(name,'QC40')
        lab = {'Q_{C/40}'}; return;
    end
    socLine = ''; tok = regexp(name,'_(\d+)$','tokens','once');
    if ~isempty(tok)
        v = str2double(tok{1});
        if v==90 || v==50
            soc = v;
        elseif v==9
            soc = 90;
        elseif v==5
            soc = 50;
        else
            soc = NaN;
        end
        if ~isnan(soc)
            socLine = sprintf('(SOC %d%%)', soc);
            name = regexprep(name,'_(\d+)$','');
        end
    end
    switch string(name)
        case "R0", base = 'R_{0}';
        case "R1", base = 'R_{1}';
        case "R2", base = 'R_{2}';
        case "tau1", base = '\tau_{1}';
        case "tau2", base = '\tau_{2}';
        otherwise, base = strrep(name0,'_','\_');
    end
    if isempty(socLine), lab = {base};
    else, lab = {base; socLine};
    end
end
