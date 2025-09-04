%% ======================================================================
%  1RC 요약 → SOC90/50 Mean + 용량(QC/2, QC/40) → 8변수 상관분석
%  - 입력: 1RC_results.mat  (변수: all_summary)
%  - 출력(저장 경로 = save_path):
%      corrmatrix_pair_kde.(fig|png)  : pairplot
%      corr_heatmap_r_p_ci.(fig|png)  : heatmap
%      corr_pearson_8x8.mat           : R, P, 변수명, 유효셀명
%      features_8vars.csv             : 최종 피처 테이블
% ======================================================================
clear; clc; close all;

%% 경로 설정 -------------------------------------------------------------
base_dir    = 'G:\공유 드라이브\BSL_Data4\HNE_SOC_moving_cutoff_5_processed\SIM_parsed';
results_mat = fullfile(base_dir, '1RC_fitting','1RC_results.mat');
if ~isfile(results_mat)
    error('파일을 찾을 수 없습니다: %s', results_mat);
end

% 모든 결과 저장 경로
save_path = fullfile(base_dir, '1RC_fitting');
if ~exist(save_path,'dir'); mkdir(save_path); end

S = load(results_mat, 'all_summary');
if ~isfield(S,'all_summary')
    error('all_summary 변수가 없습니다: %s', results_mat);
end
all_summary = S.all_summary;
cell_names  = fieldnames(all_summary);

% 열 이름 고정 (1RC)
pNames = {'R0','R1','tau1'};

% ── 셀 순서대로 QC/2, QC/40 값 입력(필요시 수정) -----------------------
QC2_user  = [56.92, 45.37, 50.47, 48.85, 55.73, 52.09];
QC40_user = [58.94, 47.97, 52.15, 51.50, 57.29, 53.39];

%% 각 셀에서 SOC90/SOC50 Mean 파라미터 추출 ----------------------------
nC = numel(cell_names);
feat = nan(nC, 8);     % [R0_90 R1_90 tau1_90 | R0_50 R1_50 tau1_50 | QC2 | QC40]
feat_names = [ ...
    strcat(pNames,'_90'), ...
    strcat(pNames,'_50'), ...
    {'QC2','QC40'}];

valid_row_90 = "SOC90_Mean";
valid_row_50 = "SOC50_Mean";

for i = 1:nC
    cname = cell_names{i};
    T = all_summary.(cname);                % 12x3 table (R0,R1,tau1)
    if isempty(T) || height(T)~=12
        warning('(%s) 요약 테이블 형식이 예상과 다릅니다. 건너뜁니다.', cname);
        continue;
    end

    v90 = nan(1,3); v50 = nan(1,3);
    for k = 1:3
        colk = pNames{k};
        v90(k) = getVal(T, valid_row_90, colk);
        v50(k) = getVal(T, valid_row_50, colk);
    end

    feat(i, 1:3) = v90;   % SOC90 mean
    feat(i, 4:6) = v50;   % SOC50 mean
end

%% --- 용량 입력 --------------------------------------------------------
fprintf('셀 순서(%d개):\n', nC);
disp(strjoin(cell_names, ', '));
feat(:,7) = QC2_user(:);
feat(:,8) = QC40_user(:);

%% 유효 행 추리기(8개 변수 모두 NaN 아님) ------------------------------
row_valid = all(~isnan(feat), 2);
if nnz(row_valid) < 3
    warning('유효 샘플이 적습니다 (n=%d). 결과 해석에 주의하세요.', nnz(row_valid));
end
feat_valid  = feat(row_valid, :);
cells_valid = cell_names(row_valid);

%% 테이블 저장 ----------------------------------------------------------
Tfeat = array2table(feat_valid, 'VariableNames', feat_names, 'RowNames', cells_valid);
writetable(Tfeat, fullfile(save_path,'features_8vars.csv'), 'WriteRowNames', true);
disp('features_8vars.csv 저장 완료');

%% ── Pairplot: 대각 KDE영역 + 진파랑 KDE선 / 비대각 검정점 + r + 파란 회귀선 ──
outfig1 = fullfile(save_path, 'corrmatrix_pair_kde.fig');
outpng1 = fullfile(save_path, 'corrmatrix_pair_kde.png');

% 유효 변수만 사용(결측 아닌 표본 ≥2)
col_ok = sum(~isnan(feat_valid), 1) >= 2;
X = feat_valid(:, col_ok);
vnames_raw = feat_names(col_ok);
vlabels = arrayfun(@prettyVarLabel, vnames_raw, 'UniformOutput', false);
p = size(X,2);

f1 = figure('Color','w','Position',[60 40 1300 950]);
[~, AX] = plotmatrix(X); % 축(grid)만 생성

% 색상 정의
darkBlue  = [0 0.447 0.741];
lightBlue = [0.35 0.65 1.00];

for ii = 1:p
    for jj = 1:p
        ax = AX(ii,jj);
        if ~isgraphics(ax,'axes'), continue; end
        cla(ax); set(ax,'Box','on','FontSize',9);

        if ii == jj
            % ── 대각: KDE 영역 + 라인
            xdata = X(:,ii); xdata = xdata(isfinite(xdata));
            if numel(xdata) >= 2
                hold(ax,'on');
                try
                    [f, xi] = ksdensity(xdata);
                    fill(ax, xi, f, lightBlue, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
                    plot(ax, xi, f, '-', 'Color', darkBlue, 'LineWidth', 2.0);
                catch
                end
                hold(ax,'off');
                if ~isempty(xdata), xlim(ax, [min(xdata) max(xdata)]); end
            end
        else
            % ── 비대각: 산점도(검정) + r + 파란 회귀선
            x = X(:,jj); y = X(:,ii); v = isfinite(x) & isfinite(y);
            if any(v)
                hold(ax,'on');
                scatter(ax, x(v), y(v), 15, 'k', 'filled');
            end
            n = nnz(v);
            if n >= 2 && std(x(v))>0 && std(y(v))>0
                C = corrcoef(x(v), y(v));
                r = C(1,2);
                text(ax, 0.05, 0.92, sprintf('r = %.2f', r), ...
                    'Units','normalized','FontSize',10,'FontWeight','bold', ...
                    'BackgroundColor','w','Margin',1);
            end
            if n >= 3 && std(x(v))>0
                c = polyfit(x(v), y(v), 1);
                xf = linspace(min(x(v)), max(x(v)), 100);
                yf = polyval(c, xf);
                plot(ax, xf, yf, '-', 'Color', darkBlue, 'LineWidth', 1.7);
            end
            hold(ax,'off');
        end
    end
end

% 라벨(페어플롯 스타일)
for j = 1:p
    if isgraphics(AX(p,j),'axes'), xlabel(AX(p,j), vlabels{j}, 'Interpreter','tex','FontSize',10,'FontWeight','bold'); end
    if isgraphics(AX(j,1),'axes'), ylabel(AX(j,1), vlabels{j}, 'Interpreter','tex','FontSize',10,'FontWeight','bold'); end
    if isgraphics(AX(j,j),'axes'), title (AX(j,j),  vlabels{j}, 'Interpreter','tex','FontSize',10,'FontWeight','bold'); end
end
try, sgtitle('Correlation Matrix (Pearson, pairwise)'); end
savefig(f1, outfig1);
exportgraphics(f1, outpng1, 'Resolution', 220);

%% ── Heatmap: r + p + 95% CI (Blue→Cyan→Green, no yellow) + red star(상단중앙) ──
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

% 축 라벨: pairplot과 동일(두 줄 라벨)
ax = gca;
% vlabels -> lbls('\newline'로 합치기; 1줄 라벨(QC2/40)도 대응)
lbls = cell(size(vlabels));
for k = 1:numel(vlabels)
    c = vlabels{k};
    if numel(c) >= 2, lbls{k} = sprintf('%s\\newline%s', c{1}, c{2});
    else,             lbls{k} = c{1};
    end
end

set(ax,'XTick',1:pN,'YTick',1:pN, ...
       'TickLabelInterpreter','tex','FontSize',10,'FontWeight','bold');
set(ax,'XTickLabel',lbls,'YTickLabel',lbls);
xtickangle(ax,45); hold on;

% 옅은 흰 그리드
for k = 0.5:1:pN+0.5
    plot([0.5 pN+0.5],[k k],'w-','LineWidth',0.5);
    plot([k k],[0.5 pN+0.5],'w-','LineWidth',0.5);
end

% 셀 텍스트 + 빨간 별(정중앙-상단)
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

        % 별(윗변에서 margin만큼 아래)
        if isfinite(r) && abs(r) > 0.9 && i~=j
            ydir   = get(gca,'YDir');   % imagesc 기본: 'reverse'
            margin = 0.01;
            if strcmpi(ydir,'reverse')
                y = i - 0.5 + margin;   % 윗변(i-0.5)에서 아래로 margin
            else
                y = i + 0.5 - margin;
            end
            text(j, y, '*', 'Color',[0.85 0 0], 'FontSize',12, 'FontWeight','bold', ...
                'HorizontalAlignment','center','VerticalAlignment','top');
        end
    end
end
title('1RC - Correlation Matrix heatmap  (r, p, 95%CI; * if |r|>0.9)');

savefig(f2, outfig2);
exportgraphics(f2, outpng2, 'Resolution', 220);

%% 상관 분석(피어슨) ----------------------------------------------------
[R_pearson, P_pearson] = corr(feat_valid, 'Type','Pearson', 'Rows','pairwise');

% MAT로 저장 (라벨도 같이 저장)
mat_out = fullfile(save_path, 'corr_pearson_8x8.mat');
save(mat_out, 'R_pearson', 'P_pearson', 'feat_names', 'cells_valid');
disp(['MAT 저장 완료: ' mat_out]);

% 콘솔 프리뷰
fprintf('\n상관계수(미리보기):\n');
disp(array2table(R_pearson, 'VariableNames', feat_names, 'RowNames', feat_names));
fprintf('\n유의확률 p-value(미리보기):\n');
disp(array2table(P_pearson, 'VariableNames', feat_names, 'RowNames', feat_names));

%% -------- 보조 함수 ---------------------------------------------------
function y = getVal(T, rowName, colName)
    if ~ismember(rowName, string(T.Properties.RowNames)), y = NaN; return; end
    if ~ismember(colName, string(T.Properties.VariableNames)), y = NaN; return; end
    y = T{rowName, colName};
    if isempty(y), y = NaN; end
end

% 두 줄 라벨 생성(페어플롯/히트맵 공통)
function lab = prettyVarLabel(name0)
    % name0 예: 'R0_90','R1_50','tau1_90','QC2','QC40','R0_5' 등
    name = char(name0);

    % 용량 표기
    if strcmp(name,'QC2'),  lab = {'Q_{C/2}'};  return; end
    if strcmp(name,'QC40'), lab = {'Q_{C/40}'}; return; end

    % SOC 접미사 (_90/_50, 예외: _9/_5)
    socLine = '';
    tok = regexp(name,'_(\d+)$','tokens','once');
    if ~isempty(tok)
        v = str2double(tok{1});
        if     v==90, soc=90;
        elseif v==50, soc=50;
        elseif v==9,  soc=90;
        elseif v==5,  soc=50;
        else,  soc=NaN;
        end
        if ~isnan(soc)
            socLine = sprintf('(SOC %d%%)', soc);
            name = regexprep(name,'_(\d+)$','');
        end
    end

    % 본체 라벨
    switch string(name)
        case "R0",   base = 'R_{0}';
        case "R1",   base = 'R_{1}';
        case "tau1", base = '\tau_{1}';
        otherwise,   base = strrep(name0,'_','\_');
    end

    if isempty(socLine), lab = {base}; else, lab = {base; socLine}; end
end
