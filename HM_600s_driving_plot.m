%% ======================================================================
%  HM_2RC_Loadwise_subplot_bySOC_colorSOH  (파일명 순서대로 QC40 매핑)
%  + (ADD) 부하별 Boxchart : SOC별 3×2 subplot로 분포 확인
%
%  - 입력: 2RC_results_600s.mat (all_para_hats 포함)
%  - SOC별 3×2 subplot: x축=주행부하(US06~HW2)
%  - 선 색:
%       * QC40가 주어지면 QC40로 컬러매핑 + colorbar(QC40)
%       * QC40 = [] 이면 "셀 인덱스(1..N)"로 컬러매핑 + colorbar(Cell index)  [옵션2]
%  - y축: 무조건 0부터 시작
%  - QC40는 "파일명 정렬 순서(sort)"와 동일한 순서로 준비되어 있다고 가정
%% ======================================================================
clc; clear; close all;

set(groot,'defaultTextInterpreter','tex');
set(groot,'defaultAxesTickLabelInterpreter','none');
set(groot,'defaultLegendInterpreter','none');

%% ── 경로 설정 ────────────────────────────────────────────────────────
save_path = 'G:\공유 드라이브\BSL_Data4\HNE_fresh_integrated_7_Drivingprocessed\2RC_fitting_600s';
mat_file  = fullfile(save_path,'2RC_results_600s.mat');
if ~isfile(mat_file)
    error('2RC_results_600s.mat 를 찾을 수 없습니다: %s', mat_file);
end

outDir = fullfile(save_path,'parameter_load');
if ~exist(outDir,'dir'), mkdir(outDir); end

%% ── SOH (QC40): "파일명 순서"대로 들어있다고 가정 ────────────────────
% ✅ QC40를 비우면([]) 자동으로 "셀 인덱스(1..N)"로 컬러매핑 (옵션2)

% QC40 = [57.49
% 57.57
% 54
% 52.22
% 53.45
% 51.28
% 57.91
% 56.51
% 42.14
% 57.27
% 57.18
% 43.92
% 58.4];

QC40 = [];

capLabel_QC40 = "Capacity (QC/40, Ah)";
capLabel_IDX  = "Cell index (sorted file order)";

%% ── SOC / Load 정의 ──────────────────────────────────────────────────
socVals = [90 70 50 30];

loadNames = {'US06','UDDS','HWFET','WLTP','CITY1','CITY2','HW1','HW2'};
nLoads   = numel(loadNames);
nSOC     = numel(socVals);
nSeg_expected = nSOC*nLoads;  % 32

%% ── 2RC 결과 로드 ────────────────────────────────────────────────────
S = load(mat_file,'all_para_hats');
if ~isfield(S,'all_para_hats')
    error('mat 파일에 all_para_hats가 없습니다: %s', mat_file);
end
all_para_hats = S.all_para_hats;

cellNames_raw = fieldnames(all_para_hats);
if isempty(cellNames_raw)
    error('all_para_hats 에 셀 데이터가 없습니다.');
end

% ✅ 파일명 순서대로 정렬
cellNames_raw_sorted = sort(cellNames_raw);

%% ── 셀별 SOC×Load×Param 텐서 구성 ─────────────────────────────────────
param_all_cells = struct();
cells_for_plot  = strings(0,1);

% capVec는 "컬러매핑용 스칼라" (QC40 또는 셀 인덱스)
capVec = nan(0,1);

for ci = 1:numel(cellNames_raw_sorted)
    key  = cellNames_raw_sorted{ci};
    Praw = all_para_hats.(key);  % [nSeg × 8] = [R0 R1 R2 tau1 tau2 RMSE exitflag iter]

    if size(Praw,2) < 6
        warning('(%s) 열<6이라 skip (size=%dx%d)', key, size(Praw,1), size(Praw,2));
        continue;
    end

    nUse = min(size(Praw,1), nSeg_expected);
    Puse = Praw(1:nUse, 1:6);

    T = nan(nSOC, nLoads, 6);
    for idx = 1:nUse
        socIdx  = ceil(idx / nLoads);          % 1..4
        loadIdx = idx - (socIdx-1)*nLoads;     % 1..8
        if socIdx<1 || socIdx>nSOC || loadIdx<1 || loadIdx>nLoads, continue; end
        T(socIdx, loadIdx, :) = Puse(idx,:);
    end

    param_all_cells.(key) = T;
    cells_for_plot(end+1,1) = string(key); %#ok<SAGROW>

    % ===== capVec 매칭 규칙 =====
    if ~isempty(QC40)
        % QC40가 있으면: 파일명 정렬 순서로 대응
        if ci <= numel(QC40)
            capVec(end+1,1) = QC40(ci); %#ok<SAGROW>
        else
            capVec(end+1,1) = NaN;      %#ok<SAGROW>
        end
    else
        % QC40가 비어 있으면: 셀 인덱스(1..N) 사용 (옵션2)
        capVec(end+1,1) = ci; %#ok<SAGROW>
    end
end

if isempty(cells_for_plot)
    error('플롯 가능한 셀이 없습니다.');
end

% ===== capVec 보정 =====
% QC40 모드인데 길이 부족 → NaN 생김. 이때도 "셀 인덱스"로 fallback(옵션2)로 돌려버리는게 가장 안전.
if any(~isfinite(capVec))
    if ~isempty(QC40)
        warning("QC40 길이 부족/불일치로 capVec에 NaN이 있습니다. → 자동으로 '셀 인덱스' 컬러매핑으로 전환합니다.");
        capVec = (1:numel(capVec))';
        useQC40 = false;
    else
        % QC40 자체가 비어있던 케이스는 여기로 안 와야 정상
        capVec(~isfinite(capVec)) = mean(capVec(isfinite(capVec)));
        useQC40 = false;
    end
else
    useQC40 = ~isempty(QC40);
end

% colorbar 라벨 자동 선택
if useQC40
    capLabel = capLabel_QC40;
else
    capLabel = capLabel_IDX;
end

%% ── 컬러맵 (네 스타일 유지) ───────────────────────────────────────────
Nmap = 256;
anchors = [0.88 0.16 0.24; 0.83 0.70 0.86; 0.16 0.38 0.92];
x0  = [0 0.5 1];
xi = linspace(0,1,Nmap)';

cmap = [interp1(x0,anchors(:,1),xi,'pchip'), ...
        interp1(x0,anchors(:,2),xi,'pchip'), ...
        interp1(x0,anchors(:,3),xi,'pchip')];
cmap = min(max(cmap,0),1);

hsvv = rgb2hsv(cmap);
hsvv(:,2) = max(0.35,hsvv(:,2));
hsvv(:,2) = min(1.0,hsvv(:,2)*1.2);
hsvv(:,3) = max(0.75,hsvv(:,3)*0.95);
cmap = hsv2rgb(hsvv);

capMin = min(capVec);
capMax = max(capVec);

mapColor = @(v) cmap( max(1, min(Nmap, 1 + round((v-capMin)/max(capMax-capMin,eps)*(Nmap-1)))), : );

%% ── SOC별 3×2 subplot 생성 (라인 플롯) ────────────────────────────────
paramOrder  = [1 6 2 4 3 5];           % pIdx: R0,RMSE,R1,tau1,R2,tau2
paramTitles = {'R_0 [m\Omega]','RMSE [V]', ...
               'R_1 [m\Omega]','\tau_1 [s]', ...
               'R_2 [m\Omega]','\tau_2 [s]'};
scaleFactor = [1e3,1e3,1e3,1,1,1];    % pIdx 기준 (R0~R2만 mΩ)
x = 1:nLoads;

for s = 1:nSOC
    socVal = socVals(s);

    fig = figure('Color','w', ...
        'Name', sprintf('SOC%d – loadwise params (color=%s)', socVal, capLabel), ...
        'Position',[80 80 1700 850]);

    tl = tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

    for k = 1:6
        p = paramOrder(k);

        ax = nexttile(tl); hold(ax,'on'); grid(ax,'on');
        Y_all = [];

        for ci = 1:numel(cells_for_plot)
            key = char(cells_for_plot(ci));
            T = param_all_cells.(key);          % [4×8×6]

            Y = squeeze(T(s,:,p));              % [1×8]
            if all(isnan(Y)), continue; end

            Y = Y * scaleFactor(p);
            Y_all = [Y_all, Y]; %#ok<AGROW>

            col = mapColor(capVec(ci));

            plot(ax, x, Y, '-o', ...
                'LineWidth', 1.5, 'MarkerSize', 5, ...
                'Color', col, 'MarkerFaceColor', col, ...
                'DisplayName', key);
        end

        xlim(ax, [1 nLoads]);
        xticks(ax, 1:nLoads);
        xticklabels(ax, loadNames);
        xtickangle(ax, 45);

        xlabel(ax, 'Driving load');
        title(ax, paramTitles{k}, 'Interpreter','tex');

        % y축: 무조건 0부터
        if ~isempty(Y_all)
            ymax = max(Y_all, [], 'omitnan');
            if ~isfinite(ymax) || ymax <= 0, ymax = 1; end
            ylim(ax, [0, ymax*1.05]);
        else
            ylim(ax, [0 1]);
        end

        if k == 1
            legend(ax, 'Location','bestoutside', 'Interpreter','none');
        end
    end

    title(tl, sprintf('SOC %d – 2RC params vs driving load (color = %s)', socVal, capLabel), ...
        'Interpreter','none');

    % ✅ colorbar: MATLAB 버전 호환
    colormap(fig, cmap);
    try
        cb = colorbar(tl, 'Location','eastoutside');
        cb.Label.String = capLabel;
        clim(tl, [capMin capMax]);
    catch
        cb = colorbar('eastoutside');
        cb.Label.String = capLabel;
        caxis([capMin capMax]);   % clim 대신
    end

    fig_base = sprintf('SOC%d_loadwise_3x2_subplot_color_%s', socVal, safeTag(capLabel));
    savefig(fig, fullfile(outDir, [fig_base '.fig']));
    exportgraphics(fig, fullfile(outDir, [fig_base '.png']), 'Resolution', 220);

    fprintf('→ SOC %d 플롯 저장: %s\n', socVal, fullfile(outDir, [fig_base '.png']));
end

disp('완료: SOC별 3×2 subplot(라인) + 컬러바( QC40 또는 Cell index ) 완료.');

%% ======================================================================
%  (ADD) 부하별 분포 Boxchart : SOC별로 3×2 subplot 저장
%   - boxplot 경고 제거용: boxchart 사용
%   - y축: 무조건 0부터
%% ======================================================================

outDir_box = fullfile(outDir,'boxchart_subplot');
if ~exist(outDir_box,'dir'), mkdir(outDir_box); end

xCat = categorical(loadNames, loadNames, 'Ordinal', true);  % 부하 순서 고정

for s = 1:nSOC
    socVal = socVals(s);

    figB = figure('Color','w', ...
        'Name', sprintf('SOC%d – boxchart by load', socVal), ...
        'Position',[90 90 1700 850]);

    tlB = tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

    for k = 1:6
        p = paramOrder(k);
        ax = nexttile(tlB); hold(ax,'on'); grid(ax,'on'); box(ax,'on');

        % ---- 각 load별로 boxchart ----
        Y_all = [];
        for l = 1:nLoads
            y = nan(numel(cells_for_plot),1);

            for ci = 1:numel(cells_for_plot)
                key = char(cells_for_plot(ci));
                TT  = param_all_cells.(key);   % [4×8×6]
                val = TT(s,l,p);
                y(ci) = val * scaleFactor(p);
            end

            y = y(isfinite(y));    % NaN 제거
            if isempty(y), continue; end

            boxchart(ax, repmat(xCat(l), numel(y), 1), y);
            Y_all = [Y_all; y]; %#ok<AGROW>
        end

        title(ax, paramTitles{k}, 'Interpreter','tex');
        xlabel(ax, 'Driving load');
        ax.XTickLabelRotation = 45;

        % ---- y축 0부터 ----
        ymax = max(Y_all, [], 'omitnan');
        if ~isfinite(ymax) || ymax<=0, ymax = 1; end
        ylim(ax, [0, ymax*1.05]);
    end

    title(tlB, sprintf('SOC %d – distribution across driving loads (boxchart)', socVal), ...
        'Interpreter','none');

    fig_base = sprintf('SOC%d_boxplot_by_load', socVal);
    savefig(figB, fullfile(outDir_box, [fig_base '.fig']));
    exportgraphics(figB, fullfile(outDir_box, [fig_base '.png']), 'Resolution', 220);

    fprintf('→ (ADD) SOC %d 3×2 boxchart 저장: %s\n', socVal, fullfile(outDir_box, [fig_base '.png']));
end

disp('완료: (ADD) SOC별 3×2 boxchart 생성 완료.');

%% ======================================================================
%  (ADD) tau1–tau2 plane: load-wise median points + (optional) 1σ ellipse
%   - SOC별로 1장씩 저장
%   - 각 부하: 점 = (median tau1, median tau2) across cells
%   - (옵션) 셀 분포 1σ 타원: (tau1,tau2) across cells
%   - 로그축(권장): useLog = true
%% ======================================================================

outDir_plane = fullfile(outDir,'tau1tau2_plane');
if ~exist(outDir_plane,'dir'), mkdir(outDir_plane); end

useLog      = true;   % ★ 권장: tau 스케일 차이 줄이기
doEllipse   = true;   % 1σ 타원 표시 여부
minNellipse = 3;      % 타원 계산 최소 샘플 수(셀 수)

for s = 1:nSOC
    socVal = socVals(s);

    % --- 부하별: 셀들의 tau1/tau2 수집 -> 중앙값 점 ---
    medTau1 = nan(nLoads,1);
    medTau2 = nan(nLoads,1);

    % ellipse용: load별 셀 샘플 저장
    tau1_cells = cell(nLoads,1);
    tau2_cells = cell(nLoads,1);

    for l = 1:nLoads
        v1 = nan(numel(cells_for_plot),1);
        v2 = nan(numel(cells_for_plot),1);

        for ci = 1:numel(cells_for_plot)
            key = char(cells_for_plot(ci));
            TT  = param_all_cells.(key);   % [4×8×6], pIdx: 4=tau1, 5=tau2

            v1(ci) = TT(s,l,4);  % tau1 [s]
            v2(ci) = TT(s,l,5);  % tau2 [s]
        end

        good = isfinite(v1) & isfinite(v2) & v1>0 & v2>0;
        v1 = v1(good); v2 = v2(good);

        tau1_cells{l} = v1;
        tau2_cells{l} = v2;

        if ~isempty(v1)
            medTau1(l) = median(v1,'omitnan');
            medTau2(l) = median(v2,'omitnan');
        end
    end

    % --- 플롯 ---
    figP = figure('Color','w', ...
        'Name', sprintf('SOC%d – tau1-tau2 plane (by load)', socVal), ...
        'Position',[120 120 980 720]);
    ax = axes(figP); hold(ax,'on'); grid(ax,'on'); box(ax,'on');

    % 축 변환(로그 권장)
    if useLog
        % log10 좌표로 그림 (0/음수 방지 이미 했음)
        Xp = log10(medTau1);
        Yp = log10(medTau2);
        xlabel(ax,'log_{10}(\tau_1 [s])');
        ylabel(ax,'log_{10}(\tau_2 [s])');
        title(ax, sprintf('SOC %d – load-wise median in log(\\tau) space', socVal), 'Interpreter','none');
    else
        Xp = medTau1;
        Yp = medTau2;
        xlabel(ax,'\tau_1 [s]');
        ylabel(ax,'\tau_2 [s]');
        title(ax, sprintf('SOC %d – load-wise median in (\\tau_1,\\tau_2)', socVal), 'Interpreter','none');
    end

    % --- 타원(옵션): load별 셀 분포 1σ ---
    if doEllipse
        for l = 1:nLoads
            v1 = tau1_cells{l};
            v2 = tau2_cells{l};
            if numel(v1) < minNellipse, continue; end

            if useLog
                x = log10(v1); y = log10(v2);
            else
                x = v1; y = v2;
            end

            good = isfinite(x) & isfinite(y);
            x = x(good); y = y(good);
            if numel(x) < minNellipse, continue; end

            % 1σ ellipse from covariance
            mu = [mean(x,'omitnan'), mean(y,'omitnan')];
            C  = cov([x y], 'omitrows');   % 2x2
            if any(~isfinite(C(:))) || rank(C) < 2
                continue;
            end

            % eigen-decomposition
            [V,D] = eig(C);
            d = diag(D);
            if any(d<=0), continue; end

            t = linspace(0,2*pi,200);
            circ = [cos(t); sin(t)];
            ell  = (V*diag(sqrt(d))*circ) + mu(:);

            plot(ax, ell(1,:), ell(2,:), '-', 'LineWidth',1.2); %#ok<*UNRCH>
        end
    end

    % --- 중앙값 점 + 라벨 ---
    scatter(ax, Xp, Yp, 80, 'filled');

    for l = 1:nLoads
        if ~isfinite(Xp(l)) || ~isfinite(Yp(l)), continue; end
        text(ax, Xp(l), Yp(l), ['  ' loadNames{l}], ...
            'Interpreter','none', 'FontSize',10, ...
            'VerticalAlignment','middle');
    end

    % 보기 좋게 axis padding
    xg = Xp(isfinite(Xp)); yg = Yp(isfinite(Yp));
    if ~isempty(xg) && ~isempty(yg)
        xr = range(xg); yr = range(yg);
        if xr==0, xr = 1; end
        if yr==0, yr = 1; end
        xlim(ax, [min(xg)-0.15*xr, max(xg)+0.25*xr]);
        ylim(ax, [min(yg)-0.15*yr, max(yg)+0.15*yr]);
    end

    fig_base = sprintf('SOC%d_tau1tau2_plane_median%s%s', ...
        socVal, ternary(useLog,'_log',''), ternary(doEllipse,'_ellipse1s',''));
    savefig(figP, fullfile(outDir_plane, [fig_base '.fig']));
    exportgraphics(figP, fullfile(outDir_plane, [fig_base '.png']), 'Resolution', 220);

    fprintf('→ (ADD) SOC %d tau1–tau2 plane 저장: %s\n', ...
        socVal, fullfile(outDir_plane, [fig_base '.png']));
end

disp('완료: (ADD) SOC별 tau1–tau2 평면(중앙값 점 + 옵션 1σ 타원) 생성 완료.');


%% ---- local helper (MATLAB R2016b+면 스크립트 끝에 함수 OK) ----
function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end


%% ===================== 로컬 함수 =====================
function tag = safeTag(s)
% 파일명에 쓰기 안전한 태그로 변환
    s = char(string(s));
    s = regexprep(s,'[^a-zA-Z0-9]+','_');
    s = regexprep(s,'_{2,}','_');
    tag = s;
end
