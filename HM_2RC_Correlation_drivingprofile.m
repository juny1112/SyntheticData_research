%% ======================================================================
%  2RC(선택 부하: Tbl_<LOAD>_ECM) + 스칼라(온도별 DCIR/Power 포함)
%    → 피처 구성 → 상관분석 (Pearson)
%
%  (NEW)
%   - LOAD_USE_STR에 "US06 CITY1" 처럼 입력하면 해당 주행부하들만 2RC 피처로 사용
%   - 피처명: <LOAD>_<param>_<SOC>  (예: CITY1_R0_70)
% ======================================================================
clear; clc; close all;

%% ── 설정(토글) ---------------------------------------------------------
SOC_use = [70];            % 분석에 포함할 SOC들 (예: [70], [70 50] 등)
DRAW_PAIRPLOT = true;

% ★★★ 여기만 바꾸면 됨: 사용할 주행부하 입력(공백/콤마/세미콜론 지원)
% 'US06','UDDS','HWFET','WLTP','CITY1','CITY2','HW1','HW2'
LOAD_USE_STR = "US06 CITY1";   % 예: "US06 CITY1", "US06,CITY1", "US06;CITY1;HW1"

% 온도 리스트 45 35 20 10 0도
TEMP_list = [20];

% 스칼라 활성화 토글(기본)
USE_DELTA            = false;
USE_RAW_DCIR10S      = true;
USE_RAW_DCIR1S       = false;
USE_RCHARG           = true;
USE_R1S              = true;
USE_RCHARG_8090      = false;
USE_POWER            = true;

USE_DCIR1S_BYTEMP    = USE_RAW_DCIR1S;
USE_DDELTA_BYTEMP    = USE_DELTA;
USE_DCIR10S_BYTEMP   = USE_RAW_DCIR10S;
USE_POWER_BYTEMP     = USE_POWER;

%% ── 경로/파일 ----------------------------------------------------------
matPath   = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\20degC\2RC_fitting_600s\2RC_results_600s.mat";

save_path = 'G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\Correlation_Anlaysis';
if ~exist(save_path, 'dir'), mkdir(save_path); end

%% ── 주행부하 파싱/검증(입력 문자열 기반) --------------------------------
loadNames_all = {'US06','UDDS','HWFET','WLTP','CITY1','CITY2','HW1','HW2'};

load_use = parseLoadList(LOAD_USE_STR);       % {"US06","CITY1",...}
load_use = upper(load_use);

% 표준 리스트도 upper로 만들어서 비교
all_upper = upper(string(loadNames_all));

% 존재하는 것만 남기기
ok = ismember(string(load_use), all_upper);
if any(~ok)
    warning('알 수 없는 주행부하 입력이 있어 제외합니다: %s', strjoin(load_use(~ok), ', '));
end
load_use = load_use(ok);

% 표준 표기(원래 대소문자)로 되돌리기
load_use_std = cell(1,numel(load_use));
for i = 1:numel(load_use)
    idx = find(all_upper == string(load_use(i)), 1, 'first');
    load_use_std{i} = loadNames_all{idx};
end
load_use = load_use_std;

if isempty(load_use)
    error('선택된 주행부하가 없습니다. LOAD_USE_STR="%s" 를 확인하세요.', string(LOAD_USE_STR));
end

fprintf('>> 선택된 주행부하: %s\n', strjoin(load_use, ', '));

%% ── 2RC 테이블 로드(부하별) --------------------------------------------
S = load(matPath);

getTblECM = @(loadName) localGetTblECM(S, loadName);

% 선택 부하 중 실제 존재하는 것만(테이블 없는 경우 제외)
load_use_ok = {};
for i = 1:numel(load_use)
    try
        Ttmp = getTblECM(load_use{i});
        if istable(Ttmp)
            load_use_ok{end+1} = load_use{i}; %#ok<AGROW>
        end
    catch
        warning('mat 파일에 %s ECM 테이블이 없어 제외합니다.', load_use{i});
    end
end
load_use = load_use_ok;

if isempty(load_use)
    error('선택한 부하들에 대해 ECM 테이블을 찾지 못했습니다. matPath / 저장 변수를 확인하세요.');
end

%% ── 셀 이름 정합(부하들 간 공통 셀만) -----------------------------------
cell_sets = cell(numel(load_use),1);
for i = 1:numel(load_use)
    T = getTblECM(load_use{i});
    cell_sets{i} = T.Properties.RowNames;
end

cell_names = cell_sets{1};
for i = 2:numel(cell_sets)
    cell_names = intersect(cell_names, cell_sets{i}, 'stable');
end
if isempty(cell_names)
    error('선택한 부하 테이블들 간 RowNames(셀 이름) 교집합이 비었습니다.');
end
nC = numel(cell_names);

fprintf('>> 공통 셀 개수: %d\n', nC);

%% ── 2RC 피처 구성: (부하 선택 + SOC_use) -------------------------------
pNames_2RC = {'R0','R1','R2','tau1','tau2'};

feat2RC_names = {};
for li = 1:numel(load_use)
    L = load_use{li};
    for s = SOC_use(:).'
        for pi = 1:numel(pNames_2RC)
            feat2RC_names{end+1} = sprintf('%s_%s_%d', L, pNames_2RC{pi}, s); %#ok<AGROW>
        end
    end
end

X2RC = nan(nC, numel(feat2RC_names));

col = 0;
for li = 1:numel(load_use)
    L = load_use{li};
    TblL = getTblECM(L);
    TblL = TblL(cell_names, :);   % 공통 셀 순서로 정렬
    vnames = TblL.Properties.VariableNames;

    % TblL에 어떤 SOC가 있는지 추출
    soc_list = [];
    for ii = 1:numel(vnames)
        tok = regexp(vnames{ii}, '^SOC(\d+)_', 'tokens', 'once');
        if ~isempty(tok)
            soc_list(end+1) = str2double(tok{1}); %#ok<AGROW>
        end
    end
    SOC_inTbl = unique(soc_list, 'stable');

    for s = SOC_use(:).'
        if ~ismember(s, SOC_inTbl)
            warning('[%s] Tbl_%s_ECM 에 SOC%d 변수가 없어 NaN으로 남습니다.', L, L, s);
        end

        for pi = 1:numel(pNames_2RC)
            col = col + 1;
            pname = pNames_2RC{pi};

            if pi <= 3
                varName = sprintf('SOC%d_%s_mOhm', s, pname);
            else
                varName = sprintf('SOC%d_%s', s, pname);
            end

            if ismember(varName, vnames)
                X2RC(:, col) = TblL{:, varName};
            else
                X2RC(:, col) = NaN;
            end
        end
    end
end

%% ── 사용자 스칼라 입력(네 기존 코드 그대로) -----------------------------
fprintf('셀 순서(%d개):\n', nC);
disp(strjoin(cell_names, ', '));

% ======================================================================
% [여기부터 "입력 받아주는 구간"]  (네가 준 값 그대로)
% ======================================================================
QC2_user      = [56.14;55.93;52.55;50.52;52.13;48.53;56.34;55.15;39.89;55.63;55.57;56.86];
QC40_user     = [57.49;57.57;54;52.22;53.45;51.28;57.91;56.51;42.14;57.27;57.18;58.4];
Rcharg_user   = [2.17
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
2.09];
R1s_user      = [1.16;1.04;1.54;1.49;1.27;1.35;1.04;1.16;2.84;1.15;1.13;1.10];
Rcharg_80_90_avg_user = [2.75;2.41;4.43;3.56;3.51;4.25;2.64;2.38;8.36;2.52;2.49;2.72];

DCIR1s_T45_user     = [0.70;0.69;1.03;0.85;0.85;0.80;0.70;0.80;1.61;0.63;0.73;0.73];
DCIR10s_T45_user    = nan(numel(QC2_user),1);
DCIRdelta_T45_user  = [0.27;0.27;0.31;0.32;0.29;0.32;0.28;0.27;0.55;0.27;0.27;0.27];
Power_T45_user      = [3070.96;3122.62;2220.93;2526.49;2599.85;2642.55;3042.75;2796.45;1372.85;3316.18;2976.85;2993.30];

DCIR1s_T35_user     = [0.81;0.77;1.19;0.99;0.98;0.90;0.79;0.88;1.89;0.72;0.82;0.81];
DCIR10s_T35_user    = nan(numel(QC2_user),1);
DCIRdelta_T35_user  = [0.30;0.29;0.35;0.35;0.32;0.35;0.29;0.29;0.58;0.29;0.29;0.29];
Power_T35_user      = [2680.64;2813.73;1934.89;2213.15;2294.82;2365.99;2761.76;2541.74;1195.84;2964.25;2673.08;2725.46];

DCIR1s_T20_user     = [1.17;1.07;1.67;1.27;1.39;1.23;1.03;1.17;2.84;0.78;1.16;1.12];
DCIR10s_T20_user    = [1.60 
1.41 
2.33 
1.88 
2.06 
1.93 
1.35 
1.53 
3.55 
0.82 
1.52 
1.47];
DCIRdelta_T20_user  = [0.29;0.27;0.40;0.35;0.31;0.34;0.28;0.28;0.82;0.28;0.28;0.27];
Power_T20_user      = [2089.79 
2372.03 
1427.37 
1735.16 
1603.14 
1677.27 
2476.97 
2191.48 
914.67 
4067.20 
2196.09 
2278.82];

DCIR1s_T10_user     = [1.72;1.48;2.34;1.88;2.01;1.76;1.48;1.61;4.17;1.23;1.65;1.59];
DCIR10s_T10_user    = nan(numel(QC2_user),1);
DCIRdelta_T10_user  = [0.35;0.31;0.46;0.40;0.37;0.40;0.32;0.32;0.90;0.33;0.33;0.32];
Power_T10_user      = [1384.03;1603.71;1020.97;1242.42;1199.89;1306.61;1596.86;1492.22;553.86;1840.07;1446.86;1503.46];

DCIR1s_T0_user      = [2.37;2.28;3.78;2.93;3.11;3.46;2.37;2.45;6.54;2.20;2.29;2.71];
DCIR10s_T0_user     = nan(numel(QC2_user),1);
DCIRdelta_T0_user   = [0.39;0.37;0.54;0.48;0.44;0.51;0.36;0.38;1.03;0.40;0.37;0.39];
Power_T0_user       = [1034.84;1078.04;656.79;829.19;798.23;708.00;1046.83;1009.51;367.54;1101.57;1075.59;921.66];
% ======================================================================
% [입력 구간 끝]
% ======================================================================

% 길이 보정(기본 스칼라)
QC2_user               = ensureLength(QC2_user,               nC);
QC40_user              = ensureLength(QC40_user,              nC);
Rcharg_user            = ensureLength(Rcharg_user,            nC);
R1s_user               = ensureLength(R1s_user,               nC);
Rcharg_80_90_avg_user  = ensureLength(Rcharg_80_90_avg_user,  nC);

% 길이 보정(온도별)
DCIR1s_T45_user    = ensureLength(DCIR1s_T45_user,    nC);
DCIR10s_T45_user   = ensureLength(DCIR10s_T45_user,   nC);
DCIRdelta_T45_user = ensureLength(DCIRdelta_T45_user, nC);
Power_T45_user     = ensureLength(Power_T45_user,     nC);

DCIR1s_T35_user    = ensureLength(DCIR1s_T35_user,    nC);
DCIR10s_T35_user   = ensureLength(DCIR10s_T35_user,   nC);
DCIRdelta_T35_user = ensureLength(DCIRdelta_T35_user, nC);
Power_T35_user     = ensureLength(Power_T35_user,     nC);

DCIR1s_T20_user    = ensureLength(DCIR1s_T20_user,    nC);
DCIR10s_T20_user   = ensureLength(DCIR10s_T20_user,   nC);
DCIRdelta_T20_user = ensureLength(DCIRdelta_T20_user, nC);
Power_T20_user     = ensureLength(Power_T20_user,     nC);

DCIR1s_T10_user    = ensureLength(DCIR1s_T10_user,    nC);
DCIR10s_T10_user   = ensureLength(DCIR10s_T10_user,   nC);
DCIRdelta_T10_user = ensureLength(DCIRdelta_T10_user, nC);
Power_T10_user     = ensureLength(Power_T10_user,     nC);

DCIR1s_T0_user     = ensureLength(DCIR1s_T0_user,     nC);
DCIR10s_T0_user    = ensureLength(DCIR10s_T0_user,    nC);
DCIRdelta_T0_user  = ensureLength(DCIRdelta_T0_user,  nC);
Power_T0_user      = ensureLength(Power_T0_user,      nC);

%% ── 온도별 데이터 컨테이너 구성 ----------------------------------------
DCIR1s_byT    = struct('T45',DCIR1s_T45_user,'T35',DCIR1s_T35_user,'T20',DCIR1s_T20_user,'T10',DCIR1s_T10_user,'T0',DCIR1s_T0_user);
DCIR10s_byT   = struct('T45',DCIR10s_T45_user,'T35',DCIR10s_T35_user,'T20',DCIR10s_T20_user,'T10',DCIR10s_T10_user,'T0',DCIR10s_T0_user);
DCIRd_byT     = struct('T45',DCIRdelta_T45_user,'T35',DCIRdelta_T35_user,'T20',DCIRdelta_T20_user,'T10',DCIRdelta_T10_user,'T0',DCIRdelta_T0_user);
Power_byT     = struct('T45',Power_T45_user,'T35',Power_T35_user,'T20',Power_T20_user,'T10',Power_T10_user,'T0',Power_T0_user);

%% ── 피처명 구성 --------------------------------------------------------
scalar_pool_names = {'QC2','QC40','Rcharg','R1s','Rcharg_80_90_avg'};
for t = TEMP_list
    scalar_pool_names{end+1} = sprintf('DCIR_1s_T%d', t);            %#ok<AGROW>
    scalar_pool_names{end+1} = sprintf('DCIR_10s_T%d', t);           %#ok<AGROW>
    scalar_pool_names{end+1} = sprintf('DCIR_delta_10s_1s_T%d', t);  %#ok<AGROW>
    scalar_pool_names{end+1} = sprintf('Power_T%d', t);              %#ok<AGROW>
end

active_scalar_names = {'QC2','QC40'};

% ===== 여기 순서만 바꿈: Rcharg -> Rcharg_80_90_avg -> R1s =====
if USE_RCHARG,          active_scalar_names{end+1} = 'Rcharg';           end
if USE_RCHARG_8090,     active_scalar_names{end+1} = 'Rcharg_80_90_avg'; end
if USE_R1S,             active_scalar_names{end+1} = 'R1s';              end
% ===============================================================
for t = TEMP_list
    if USE_DCIR1S_BYTEMP,  active_scalar_names{end+1} = sprintf('DCIR_1s_T%d', t); end %#ok<AGROW>
    if USE_DCIR10S_BYTEMP, active_scalar_names{end+1} = sprintf('DCIR_10s_T%d', t); end %#ok<AGROW>
    if USE_DDELTA_BYTEMP,  active_scalar_names{end+1} = sprintf('DCIR_delta_10s_1s_T%d', t); end %#ok<AGROW>
    if USE_POWER_BYTEMP,   active_scalar_names{end+1} = sprintf('Power_T%d', t); end %#ok<AGROW>
end

feat_full_names   = [feat2RC_names, scalar_pool_names];
feat_active_names = [feat2RC_names, active_scalar_names];

%% ── 피처 행렬 생성 -----------------------------------------------------
feat_full   = nan(nC, numel(feat_full_names));
feat_active = nan(nC, numel(feat_active_names));

for i = 1:nC
    row_2rc = X2RC(i,:);

    scalar_vals = containers.Map();
    scalar_vals('QC2')               = QC2_user(i);
    scalar_vals('QC40')              = QC40_user(i);
    scalar_vals('Rcharg')            = Rcharg_user(i);
    scalar_vals('R1s')               = R1s_user(i);
    scalar_vals('Rcharg_80_90_avg')  = Rcharg_80_90_avg_user(i);

    for t = TEMP_list
        keyT = sprintf('T%d', t);
        d1  = DCIR1s_byT.(keyT)(i);
        d10 = DCIR10s_byT.(keyT)(i);
        dd  = DCIRd_byT.(keyT)(i);
        pw  = Power_byT.(keyT)(i);

        if isfinite(dd)
            dDCIR = dd;
        elseif isfinite(d10) && isfinite(d1)
            dDCIR = d10 - d1;
        else
            dDCIR = NaN;
        end

        scalar_vals(sprintf('DCIR_1s_T%d', t))           = d1;
        scalar_vals(sprintf('DCIR_10s_T%d', t))          = d10;
        scalar_vals(sprintf('DCIR_delta_10s_1s_T%d', t)) = dDCIR;
        scalar_vals(sprintf('Power_T%d', t))             = pw;
    end

    feat_full(i,:)   = [row_2rc, cellfun(@(nm) scalar_vals(nm), scalar_pool_names)];
    feat_active(i,:) = [row_2rc, cellfun(@(nm) scalar_vals(nm), active_scalar_names)];
end

%% ── 유효 행 필터(활성 기준) --------------------------------------------
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

%% ── 상관분석/그림 (Pearson) --------------------------------------------
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

    try
        sgtitle(sprintf('Correlation Matrix (Pearson, pairwise) | Loads: %s', strjoin(load_use, ', ')), ...
                'Interpreter','none');
    catch
    end

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

        % txt = sprintf('r=%.2f\np=%.3g\nCI[%.2f, %.2f]', r, pval, lo, hi);
        txt = sprintf('r=%.2f\np=%.3g', r, pval);
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

title(sprintf('Loads: %s - Correlation heatmap (r, p, 95%% CI; * if |r|>0.9)', strjoin(load_use, ', ')), ...
      'Interpreter','none');
savefig(f2, outfig2);
exportgraphics(f2, outpng2, 'Resolution', 220);

%% ── 상관 행렬 저장 -----------------------------------------------------
[R_pearson, P_pearson] = deal(Rmat, Pmat);
Tcorr = array2table(R_pearson, 'VariableNames', vnames_active, 'RowNames', vnames_active);
Tpval = array2table(P_pearson,  'VariableNames', vnames_active, 'RowNames', vnames_active);

writetable(Tcorr, fullfile(save_path, 'corr_pearson.csv'), 'WriteRowNames', true);
writetable(Tpval,  fullfile(save_path, 'pval_pearson.csv'), 'WriteRowNames', true);
save(fullfile(save_path, 'corr_pearson.mat'), 'R_pearson', 'P_pearson', 'vnames_active', 'cells_active_valid', 'SOC_use', 'LOAD_USE_STR');

disp('corr_pearson.csv / pval_pearson.csv / corr_pearson.mat 저장 완료');

%% ========================= 보조 함수들 =================================
function loads = parseLoadList(str0)
    % "US06 CITY1", "US06,CITY1", "US06; CITY1" 모두 허용
    if isstring(str0), str0 = char(str0); end
    str0 = strtrim(str0);
    if isempty(str0)
        loads = {};
        return
    end
    parts = regexp(str0, '[,;\s]+', 'split');
    parts = parts(~cellfun(@isempty,parts));
    loads = parts(:).';
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
    % 1) Tbl_Load_ECM.(loadName)
    if isfield(S,'Tbl_Load_ECM') && isstruct(S.Tbl_Load_ECM) && isfield(S.Tbl_Load_ECM, loadName)
        T = S.Tbl_Load_ECM.(loadName);
        return
    end
    % 2) Tbl_<LOAD>_ECM
    varName = sprintf('Tbl_%s_ECM', loadName);
    if isfield(S, varName)
        T = S.(varName);
        return
    end
    error('ECM table not found for load=%s', loadName);
end

function lab = prettyVarLabel(name0)
    name = char(name0);

    % 기본 스칼라
    switch name
        case 'QC2',                 lab = {'Q_{C/2}'}; return;
        case 'QC40',                lab = {'Q_{C/40}'}; return;
        case 'R1s',                 lab = {'R_{1s}'}; return;
        case 'Rcharg',              lab = {'R_{charg}'}; return;
        case 'Rcharg_80_90_avg',    lab = {'R_{charg,80-90}','(avg)'}; return;
    end

    % 온도별 스칼라 (DCIR/Power)
    tok = regexp(name,'^(DCIR_1s|DCIR_10s|DCIR_delta_10s_1s|Power)_T(\d+)$','tokens','once');
    if ~isempty(tok)
        baseKey = tok{1};
        Tval    = str2double(tok{2});
        switch baseKey
            case 'DCIR_1s',              base = 'DCIR (1 s)';
            case 'DCIR_10s',             base = 'DCIR (10 s)';
            case 'DCIR_delta_10s_1s',    base = '\DeltaDCIR (10-1 s)';
            case 'Power',                base = 'Power';
            otherwise,                   base = strrep(name0,'_','\_');
        end
        lab = {base; sprintf('(T %d^{\\circ}C)', Tval)};
        return
    end

    % (NEW) 주행부하 + SOC 파라미터: "US06_R0_70", "CITY1_tau2_50" 등
    % 형식: <LOAD>_<param>_<SOC>
    tok2 = regexp(name,'^([A-Za-z0-9]+)_(R0|R1|R2|tau1|tau2)_(\d+)$','tokens','once');
    if ~isempty(tok2)
        loadName = tok2{1};
        pName    = tok2{2};
        socVal   = str2double(tok2{3});

        switch string(pName)
            case "R0",   base = 'R_{0}';
            case "R1",   base = 'R_{1}';
            case "R2",   base = 'R_{2}';
            case "tau1", base = '\tau_{1}';
            case "tau2", base = '\tau_{2}';
            otherwise,   base = strrep(pName,'_','\_');
        end

        lab = {sprintf('%s (%s)', base, loadName); sprintf('(SOC %d%%)', socVal)};
        return
    end

    % fallback
    lab = {strrep(name0,'_','\_')};
end
