%% ======================================================================
%  2RC(US06: Tbl_US06_ECM) + 스칼라(온도별 DCIR/Power 포함) → 피처 구성 → 상관분석 (Pearson)
%  - 입력:
%       2RC_results.mat (Tbl_US06_ECM 포함)
%          * Tbl_US06_ECM.RowNames : 셀 이름
%          * Tbl_US06_ECM.VarNames: SOC70_R0_mOhm, SOC70_R1_mOhm, ..., SOC50_tau2
%       사용자 스칼라:
%          QC/2, QC/40, R1s, Rcharg, Rcharg_80_90_avg,
%          DCIR_1s_Txx, DCIR_delta_10s_1s_Txx (온도별), Power_Txx (온도별)
%  - 출력( save_path ):
%    features_allvars_full.csv / features_allvars_active.csv
%    corr_pearson.csv / pval_pearson.csv / corr_pearson.mat
%    corrmatrix_pair_kde.(fig|png), corr_heatmap_r_p_ci.(fig|png)
% ======================================================================
clear; clc; close all;

%% ── 설정(토글) ---------------------------------------------------------
SOC_use = [50 70];           % 분석에 포함할 SOC들 (예: [70], [70 50] 등)
DRAW_PAIRPLOT = true;     % Pairplot 그리기 여부

% 온도 리스트 (요청: 45 35 20 10 0도)
TEMP_list = [45 35 20 10 0];

% 스칼라 활성화 토글(기본)
USE_DELTA            = true;    % ΔDCIR(=DCIR_10s - DCIR_1s 또는 직접 입력)
USE_RAW_DCIR10S      = false;   % DCIR_10s (온도별도 지원, 기본 off)
USE_RAW_DCIR1S       = true;    % DCIR_1s (온도별)
USE_RCHARG           = true;   % Rcharg
USE_R1S              = true;    % R1s
USE_RCHARG_8090      = true;   % Rcharg_80_90_avg
USE_POWER            = true;    % Power (온도별)

% 온도별 토글(전체 온도에 동일 적용) — 필요하면 아래를 TEMP별로 쪼개도 됨
USE_DCIR1S_BYTEMP    = USE_RAW_DCIR1S;
USE_DDELTA_BYTEMP    = USE_DELTA;
USE_DCIR10S_BYTEMP   = USE_RAW_DCIR10S;
USE_POWER_BYTEMP     = USE_POWER;

%% ── 경로/파일 ----------------------------------------------------------
% 2RC_results.mat (US06: Tbl_US06_ECM 사용)
matPath   = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\20degC\2RC_fitting\2RC_results.mat";

% 결과 저장 경로
save_path = 'G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\Correlation_Anlaysis';
if ~exist(save_path, 'dir'), mkdir(save_path); end

%% ── Tbl_US06_ECM 로드 & 2RC 파라미터 구성 -----------------------------
S = load(matPath, 'Tbl_US06_ECM');
if ~isfield(S,'Tbl_US06_ECM')
    error('Tbl_US06_ECM 이 %s 에서 발견되지 않았습니다.', matPath);
end
Tbl_US06_ECM = S.Tbl_US06_ECM;

% 셀 이름
cell_names = Tbl_US06_ECM.Properties.RowNames;
if isstring(cell_names)
    cell_names = cellstr(cell_names);
elseif ischar(cell_names)
    cell_names = cellstr(cell_names);
end
nC = numel(cell_names);

% SOC 리스트를 VarNames 에서 자동 추출 (SOC70_..., SOC50_... 이런 형태)
vnames = Tbl_US06_ECM.Properties.VariableNames;
soc_list = [];
for i = 1:numel(vnames)
    tok = regexp(vnames{i}, '^SOC(\d+)_', 'tokens', 'once');
    if ~isempty(tok)
        soc_list(end+1) = str2double(tok{1}); %#ok<AGROW>
    end
end
SOC_2RC = unique(soc_list, 'stable');  % 예: [70 50]
if isempty(SOC_2RC)
    error('Tbl_US06_ECM VarNames 에 "SOCxx_..." 형식의 변수를 찾지 못했습니다.');
end

% 표준 파라미터 이름
pNames_2RC = {'R0','R1','R2','tau1','tau2'};
nSOC_all   = numel(SOC_2RC);

% P2RC 초기화: 각 필드 [nC × nSOC_all]
P2RC = struct();
for k = 1:numel(pNames_2RC)
    P2RC.(pNames_2RC{k}) = nan(nC, nSOC_all);
end

% Tbl_US06_ECM → P2RC 채우기
for sIdx = 1:nSOC_all
    soc = SOC_2RC(sIdx);
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
            warning('Tbl_US06_ECM 에 변수 %s 가 없습니다. (SOC=%d, param=%s)', varName, soc, pname);
        else
            P2RC.(pname)(:, sIdx) = Tbl_US06_ECM{:, varName};
        end
    end
end

%% ── SOC 선택/정합 ------------------------------------------------------
SOC_use = SOC_use(:).';
SOC_use = SOC_use(ismember(SOC_use, SOC_2RC));     % Tbl_US06_ECM에 존재하는 SOC만
if isempty(SOC_use)
    error('선택한 SOC(%s)가 Tbl_US06_ECM 데이터에 존재하지 않습니다. (SOC_2RC=%s)', ...
        mat2str(SOC_use), mat2str(SOC_2RC));
end
idxSOC_2RC = arrayfun(@(s)find(SOC_2RC==s,1), SOC_use);
nSOC = numel(SOC_use);

% SOC별 파라미터명 구성 (예: R0_70, R1_70, ..., tau2_70 ...)
soc_param_names_2RC = {};
for s = SOC_use
    soc_param_names_2RC = [soc_param_names_2RC, strcat(pNames_2RC, ['_' num2str(s)])]; %#ok<AGROW>
end

%% ── 사용자 스칼라 입력(셀 순서=Tbl_US06_ECM Row 순서) ------------------
fprintf('셀 순서(%d개):\n', nC);
disp(strjoin(cell_names, ', '));

% ======================================================================
% [여기부터 "입력 받아주는 구간"]
% - 반드시 cell_names 순서와 동일한 길이(nC)로 넣어야 함
% - 길이 안 맞으면 ensureLength가 자르거나 NaN을 뒤에 채움
% ======================================================================

% (기본 스칼라) ----------------------------------------------------------
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
58.4];

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
2.63];

R1s_user      = [1.16
1.04
1.54
1.49
1.27
1.35
1.04
1.16
2.84
1.15
1.13
1.10];

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
2.72];

% (온도별 DCIR / Power) --------------------------------------------------
% - DCIR_1s_Txx_user       : 온도 xx도의 DCIR 1초
% - DCIRdelta_Txx_user     : 온도 xx도의 ΔDCIR(10-1초) (직접 입력)
% - DCIR10s_Txx_user       : (옵션) 온도 xx도의 DCIR 10초 (없으면 NaN 유지)
% - Power_Txx_user         : 온도 xx도의 Power (단위는 네 데이터 기준 그대로 넣어도 됨)

% 45도
DCIR1s_T45_user     = [0.70 
0.69 
1.03 
0.85 
0.85 
0.80 
0.70 
0.80 
1.61 
0.63 
0.73 
0.73  ];

DCIR10s_T45_user    = nan(nC,1);

DCIRdelta_T45_user  = [0.27 
0.27 
0.31 
0.32 
0.29 
0.32 
0.28 
0.27 
0.55 
0.27 
0.27 
0.27 ];

Power_T45_user      = [3070.96 
3122.62 
2220.93 
2526.49 
2599.85 
2642.55 
3042.75 
2796.45 
1372.85 
3316.18 
2976.85 
2993.30  ];

% 35도
DCIR1s_T35_user     = [0.81 
0.77 
1.19 
0.99 
0.98 
0.90 
0.79 
0.88 
1.89 
0.72 
0.82 
0.81 ];

DCIR10s_T35_user    = nan(nC,1);

DCIRdelta_T35_user  = [0.30 
0.29 
0.35 
0.35 
0.32 
0.35 
0.29 
0.29 
0.58 
0.29 
0.29 
0.29 ];

Power_T35_user      = [2680.64 
2813.73 
1934.89 
2213.15 
2294.82 
2365.99 
2761.76 
2541.74 
1195.84 
2964.25 
2673.08 
2725.46 ];

% 20도 (일단 여기만 채워도 됨)
DCIR1s_T20_user     = [1.17 
1.07 
1.67 
1.27 
1.39 
1.23 
1.03 
1.17 
2.84 
0.78 
1.16 
1.12 ];  % 예시(네 기존값)

DCIR10s_T20_user    = nan(nC,1); % 없으면 NaN 유지

DCIRdelta_T20_user  = [0.29 
0.27 
0.40 
0.35 
0.31 
0.34 
0.28 
0.28 
0.82 
0.28 
0.28 
0.27 ];  % 예시(네 기존값)

Power_T20_user      = [1966.29 
2148.17 
1384.54 
1760.16 
1673.50 
1796.91 
2190.84 
1986.73 
770.17 
2731.95 
1996.96 
2075.03 ]; % <- 여기 Power 값 넣어줘

% 10도
DCIR1s_T10_user     = [1.72 
1.48 
2.34 
1.88 
2.01 
1.76 
1.48 
1.61 
4.17 
1.23 
1.65 
1.59 ];

DCIR10s_T10_user    = nan(nC,1);

DCIRdelta_T10_user  = [0.35 
0.31 
0.46 
0.40 
0.37 
0.40 
0.32 
0.32 
0.90 
0.33 
0.33 
0.32 ];

Power_T10_user      = [1384.03 
1603.71 
1020.97 
1242.42 
1199.89 
1306.61 
1596.86 
1492.22 
553.86 
1840.07 
1446.86 
1503.46 ];

% 0도
DCIR1s_T0_user      = [2.37 
2.28 
3.78 
2.93 
3.11 
3.46 
2.37 
2.45 
6.54 
2.20 
2.29 
2.71 ];

DCIR10s_T0_user     = nan(nC,1);

DCIRdelta_T0_user   = [0.39 
0.37 
0.54 
0.48 
0.44 
0.51 
0.36 
0.38 
1.03 
0.40 
0.37 
0.39 ];

Power_T0_user       = [1034.84 
1078.04 
656.79 
829.19 
798.23 
708.00 
1046.83 
1009.51 
367.54 
1101.57 
1075.59 
921.66 ];

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

%% ── DCIR 사용 가능 여부 체크 (온도별 보호 로직) -------------------------
%  - DCIR_10s도 없고 ΔDCIR도 다 NaN이면: ΔDCIR 토글을 강제로 끄는 대신,
%    해당 온도 피처가 활성화돼있더라도 NaN이라면 이후 row_valid에서 빠질 수 있음
%  - 네가 "온도별 DCIR이 존재"한다고 했으니, 여기서는 경고만 하고 토글 자동 변경은 하지 않음

% 온도별 데이터 컨테이너 구성(이름 기반 접근용)
DCIR1s_byT    = struct('T45',DCIR1s_T45_user,'T35',DCIR1s_T35_user,'T20',DCIR1s_T20_user,'T10',DCIR1s_T10_user,'T0',DCIR1s_T0_user);
DCIR10s_byT   = struct('T45',DCIR10s_T45_user,'T35',DCIR10s_T35_user,'T20',DCIR10s_T20_user,'T10',DCIR10s_T10_user,'T0',DCIR10s_T0_user);
DCIRd_byT     = struct('T45',DCIRdelta_T45_user,'T35',DCIRdelta_T35_user,'T20',DCIRdelta_T20_user,'T10',DCIRdelta_T10_user,'T0',DCIRdelta_T0_user);
Power_byT     = struct('T45',Power_T45_user,'T35',Power_T35_user,'T20',Power_T20_user,'T10',Power_T10_user,'T0',Power_T0_user);

%% ── 피처명 구성 --------------------------------------------------------
% 스칼라 풀(전체 CSV에 포함될 후보들)
scalar_pool_names = {'QC2','QC40','Rcharg','R1s','Rcharg_80_90_avg'};

% 온도별 후보명 추가
for t = TEMP_list
    scalar_pool_names{end+1} = sprintf('DCIR_1s_T%d', t);            %#ok<AGROW>
    scalar_pool_names{end+1} = sprintf('DCIR_10s_T%d', t);           %#ok<AGROW>
    scalar_pool_names{end+1} = sprintf('DCIR_delta_10s_1s_T%d', t);  %#ok<AGROW>
    scalar_pool_names{end+1} = sprintf('Power_T%d', t);              %#ok<AGROW>
end

% active: 순서는 여기서 조정
active_scalar_names = {'QC2','QC40'};
if USE_RCHARG,          active_scalar_names{end+1} = 'Rcharg';           end
if USE_R1S,             active_scalar_names{end+1} = 'R1s';              end
if USE_RCHARG_8090,     active_scalar_names{end+1} = 'Rcharg_80_90_avg'; end

% 온도별 active 추가
for t = TEMP_list
    if USE_DCIR1S_BYTEMP
        active_scalar_names{end+1} = sprintf('DCIR_1s_T%d', t); %#ok<AGROW>
    end
    if USE_DCIR10S_BYTEMP
        active_scalar_names{end+1} = sprintf('DCIR_10s_T%d', t); %#ok<AGROW>
    end
    if USE_DDELTA_BYTEMP
        active_scalar_names{end+1} = sprintf('DCIR_delta_10s_1s_T%d', t); %#ok<AGROW>
    end
    if USE_POWER_BYTEMP
        active_scalar_names{end+1} = sprintf('Power_T%d', t); %#ok<AGROW>
    end
end

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

    % 스칼라 Map (키: scalar_pool_names 전체를 채움)
    scalar_vals = containers.Map();

    % 기본 스칼라
    scalar_vals('QC2')               = QC2_user(i);
    scalar_vals('QC40')              = QC40_user(i);
    scalar_vals('Rcharg')            = Rcharg_user(i);
    scalar_vals('R1s')               = R1s_user(i);
    scalar_vals('Rcharg_80_90_avg')  = Rcharg_80_90_avg_user(i);

    % 온도별 스칼라(DCIR/Power)
    for t = TEMP_list
        keyT = sprintf('T%d', t);

        d1  = DCIR1s_byT.(keyT)(i);
        d10 = DCIR10s_byT.(keyT)(i);
        dd  = DCIRd_byT.(keyT)(i);
        pw  = Power_byT.(keyT)(i);

        % ΔDCIR: 우선순위
        %  1) DCIRdelta_Txx_user(i) 가 finite면 그 값 사용
        %  2) 아니면 DCIR10s_Txx_user, DCIR1s_Txx_user 둘 다 finite면 10s - 1s
        %  3) 아니면 NaN
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
title('US06(2RC) - Correlation heatmap (r, p, 95% CI; * if |r|>0.9)');
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

    % 기본 스칼라
    switch name
        case 'QC2',                 lab = {'Q_{C/2}'}; return;
        case 'QC40',                lab = {'Q_{C/40}'}; return;
        case 'R1s',                 lab = {'R_{1s}'}; return;
        case 'Rcharg',              lab = {'R_{charg}'}; return;
        case 'Rcharg_80_90_avg',    lab = {'R_{charg,80-90}','(avg)'}; return;
    end

    % 온도별 스칼라 (DCIR/Power)
    tokT = regexp(name,'_(T\d+)$','tokens','once');
    if isempty(tokT)
        tokT = regexp(name,'_T(\d+)$','tokens','once');
    end

    % DCIR_1s_T20 / DCIR_delta_10s_1s_T20 / Power_T20 같은 형식
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

    % SOC 파라미터: R0_70, tau2_50 등
    socLine = '';
    tokS = regexp(name,'_(\d+)$','tokens','once');
    if ~isempty(tokS)
        v = str2double(tokS{1});
        socLine = sprintf('(SOC %d%%)', v);
        name = regexprep(name,'_(\d+)$','');
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
