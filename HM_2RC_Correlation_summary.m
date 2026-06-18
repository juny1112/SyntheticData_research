%% ======================================================================
%  2RC(선택 부하: Tbl_<LOAD>_ECM, 주행부하 온도별) + 스칼라(온도별 DCIR/Power 포함)
%    → 피처 구성 → Label vs ECM correlation heatmap (Pearson)
%
%  [목표]
%   - x축: Features (ECM parameters)
%   - y축: Labels (SOH, SOH-x; scalar labels 자동 선택)
%   - 전체 상관행렬이 아니라 "보여주고 싶은 상관성만" 직사각 heatmap으로 표시
%
%  [핵심]
%   - x축은 feat2RC_names 전체
%   - y축은 active_scalar_names 전체 (온도/토글 기반 자동 생성)
%   - LABEL_NAMES 같은 수동 선택 없음
%
%  [기능]
%   - LOAD_USE_STR에 "US06 CITY1" 처럼 입력하면 해당 주행부하들만 2RC 피처로 사용
%   - 주행부하 2RC용 온도와 스칼라용 온도를 분리
%   - RowNames 완전일치 대신 "셀 ID" 기준 매칭
%   - 셀 ID 추출 규칙:
%       1) 문자열 맨 앞 x01 / X01 우선
%       2) 없으면 문자열에서 처음 나오는 숫자 사용
%   - heatmap에서 p값 표시 여부 토글 가능
% ======================================================================
clear; clc; close all;

%% ── 설정(토글) ---------------------------------------------------------
SOC_use = [70 50];
SHOW_P_IN_HEATMAP = true;   % true: r,p 둘 다 표시 / false: r만 표시

% 사용할 주행부하 입력(공백/콤마/세미콜론 지원)
% 'US06','UDDS','HWFET','WLTP','CITY1','CITY2','HW1','HW2'
LOAD_USE_STR = "US06";

% 온도 리스트 분리
TEMP_list_load   = [20];          % 주행부하 2RC용 20 10 0
TEMP_list_scalar = [20];    % DCIR / Power / 기타 스칼라용 45 35 20 10 0

% 스칼라 활성화 토글
USE_DELTA            = false;
USE_RAW_DCIR10S      = true;
USE_RAW_DCIR1S       = false;
USE_RCHARG           = true;
USE_R1S              = false;
USE_RCHARG_8090      = false;
USE_POWER            = false;

USE_DCIR1S_BYTEMP    = USE_RAW_DCIR1S;
USE_DDELTA_BYTEMP    = USE_DELTA;
USE_DCIR10S_BYTEMP   = USE_RAW_DCIR10S;
USE_POWER_BYTEMP     = USE_POWER;

% 추가 시각화 옵션
MARK_STRONG_CORR = true;
STRONG_CORR_TH   = 0.95;   % 별표 기준
SHOW_ONLY_ABS_R_OVER = []; % 예: 0.7 로 두면 |r|>=0.7인 feature만 표시, []면 전체 표시

%% ── 경로/파일 ----------------------------------------------------------
baseDir_2RC = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed";

save_path = 'G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\Correlation_Anlaysis';
if ~exist(save_path, 'dir'), mkdir(save_path); end

%% ── 주행부하 파싱/검증(입력 문자열 기반) --------------------------------
loadNames_all = {'US06','UDDS','HWFET','WLTP','CITY1','CITY2','HW1','HW2'};

load_use = parseLoadList(LOAD_USE_STR);
load_use = upper(load_use);

all_upper = upper(string(loadNames_all));

ok = ismember(string(load_use), all_upper);
if any(~ok)
    warning('알 수 없는 주행부하 입력이 있어 제외합니다: %s', strjoin(load_use(~ok), ', '));
end
load_use = load_use(ok);

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

%% ── 2RC 테이블 로드(주행부하 온도별 + 부하별) ---------------------------
S_byT = struct();
temp_load_ok = [];

for t = TEMP_list_load
    matPath_t = fullfile(baseDir_2RC, sprintf('%ddegC', t), '2RC_fitting_600s', '2RC_results_600s.mat');

    if isfile(matPath_t)
        S_byT.(sprintf('T%d', t)) = load(matPath_t);
        temp_load_ok(end+1) = t; %#ok<AGROW>
        fprintf('>> 2RC mat loaded: T=%d°C\n', t);
    else
        warning('2RC mat 파일이 없습니다: %s', matPath_t);
    end
end

TEMP_list_load = temp_load_ok;

if isempty(TEMP_list_load)
    error('읽을 수 있는 2RC_results_600s.mat 파일이 없습니다.');
end

getTblECM_byT = @(t, loadName) localGetTblECM(S_byT.(sprintf('T%d', t)), loadName);

% 선택 부하 중 적어도 하나의 주행부하 온도에서 존재하는 것만 남김
load_use_ok = {};
for i = 1:numel(load_use)
    found_any = false;
    for t = TEMP_list_load
        try
            Ttmp = getTblECM_byT(t, load_use{i});
            if istable(Ttmp)
                found_any = true;
                break;
            end
        catch
        end
    end
    if found_any
        load_use_ok{end+1} = load_use{i}; %#ok<AGROW>
    else
        warning('모든 주행부하 온도에서 %s ECM 테이블을 찾지 못해 제외합니다.', load_use{i});
    end
end
load_use = load_use_ok;

if isempty(load_use)
    error('선택한 부하들에 대해 어느 주행부하 온도에서도 ECM 테이블을 찾지 못했습니다.');
end

%% ── 셀 ID 정합(주행부하 온도 × 부하 공통 셀만) -------------------------
cell_id_sets = {};
cell_name_sets = {};

for t = TEMP_list_load
    for i = 1:numel(load_use)
        try
            T = getTblECM_byT(t, load_use{i});
            raw_names = T.Properties.RowNames;

            ids_raw = extractCellIDs(raw_names);
            [ids_unique, idx_keep] = uniqueCellIDsFirst(ids_raw);
            raw_names_unique = raw_names(idx_keep);

            cell_id_sets{end+1}   = ids_unique;       %#ok<AGROW>
            cell_name_sets{end+1} = raw_names_unique; %#ok<AGROW>

            fprintf('>> T=%d, %s : %d rows, %d unique IDs\n', ...
                t, load_use{i}, numel(raw_names), numel(ids_unique));
        catch
            warning('T=%d°C, load=%s 테이블 없음', t, load_use{i});
        end
    end
end

if isempty(cell_id_sets)
    error('셀 ID를 얻을 수 있는 ECM 테이블이 없습니다.');
end

cell_ids_common = cell_id_sets{1};
for i = 2:numel(cell_id_sets)
    cell_ids_common = intersect(cell_ids_common, cell_id_sets{i}, 'stable');
end

if isempty(cell_ids_common)
    fprintf('\n=== 디버그: 온도/부하별 추출된 cell IDs ===\n');
    for i = 1:numel(cell_id_sets)
        fprintf('set %d: ', i);
        disp(cell_id_sets{i}(:).');
    end
    error('선택한 주행부하 온도/부하 테이블들 간 공통 cell ID가 비었습니다.');
end

cell_ids = cell_ids_common(:);
cell_names = arrayfun(@(x) sprintf('ID%02d', x), cell_ids, 'UniformOutput', false);

nC = numel(cell_ids);
fprintf('>> 공통 셀 개수: %d\n', nC);
fprintf('>> 공통 셀 ID: ');
disp(cell_ids(:).');

%% ── 2RC 피처 구성: (주행부하 온도 + 부하 + SOC_use) --------------------
pNames_2RC = {'R0','R1','R2','tau1','tau2'};

feat2RC_names = {};
for t = TEMP_list_load
    for li = 1:numel(load_use)
        L = load_use{li};
        for s = SOC_use(:).'
            for pi = 1:numel(pNames_2RC)
                feat2RC_names{end+1} = sprintf('T%d_%s_%s_%d', t, L, pNames_2RC{pi}, s); %#ok<AGROW>
            end
        end
    end
end

X2RC = nan(nC, numel(feat2RC_names));

col = 0;
for t = TEMP_list_load
    for li = 1:numel(load_use)
        L = load_use{li};

        try
            TblL = getTblECM_byT(t, L);

            raw_names_tbl = TblL.Properties.RowNames;
            ids_tbl_raw   = extractCellIDs(raw_names_tbl);
            [ids_tbl, idx_keep_tbl] = uniqueCellIDsFirst(ids_tbl_raw);

            TblL = TblL(idx_keep_tbl, :);
            vnames = TblL.Properties.VariableNames;

            [tf_map, loc_map] = ismember(cell_ids, ids_tbl);

            if ~all(tf_map)
                warning('[T%d, %s] 일부 cell ID가 현재 테이블에 없습니다.', t, L);
            end

            soc_list = [];
            for ii = 1:numel(vnames)
                tok = regexp(vnames{ii}, '^SOC(\d+)_', 'tokens', 'once');
                if ~isempty(tok)
                    soc_list(end+1) = str2double(tok{1}); %#ok<AGROW>
                end
            end
            SOC_inTbl = unique(soc_list, 'stable');

        catch
            TblL = [];
            vnames = {};
            SOC_inTbl = [];
            tf_map = false(nC,1);
            loc_map = nan(nC,1);
            warning('[T%d, %s] ECM 테이블이 없어 해당 피처는 NaN으로 둡니다.', t, L);
        end

        for s = SOC_use(:).'
            if ~ismember(s, SOC_inTbl)
                warning('[T%d, %s] SOC%d 변수가 없어 NaN으로 남습니다.', t, L, s);
            end

            for pi = 1:numel(pNames_2RC)
                col = col + 1;
                pname = pNames_2RC{pi};

                if pi <= 3
                    varName = sprintf('SOC%d_%s_mOhm', s, pname);
                else
                    varName = sprintf('SOC%d_%s', s, pname);
                end

                tmpcol = nan(nC,1);

                if ~isempty(TblL) && ismember(varName, vnames)
                    idx_ok = find(tf_map);
                    tmpcol(idx_ok) = TblL{loc_map(idx_ok), varName};
                end

                X2RC(:, col) = tmpcol;
            end
        end
    end
end

%% ── 사용자 스칼라 입력 -------------------------------------------------
fprintf('셀 순서(%d개):\n', nC);
disp(strjoin(cell_names, ', '));

% ======================================================================
% [여기부터 입력 구간]
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
DCIR10s_T45_user    = [0.90
0.86
1.22
1.21
1.03
1.11
0.89
0.97
2.10
0.95
0.91
0.91];
DCIRdelta_T45_user  = [0.27;0.27;0.31;0.32;0.29;0.32;0.28;0.27;0.55;0.27;0.27;0.27];
Power_T45_user      = [3070.96;3122.62;2220.93;2526.49;2599.85;2642.55;3042.75;2796.45;1372.85;3316.18;2976.85;2993.30];

DCIR1s_T35_user     = [0.81;0.77;1.19;0.99;0.98;0.90;0.79;0.88;1.89;0.72;0.82;0.81];
DCIR10s_T35_user    = [1.04
0.98
1.41
1.39
1.18
1.27
1.00
1.08
2.48
1.08
1.03
1.01];
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
DCIR10s_T10_user    = [2.03
1.75
2.58
2.55
2.26
2.28
1.78
1.89
4.69
1.94
1.94
1.91];
DCIRdelta_T10_user  = [0.35;0.31;0.46;0.40;0.37;0.40;0.32;0.32;0.90;0.33;0.33;0.32];
Power_T10_user      = [1384.03;1603.71;1020.97;1242.42;1199.89;1306.61;1596.86;1492.22;553.86;1840.07;1446.86;1503.46];

DCIR1s_T0_user      = [2.37;2.28;3.78;2.93;3.11;3.46;2.37;2.45;6.54;2.20;2.29;2.71];
DCIR10s_T0_user     = [2.63
2.48
3.76
3.50
3.19
3.34
2.51
2.66
6.60
2.83
2.50
2.96];
DCIRdelta_T0_user   = [0.39;0.37;0.54;0.48;0.44;0.51;0.36;0.38;1.03;0.40;0.37;0.39];
Power_T0_user       = [1034.84;1078.04;656.79;829.19;798.23;708.00;1046.83;1009.51;367.54;1101.57;1075.59;921.66];
% ======================================================================
% [입력 구간 끝]
% ======================================================================

%% ── 길이 보정 ---------------------------------------------------------
QC2_user               = ensureLength(QC2_user,               nC);
QC40_user              = ensureLength(QC40_user,              nC);
Rcharg_user            = ensureLength(Rcharg_user,            nC);
R1s_user               = ensureLength(R1s_user,               nC);
Rcharg_80_90_avg_user  = ensureLength(Rcharg_80_90_avg_user,  nC);

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
DCIR1s_byT  = struct();
DCIR10s_byT = struct();
DCIRd_byT   = struct();
Power_byT   = struct();

DCIR1s_byT.T45  = DCIR1s_T45_user;
DCIR1s_byT.T35  = DCIR1s_T35_user;
DCIR1s_byT.T20  = DCIR1s_T20_user;
DCIR1s_byT.T10  = DCIR1s_T10_user;
DCIR1s_byT.T0   = DCIR1s_T0_user;

DCIR10s_byT.T45 = DCIR10s_T45_user;
DCIR10s_byT.T35 = DCIR10s_T35_user;
DCIR10s_byT.T20 = DCIR10s_T20_user;
DCIR10s_byT.T10 = DCIR10s_T10_user;
DCIR10s_byT.T0  = DCIR10s_T0_user;

DCIRd_byT.T45   = DCIRdelta_T45_user;
DCIRd_byT.T35   = DCIRdelta_T35_user;
DCIRd_byT.T20   = DCIRdelta_T20_user;
DCIRd_byT.T10   = DCIRdelta_T10_user;
DCIRd_byT.T0    = DCIRdelta_T0_user;

Power_byT.T45   = Power_T45_user;
Power_byT.T35   = Power_T35_user;
Power_byT.T20   = Power_T20_user;
Power_byT.T10   = Power_T10_user;
Power_byT.T0    = Power_T0_user;

%% ── 피처명 구성 --------------------------------------------------------
scalar_pool_names = {'QC2','QC40','Rcharg','R1s','Rcharg_80_90_avg'};
for t = TEMP_list_scalar
    scalar_pool_names{end+1} = sprintf('DCIR_1s_T%d', t);            %#ok<AGROW>
    scalar_pool_names{end+1} = sprintf('DCIR_10s_T%d', t);           %#ok<AGROW>
    scalar_pool_names{end+1} = sprintf('DCIR_delta_10s_1s_T%d', t);  %#ok<AGROW>
    scalar_pool_names{end+1} = sprintf('Power_T%d', t);              %#ok<AGROW>
end

active_scalar_names = {'QC2','QC40'};
if USE_RCHARG,          active_scalar_names{end+1} = 'Rcharg';           end
if USE_RCHARG_8090,     active_scalar_names{end+1} = 'Rcharg_80_90_avg'; end
if USE_R1S,             active_scalar_names{end+1} = 'R1s';              end

for t = TEMP_list_scalar
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

    for t = TEMP_list_scalar
        keyT = sprintf('T%d', t);

        if isfield(DCIR1s_byT, keyT),  d1  = DCIR1s_byT.(keyT)(i);  else, d1  = NaN; end
        if isfield(DCIR10s_byT, keyT), d10 = DCIR10s_byT.(keyT)(i); else, d10 = NaN; end
        if isfield(DCIRd_byT, keyT),   dd  = DCIRd_byT.(keyT)(i);   else, dd  = NaN; end
        if isfield(Power_byT, keyT),   pw  = Power_byT.(keyT)(i);   else, pw  = NaN; end

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

if isempty(feat_active_valid)
    error('유효한 행이 없습니다. active feature 구성 또는 NaN 포함 여부를 확인하세요.');
end

%% ── CSV 저장 -----------------------------------------------------------
T_full   = array2table(feat_full, 'VariableNames', feat_full_names, 'RowNames', cell_names);
T_active = array2table(feat_active_valid, 'VariableNames', feat_active_names, 'RowNames', cells_active_valid);

csv_full   = fullfile(save_path, 'features_allvars_full.csv');
csv_active = fullfile(save_path, 'features_allvars_active.csv');
writetable(T_full,   csv_full,   'WriteRowNames', true);
writetable(T_active, csv_active, 'WriteRowNames', true);

disp(['features_allvars_full.csv 저장: ' csv_full]);
disp(['features_allvars_active.csv 저장: ' csv_active]);

%% ── Label / Feature 분리 -----------------------------------------------
% x축: ECM feature 전체
feature_names_ecm = feat2RC_names;

% y축: active scalar 전체 자동 선택
label_names = active_scalar_names;

idx_feat_ecm = find(ismember(feat_active_names, feature_names_ecm));
idx_label    = find(ismember(feat_active_names, label_names));

if isempty(idx_feat_ecm)
    error('ECM feature가 없습니다.');
end
if isempty(idx_label)
    error('선택된 active scalar label이 없습니다. USE_* 토글을 확인하세요.');
end

X_feat  = feat_active_valid(:, idx_feat_ecm);   % x축: ECM features
X_label = feat_active_valid(:, idx_label);      % y축: scalar labels

feat_names_plot  = feat_active_names(idx_feat_ecm);
label_names_plot = feat_active_names(idx_label);

%% ── 상관 계산 ----------------------------------------------------------
[R_lf, P_lf] = corr(X_label, X_feat, 'Type','Pearson', 'Rows','pairwise');

% 원하면 |r| 큰 feature만 남김
if ~isempty(SHOW_ONLY_ABS_R_OVER)
    maxAbsCorr = max(abs(R_lf), [], 1, 'omitnan');
    keep_feat = maxAbsCorr >= SHOW_ONLY_ABS_R_OVER;

    X_feat = X_feat(:, keep_feat);
    feat_names_plot = feat_names_plot(keep_feat);
    [R_lf, P_lf] = corr(X_label, X_feat, 'Type','Pearson', 'Rows','pairwise');
end

nL = size(R_lf, 1);
nF = size(R_lf, 2);

if nF == 0
    error('표시할 feature가 없습니다. SHOW_ONLY_ABS_R_OVER 조건을 완화하세요.');
end

%% ── Heatmap: Labels vs ECM features only -------------------------------
outfig2 = fullfile(save_path, 'corr_heatmap_labels_vs_ecm.fig');
outpng2 = fullfile(save_path, 'corr_heatmap_labels_vs_ecm.png');

anchors = [ ...
    0.00 0.20 0.80;
    1.00 1.00 1.00;
    0.80 0.10 0.10];
xs   = [-1 0 1];
cmap = interp1(xs, anchors, linspace(-1,1,256), 'linear','extrap');

figW = max(1400, 220 + 42*nF);
figH = max(480,  220 + 45*nL);

f2 = figure('Color','w','Position',[80 80 figW figH]);
imagesc(R_lf, [-1 1]);
colormap(cmap);

cb = colorbar;
cb.Label.String = 'Pearson r';
set(cb,'Ticks',-1:0.5:1,'TickLabels',compose('%.1f',-1:0.5:1));

ax = gca;
set(ax, 'XTick', 1:nF, 'YTick', 1:nL, ...
    'FontSize', 10, 'FontWeight', 'bold', ...
    'TickLabelInterpreter', 'tex');

% x축 라벨
xlabels = arrayfun(@prettyVarLabel, feat_names_plot, 'UniformOutput', false);
xlabels = cellfun(@(c) joinPrettyLabel(c), xlabels, 'UniformOutput', false);

% y축 라벨
ylabels = arrayfun(@prettyVarLabel, label_names_plot, 'UniformOutput', false);
ylabels = cellfun(@(c) strjoin(c, ' '), ylabels, 'UniformOutput', false);

set(ax, 'XTickLabel', xlabels, 'YTickLabel', ylabels);
xtickangle(ax, 45);

xlabel('Features (ECM parameters)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Labels (SOH, SOH-x)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Correlation: Labels vs ECM Features | Loads: %s', strjoin(load_use, ', ')), ...
    'Interpreter', 'none', 'FontSize', 13, 'FontWeight', 'bold');

hold on;

% 격자선
for k = 0.5:1:nL+0.5
    plot([0.5 nF+0.5], [k k], 'w-', 'LineWidth', 0.5);
end
for k = 0.5:1:nF+0.5
    plot([k k], [0.5 nL+0.5], 'w-', 'LineWidth', 0.5);
end

% 셀 텍스트
for i = 1:nL
    for j = 1:nF
        r = R_lf(i,j);
        pval = P_lf(i,j);

        if isnan(r), continue; end

        idx = max(1, min(256, 1 + round((r+1)/2 * 255)));
        cc  = cmap(idx,:);
        YIQ = 0.299*cc(1) + 0.587*cc(2) + 0.114*cc(3);

        if YIQ < 0.5
            tcol = [1 1 1];
        else
            tcol = [0 0 0];
        end

        if SHOW_P_IN_HEATMAP
            txt = sprintf('r=%.2f\np=%.3g', r, pval);
        else
            txt = sprintf('%.2f', r);
        end

        text(j, i, txt, ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'FontSize', 9, ...
            'FontWeight', 'bold', ...
            'Color', tcol);

        if MARK_STRONG_CORR && isfinite(r) && abs(r) >= STRONG_CORR_TH
            text(j, i-0.30, '*', ...
                'HorizontalAlignment','center', ...
                'VerticalAlignment','middle', ...
                'FontSize',12, ...
                'FontWeight','bold', ...
                'Color',[1 1 0]);
        end
    end
end

savefig(f2, outfig2);
exportgraphics(f2, outpng2, 'Resolution', 220);

%% ── 저장: Labels vs ECM -----------------------------------------------
Tcorr = array2table(R_lf, 'VariableNames', feat_names_plot, 'RowNames', label_names_plot);
Tpval = array2table(P_lf, 'VariableNames', feat_names_plot, 'RowNames', label_names_plot);

writetable(Tcorr, fullfile(save_path, 'corr_labels_vs_ecm.csv'), 'WriteRowNames', true);
writetable(Tpval, fullfile(save_path, 'pval_labels_vs_ecm.csv'), 'WriteRowNames', true);

save(fullfile(save_path, 'corr_labels_vs_ecm.mat'), ...
    'R_lf', 'P_lf', 'feat_names_plot', 'label_names_plot', ...
    'cells_active_valid', 'SOC_use', 'LOAD_USE_STR', ...
    'TEMP_list_load', 'TEMP_list_scalar', 'cell_ids', ...
    'SHOW_P_IN_HEATMAP', 'MARK_STRONG_CORR', 'STRONG_CORR_TH', 'SHOW_ONLY_ABS_R_OVER');

disp('corr_labels_vs_ecm.csv / pval_labels_vs_ecm.csv / corr_labels_vs_ecm.mat 저장 완료');

%% ========================= 보조 함수들 =================================
function loads = parseLoadList(str0)
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

        % 1순위: 문자열 맨 앞 x01, X01
        tokx = regexp(s, '^[xX]\s*0*(\d+)', 'tokens', 'once');
        if ~isempty(tokx)
            ids(k) = str2double(tokx{1});
            continue;
        end

        % 2순위: 문자열에서 처음 나오는 숫자
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

function s = joinPrettyLabel(c)
    if numel(c) == 1
        s = c{1};
    else
        s = sprintf('%s\\newline%s', c{:});
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

    % 온도별 스칼라
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

    % 주행부하 온도 포함 2RC 피처
    tok2 = regexp(name,'^T(\d+)_([A-Za-z0-9]+)_(R0|R1|R2|tau1|tau2)_(\d+)$','tokens','once');
    if ~isempty(tok2)
        Tval     = str2double(tok2{1});
        loadName = tok2{2};
        pName    = tok2{3};
        socVal   = str2double(tok2{4});

        switch string(pName)
            case "R0",   base = 'R_{0}';
            case "R1",   base = 'R_{1}';
            case "R2",   base = 'R_{2}';
            case "tau1", base = '\tau_{1}';
            case "tau2", base = '\tau_{2}';
            otherwise,   base = strrep(pName,'_','\_');
        end

        lab = {sprintf('%s (%s)', base, loadName); ...
               sprintf('(SOC %d%%, %d^{\\circ}C)', socVal, Tval)};
        return
    end

    % fallback
    tok3 = regexp(name,'^([A-Za-z0-9]+)_(R0|R1|R2|tau1|tau2)_(\d+)$','tokens','once');
    if ~isempty(tok3)
        loadName = tok3{1};
        pName    = tok3{2};
        socVal   = str2double(tok3{3});

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

    lab = {strrep(name0,'_','\_')};
end