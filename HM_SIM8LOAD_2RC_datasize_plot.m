% ======================================================================
%  SOC70_US06_UDDS_forPPT.xlsx + Length_sec 시트 기반 3×2 플롯 (+ DCIR 점)
%
%  • 입력:
%     1) folder_SIM  (원래 SIM_parsed\셀정렬 폴더)
%     2) SOC70_US06_UDDS_forPPT.xlsx
%        - 각 셀 시트: Length_label + US06/UDDS 파라미터/ RMSE
%        - Length_sec 시트: Cell, US06_full_s, UDDS_full_s
%     3) cap_user : 셀 용량(예: QC40 등), 색 농도 조절용 (직접 입력)
%     4) DCIR ECM parameters: 셀별 벡터 (각 파라미터당 nCells × 1, 안 넣으면 패스)
%
%  • 출력:
%     - 3×2 figure (R0,RMSE,R1,tau1,R2,tau2 vs data length)
%       US06 = 오렌지, UDDS = 파랑, 셀별로 색 농도 다르게
%       DCIR 값은 x=180 s 위치에 (셀별 용량 기반 그라디언트) 다이아몬드로 표시
% ======================================================================
clc; clear; close all;

%% (0) 경로 설정 & 엑셀 파일
folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_agedcell_8_processed\SIM_parsed\셀정렬';
% folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_fresh_integrated_7_Drivingprocessed\SIM_parsed';
save_root  = fileparts(folder_SIM);             % ...\SIM_parsed
save_path  = fullfile(save_root,'2RC_fitting'); % 2RC_fitting 폴더

xlsx_ppt = fullfile(save_path,'SOC70_US06_UDDS_forPPT.xlsx');
if ~exist(xlsx_ppt,'file')
    error('엑셀 파일을 찾을 수 없습니다: %s', xlsx_ppt);
end
fprintf('엑셀 파일: %s\n', xlsx_ppt);

%% (1) 시트 목록 & Length_sec 읽기
[~, sheets] = xlsfinfo(xlsx_ppt);
if isempty(sheets)
    error('엑셀 파일에 시트가 없습니다.');
end

% Length_sec 시트 분리
idx_len = strcmp(sheets,'Length_sec');
if ~any(idx_len)
    error('Length_sec 시트를 찾을 수 없습니다. 생성 코드가 수정됐는지 확인하세요.');
end

sheet_len   = sheets{idx_len};
cellSheets  = sheets(~idx_len);  % 실제 셀 데이터 시트들
nCells2     = numel(cellSheets);

% Length_sec 읽기
TlenInfo = readtable(xlsx_ppt,'Sheet',sheet_len,'VariableNamingRule','preserve');

% Cell 이름 기준으로 full length lookup 만들기
lenMap = containers.Map();   % key: sheetName, value: [US06_full, UDDS_full]
for i = 1:height(TlenInfo)
    key   = string(TlenInfo.Cell(i));
    ufull = TlenInfo.US06_full_s(i);
    dfull = TlenInfo.UDDS_full_s(i);
    lenMap(char(key)) = [ufull, dfull];
end

fprintf('Length_sec: %d cells loaded.\n', height(TlenInfo));

%% (2) 셀 용량 입력 (색 농도용)
%  - sheet 순서(cellSheets)와 동일 순서로 용량 입력
cap_user = [57.49
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
             58.40];
% cap_user = [];

if isempty(cap_user)
    warning('cap_user 가 비어 있습니다. 색 농도는 셀 index 기반으로만 설정합니다.');
end

if ~isempty(cap_user) && numel(cap_user) ~= nCells2
    error('cap_user 길이(%d)와 셀 시트 개수(%d)가 다릅니다.', ...
        numel(cap_user), nCells2);
end

%% (2.5) DCIR ECM parameter per-cell (각 파라미터당 nCells2 × 1, 안 넣으면 패스)
%  단위:
%   - DCIR_R0, R1, R2 : mΩ
%   - DCIR_tau1, tau2 : s
%   - DCIR_RMSE       : V

% 필요할 때만 값 채워 넣기 (안 쓰는 건 [] 그대로 두면 자동으로 NaN으로 처리)

% 고품셀
DCIR_R0_in = [0.93
0.89
1.34
1.27
1.43
1.25
0.96
0.92
2.27
0.92
0.91
2.44
0.95]; 

DCIR_R1_in   = [
0.55
0.48
0.70
0.67
0.61
0.72
0.53
0.49
0.82
0.49
0.50
0.90
0.49];   % 예: [ ... ] 또는 [] 그대로 두기
DCIR_R2_in   = [2.24
2.16
2.95
2.76
2.41
2.88
2.08
2.16
5.76
2.19
2.18
6.99
2.15];
DCIR_tau1_in = [9.91
9.90
9.77
9.98
9.89
9.92
9.91
9.92
8.05
9.90
9.90
7.55
9.90];
DCIR_tau2_in = [121.05
117.08
174.55
132.05
126.79
153.10
111.93
118.26
199.19
117.94
116.75
162.75
114.69];
DCIR_RMSE_in = [0.79
0.79
0.52
0.86
0.88
0.82
0.86
0.79
0.99
0.81
0.82
0.93
0.8];

% % 신품셀
% DCIR_R0_in = [0.87
% 0.88
% 0.97
% 0.92
% 0.98
% 0.96]; 
% 
% DCIR_R1_in   = [
% 0.53
% 0.54
% 0.54
% 0.54
% 0.54
% 0.53];   % 예: [ ... ] 또는 [] 그대로 두기
% DCIR_R2_in   = [2.09
% 2.09
% 2.14
% 2.09
% 2.10
% 2.08];
% DCIR_tau1_in = [9.91
% 9.61
% 9.93
% 9.91
% 9.60
% 9.91];
% DCIR_tau2_in = [111.64
% 113.23
% 112.70
% 112.40
% 111.19
% 111.93];
% DCIR_RMSE_in = [0.88
% 0.9
% 0.85
% 0.87
% 0.89
% 0.86];

% mV -> V 단위 변환 (입력이 있을 때만)
if ~isempty(DCIR_RMSE_in)
    DCIR_RMSE_in = DCIR_RMSE_in * 1e-3;
end

% 내부용 벡터로 정리 (길이 자동 맞추기, 비어 있으면 NaN 채우기)
DCIR_R0   = expandDCIR(DCIR_R0_in  , nCells2, 'DCIR_R0');
DCIR_R1   = expandDCIR(DCIR_R1_in  , nCells2, 'DCIR_R1');
DCIR_R2   = expandDCIR(DCIR_R2_in  , nCells2, 'DCIR_R2');
DCIR_tau1 = expandDCIR(DCIR_tau1_in, nCells2, 'DCIR_tau1');
DCIR_tau2 = expandDCIR(DCIR_tau2_in, nCells2, 'DCIR_tau2');
DCIR_RMSE = expandDCIR(DCIR_RMSE_in, nCells2, 'DCIR_RMSE');

% struct 로 정리 (파라미터 이름 매칭용)
DCIR_cell = struct( ...
    'R0'  , DCIR_R0(:), ...
    'R1'  , DCIR_R1(:), ...
    'R2'  , DCIR_R2(:), ...
    'tau1', DCIR_tau1(:), ...
    'tau2', DCIR_tau2(:), ...
    'RMSE', DCIR_RMSE(:));

DCIR_xpos = 180;   % DCIR 점을 찍을 x 위치 [s]

%% (3) 색상 설정 (US06=오렌지, UDDS=파랑, 용량 기반 농도)
baseBlue    = [0 0.4470 0.7410];      % UDDS
baseOrange  = [0.8500 0.3250 0.0980]; % US06
baseBlack   = [0 0 0];                % DCIR용 기본 색

color_UDDS  = cell(nCells2,1);
color_US06  = cell(nCells2,1);
color_DCIR  = cell(nCells2,1);

alpha_min = 0.35;
alpha_max = 0.95;

alpha_vals = zeros(nCells2,1);

if ~isempty(cap_user)
    cap_user = cap_user(:);
    cap_min  = min(cap_user);
    cap_max  = max(cap_user);
    for i = 1:nCells2
        if cap_max==cap_min
            alpha = 0.5*(alpha_min+alpha_max);
        else
            alpha = alpha_min + (cap_user(i)-cap_min)/(cap_max-cap_min) ...
                               * (alpha_max-alpha_min);
        end
        alpha_vals(i) = alpha;
        color_UDDS{i} = (1-alpha)*[1 1 1] + alpha*baseBlue;
        color_US06{i} = (1-alpha)*[1 1 1] + alpha*baseOrange;
        color_DCIR{i} = (1-alpha)*[1 1 1] + alpha*baseBlack; % 검은색 계열 그라디언트
    end
else
    % 용량 안 넣었으면 index 기반으로 gradient
    for i = 1:nCells2
        alpha = alpha_min + (alpha_max-alpha_min)*(i-1)/max(nCells2-1,1);
        alpha_vals(i) = alpha;
        color_UDDS{i} = (1-alpha)*[1 1 1] + alpha*baseBlue;
        color_US06{i} = (1-alpha)*[1 1 1] + alpha*baseOrange;
        color_DCIR{i} = (1-alpha)*[1 1 1] + alpha*baseBlack;
    end
end

%% (4) 플롯 준비
paramNames  = {'R0','RMSE','R1','tau1','R2','tau2'};  % 타일 순서
paramTitles = {'R_0 (m\Omega)', 'RMSE (V)', ...
               'R_1 (m\Omega)', '\tau_1 (s)', ...
               'R_2 (m\Omega)', '\tau_2 (s)'};

fig2 = figure('Name','SOC70 – US06/UDDS params vs length (Excel+Length+DCIR point)', ...
              'NumberTitle','off','Color','w', ...
              'Position',[100 100 1500 800]);
tl2 = tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

%% (5) 파라미터별 3×2 서브플롯
for p = 1:numel(paramNames)
    pname = paramNames{p};
    ptit  = paramTitles{p};

    ax = nexttile; hold(ax,'on'); grid(ax,'on');

    Y_all = [];
    X_all = [];

    for i = 1:nCells2
        sheetName = cellSheets{i};
        T = readtable(xlsx_ppt, 'Sheet', sheetName, 'VariableNamingRule','preserve');

        % Length_label → 초 단위 (US06/UDDS 각각 full length 사용)
        len_label = T.Length_label;
        nRow      = numel(len_label);

        len_sec_US06 = nan(nRow,1);
        len_sec_UDDS = nan(nRow,1);

        % full length lookup
        if isKey(lenMap, sheetName)
            fulls = lenMap(sheetName);
            full_US06 = fulls(1);
            full_UDDS = fulls(2);
        else
            warning('Length_sec에 %s 정보를 찾을 수 없습니다. 기본값 600 s 사용.', sheetName);
            full_US06 = 600;
            full_UDDS = 600;
        end

        for r = 1:nRow
            lab = string(len_label{r});
            nums = regexp(lab,'\d+','match');
            if ~isempty(nums)
                val = str2double(nums{1});
                len_sec_US06(r) = val;
                len_sec_UDDS(r) = val;
            else
                % "원본길이" 같은 case → full length 사용
                len_sec_US06(r) = full_US06;
                len_sec_UDDS(r) = full_UDDS;
            end
        end

        % 파라미터 값 선택 (US06 / UDDS)
        switch pname
            case 'R0'
                y_US06 = T.US06_R0_mOhm;
                y_UDDS = T.UDDS_R0_mOhm;
            case 'R1'
                y_US06 = T.US06_R1_mOhm;
                y_UDDS = T.UDDS_R1_mOhm;
            case 'R2'
                y_US06 = T.US06_R2_mOhm;
                y_UDDS = T.UDDS_R2_mOhm;
            case 'tau1'
                y_US06 = T.US06_tau1_s;
                y_UDDS = T.UDDS_tau1_s;
            case 'tau2'
                y_US06 = T.US06_tau2_s;
                y_UDDS = T.UDDS_tau2_s;
            case 'RMSE'
                y_US06 = T.US06_RMSE_V;
                y_UDDS = T.UDDS_RMSE_V;
            otherwise
                error('Unknown parameter name: %s', pname);
        end

        y_US06 = y_US06(:);
        y_UDDS = y_UDDS(:);

        % ---- US06 ----
        mask_u = ~isnan(len_sec_US06) & ~isnan(y_US06);
        if any(mask_u)
            x_u = len_sec_US06(mask_u);
            y_u = y_US06(mask_u);
            Y_all = [Y_all; y_u]; %#ok<AGROW>
            X_all = [X_all; x_u]; %#ok<AGROW>

            niceName = strrep(sheetName,'_','\_');
            if p==1
                plot(ax, x_u, y_u, '-o', 'LineWidth',1.6, ...
                    'MarkerSize',6, 'Color',color_US06{i}, ...
                    'DisplayName',[niceName ' (US06)']);
            else
                plot(ax, x_u, y_u, '-o', 'LineWidth',1.6, ...
                    'MarkerSize',6, 'Color',color_US06{i}, ...
                    'HandleVisibility','off');
            end
        end

        % ---- UDDS ----
        mask_d = ~isnan(len_sec_UDDS) & ~isnan(y_UDDS);
        if any(mask_d)
            x_d = len_sec_UDDS(mask_d);
            y_d = y_UDDS(mask_d);
            Y_all = [Y_all; y_d]; %#ok<AGROW>
            X_all = [X_all; x_d]; %#ok<AGROW>

            niceName = strrep(sheetName,'_','\_');
            if p==1
                plot(ax, x_d, y_d, '-s', 'LineWidth',1.6, ...
                    'MarkerSize',6, 'Color',color_UDDS{i}, ...
                    'DisplayName',[niceName ' (UDDS)']);
            else
                plot(ax, x_d, y_d, '-s', 'LineWidth',1.6, ...
                    'MarkerSize',6, 'Color',color_UDDS{i}, ...
                    'HandleVisibility','off');
            end
        end

        % ---- DCIR 점 (x = 180 s, 셀별 값) --------------------------
        dcir_vec = DCIR_cell.(pname);
        if ~isempty(dcir_vec) && ~isnan(dcir_vec(i))
            x_dcir = DCIR_xpos;
            y_dcir = dcir_vec(i);

            Y_all = [Y_all; y_dcir];
            X_all = [X_all; x_dcir];

            if p==1 && i==1
                plot(ax, x_dcir, y_dcir, 'd', ...
                    'MarkerSize',7, 'MarkerFaceColor',color_DCIR{i}, ...
                    'MarkerEdgeColor','k', ...
                    'DisplayName','DCIR @ 180 s');
            else
                plot(ax, x_dcir, y_dcir, 'd', ...
                    'MarkerSize',7, 'MarkerFaceColor',color_DCIR{i}, ...
                    'MarkerEdgeColor','k', ...
                    'HandleVisibility','off');
            end
        end
    end

    xlabel(ax,'Data length (s)');
    ylabel(ax, ptit, 'Interpreter','tex');
    title(ax, ptit, 'Interpreter','tex');

    % y축 0부터 시작
    if ~isempty(Y_all)
        ymax = max(Y_all(:), [], 'omitnan');
        ymin = min(Y_all(:), [], 'omitnan');
        if ~isfinite(ymin) || ymin >= 0, ymin = 0; end
        if ~isfinite(ymax), ymax = 1; else, ymax = ymax*1.05; end
        ylim(ax,[ymin ymax]);
    else
        ylim(ax,[0 1]);
    end

    % x축 스케일 설정
    if ~isempty(X_all)
        xmax = max(X_all);
        xlim(ax,[0 xmax*1.05]);
    end

    if p==1
        legend(ax,'Location','best','Interpreter','tex');
    end
end

title(tl2,'SOC70 – US06 / UDDS 2RC parameters vs data length','Interpreter','none');

fig2_name = 'SOC70_US06_UDDS_params_vs_length_fromExcel_Length_DCIRpoint';
savefig(fig2, fullfile(save_path, [fig2_name '.fig']));
exportgraphics(fig2, fullfile(save_path, [fig2_name '.png']), 'Resolution', 200);

fprintf('→ 엑셀+Length_sec+DCIR point 기반 3×2 subplot figure 저장 완료: %s\n', ...
    fullfile(save_path, [fig2_name '.png']));

%% ── 보조 함수: DCIR 입력 벡터 확장 ────────────────────────────────
function v = expandDCIR(v_in, nCells, nameStr)
    if isempty(v_in)
        v = NaN(nCells,1);
    else
        v = v_in(:);
        if numel(v) ~= nCells
            error('%s 길이(%d)와 셀 개수(%d)가 다릅니다.', ...
                nameStr, numel(v), nCells);
        end
    end
end
