%% =======================================================================
%  2RC 파라미터 비교 플랏 (Tbl_ECM_mean → 바로 그리기)
%  - 입력: 2RC_results.mat (Tbl_ECM_mean 포함)
%  - Tbl_ECM_mean 구조:
%       RowNames: 셀 이름(파일명 기반, 예: x05_HNE_35degC____US06_5_7_1108)
%       VarNames: SOC90_R0_mOhm, SOC90_R1_mOhm, ..., SOC30_tau2
%  - 출력: 파라미터별(R0,R1,R2,τ1,τ2) vs SOC 비교 플랏(.fig/.png)
%  - 컬러바는 capVec(QC/40 등 스칼라)로 매핑
%  - 레전드 라벨은 예: "x05_..." → "05" 로 축약
% =======================================================================

clear; clc; close all;

%% [1] 입력 설정 ----------------------------------------------------------
% 2RC_results.mat 경로 (위 fitting 코드에서 저장한 위치)
matPath = 'G:\공유 드라이브\BSL_Data4\HNE_agedcell_8_processed\SIM_parsed\2RC_fitting\2RC_results.mat'; % 2RC parameter 저장 mat

% 색상 매핑용 스칼라(capacity/SOH 등). 길이 = 셀 개수(N).
% (Tbl_ECM_mean 로드 후 N과 길이 다르면 자동으로 보정해줌)
capVec = [57.49
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
58.4];   % 예시: QC/40

% ✅ 결과 저장 폴더 (네가 지정한 경로)
outDir = 'G:\공유 드라이브\BSL_Data4\HNE_agedcell_8_processed\SIM_parsed\2RC_fitting\parameter_plot';

% 컬러바 라벨
capLabel = 'Capacity (QC/40, Ah)';

% 제목/파일 prefix
titlePrefix = '2RC Driving';
filePrefix  = 'Compare_';

% 범례 표시 여부
doLegend = true;

%% [2] Tbl_ECM_mean 로드 및 파라미터 배열 구성 ----------------------------
S = load(matPath, 'Tbl_ECM_mean');
if ~isfield(S,'Tbl_ECM_mean')
    error('Tbl_ECM_mean 이 %s 에서 발견되지 않았습니다.', matPath);
end
Tbl_ECM_mean = S.Tbl_ECM_mean;

% --- RowNames에서 "짧은 셀 이름" 뽑기 ----------------------------------
% 예: 'x05_HNE_35degC____US06_5_7_1108' → '05'
rawNames = Tbl_ECM_mean.Properties.RowNames;
if isstring(rawNames)
    rawNames = cellstr(rawNames);
elseif ischar(rawNames)
    rawNames = cellstr(rawNames);
end

N = numel(rawNames);
cellNames = cell(size(rawNames));   % 레전드에 쓸 짧은 이름

for i = 1:N
    s = char(rawNames{i});
    % 패턴: x + 숫자들 (예: x05, x4, x123 ...)
    tok = regexp(s, '^x(\d+)', 'tokens', 'once');
    if ~isempty(tok)
        cellNames{i} = tok{1};      % '05'만 사용
    else
        % x로 안 시작하면 원래 이름 그대로
        cellNames{i} = s;
    end
end

% SOC 축 (열 방향 순서)
SOCx = [90 70 50 30];
K    = numel(SOCx);

% capVec 길이 체크 (다르면 자동 재생성)
if numel(capVec) ~= N
    warning('capVec 길이(%d)가 셀 수(%d)와 다릅니다. min~max 범위에서 자동으로 재생성합니다.', ...
        numel(capVec), N);
    if isempty(capVec)
        capVec = linspace(48, 59, N)';            % 완전 비어 있으면 대충 만들어 줌
    else
        capVec = linspace(min(capVec), max(capVec), N)';  % 보간
    end
else
    capVec = capVec(:);   % 컬럼 벡터로 정리
end

% P 구조체 만들기: 각 필드가 [N(셀) × K(SOC)] 배열
pNames = {'R0','R1','R2','tau1','tau2'};
for p = 1:numel(pNames)
    P.(pNames{p}) = nan(N, K);
end

% Tbl_ECM_mean → P.R0 / P.R1 / ... / P.tau2 채우기
for k = 1:K
    soc = SOCx(k);
    for pi = 1:numel(pNames)
        pname = pNames{pi};
        if pi <= 3
            % R0,R1,R2 : mΩ
            varName = sprintf('SOC%d_%s_mOhm', soc, pname);
        else
            % tau1,tau2 : s
            varName = sprintf('SOC%d_%s', soc, pname);
        end

        if ~ismember(varName, Tbl_ECM_mean.Properties.VariableNames)
            error('변수 %s 를 Tbl_ECM_mean에서 찾을 수 없습니다.', varName);
        end

        P.(pname)(:,k) = Tbl_ECM_mean{:, varName};
    end
end

%% [3] 플랏 생성 ----------------------------------------------------------
plot_2RC_compare_vs_SOC_custom__INLINE__(cellNames, SOCx, P, capVec, ...
    'SavePath', outDir, ...
    'CapLabel', capLabel, ...
    'TitlePrefix', titlePrefix, ...
    'FilePrefix', filePrefix, ...
    'DoLegend', doLegend);

disp('완료: Tbl_ECM_mean 기준 플롯이 저장되었습니다.');

%% ====================== (내장 함수) 플랏 ================================
function plot_2RC_compare_vs_SOC_custom__INLINE__(cellNames, SOCx, P, capVec, varargin)
    % 옵션 파싱
    ip = inputParser;
    ip.addParameter('SavePath', pwd);
    ip.addParameter('CapLabel', 'Capacity (Ah)');
    ip.addParameter('TitlePrefix', '');
    ip.addParameter('FilePrefix', '');
    ip.addParameter('DoLegend', true);
    ip.parse(varargin{:});
    opt = ip.Results;

    N = numel(cellNames);

    % --- 파스텔 레드↔블루 계열 컬러맵 생성 ---
    Nmap = 256;
    anchors = [0.88 0.16 0.24; 0.83 0.70 0.86; 0.16 0.38 0.92];
    x  = [0 0.5 1];
    xi = linspace(0,1,Nmap)';
    cmap = [interp1(x,anchors(:,1),xi,'pchip'), ...
            interp1(x,anchors(:,2),xi,'pchip'), ...
            interp1(x,anchors(:,3),xi,'pchip')];
    cmap = min(max(cmap,0),1);
    hsv = rgb2hsv(cmap);
    hsv(:,2) = max(0.35,hsv(:,2));
    hsv(:,2) = min(1.0,hsv(:,2)*1.2);
    hsv(:,3) = max(0.75,hsv(:,3)*0.95);
    cmap = hsv2rgb(hsv);

    capMin = min(capVec); 
    capMax = max(capVec);
    mapColor = @(v) cmap( ...
        max(1, min(Nmap, 1 + round((v-capMin)/max(capMax-capMin,eps)*(Nmap-1)))), : );

    % 개별 파라미터 플롯 함수
    function draw_one(paramName, Y, yunit)
        fig = figure('Name',paramName+" vs SOC (2RC)", 'Color','w');
        hold on; grid on;

        for i = 1:N
            col = mapColor(capVec(i));
            plot(SOCx, Y(i,:), '-o', ...
                'LineWidth', 1.8, 'Color', col, ...
                'MarkerFaceColor', col, ...
                'DisplayName', cellNames{i});
        end
        xlabel('SOC (%)');
        ylabel(paramName + " (" + yunit + ")");
        title(strtrim(strjoin([opt.TitlePrefix, paramName+" vs SOC"], " ")));

        % y축 0부터 시작
        ymax = max(Y(:), [], 'omitnan');
        if isfinite(ymax) && ymax > 0
            ylim([0, ymax*1.05]);
        else
            ylim([0, 1]);
        end

        colormap(cmap);
        cb = colorbar('Location','eastoutside');
        cb.Label.String = opt.CapLabel;
        clim([capMin capMax]);

        if opt.DoLegend
            % 인터프리터를 'none'으로 해서 언더스코어/TeX 문제 방지
            legend('Location','best', 'Interpreter','none');
        end

        if ~exist(opt.SavePath,'dir'), mkdir(opt.SavePath); end
        savefig(fig, fullfile(opt.SavePath, ...
            sprintf('%s%s_vs_SOC.fig', opt.FilePrefix, paramName)));
        exportgraphics(fig, fullfile(opt.SavePath, ...
            sprintf('%s%s_vs_SOC.png', opt.FilePrefix, paramName)), ...
            'Resolution', 200);
    end

    % ---- 단위 표기 버전 ----
    draw_one('R0',   P.R0,   'mΩ');
    draw_one('R1',   P.R1,   'mΩ');
    draw_one('R2',   P.R2,   'mΩ');
    draw_one('tau1', P.tau1, 's');
    draw_one('tau2', P.tau2, 's');
end
