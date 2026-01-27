%% =======================================================================
%  HM_2RC_Temp_plot_US06_SOC50
%  - 입력: 각 온도 폴더의 2RC_results_600s.mat
%  - US06 테이블(Tbl_Load_ECM.US06 or Tbl_US06_ECM)
%  - SOC 50 고정
%  - x축: 온도(0/10/20) / xlim [-10 30]
%  - (A PATCH) RowNames 직접 교집합 대신 "셀 키" 추출 후 매칭
%% =======================================================================
clear; clc; close all;

% (A PATCH) Text/TeX interpreter 에러 원천 차단
set(groot,'defaultTextInterpreter','none');
set(groot,'defaultAxesTickLabelInterpreter','none');
set(groot,'defaultLegendInterpreter','none');

%% ---------------- 사용자 설정 ----------------
paths = [
"G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\0degC\2RC_fitting_600s"
"G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\10degC\2RC_fitting_600s"
"G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\20degC\2RC_fitting_600s"
];

tempsC = [0 10 20];          % x축 온도 (paths 순서와 동일해야 함)
socPick = 50;                % SOC 50 고정
loadPick = "US06";           % US06 고정

% 색상 매핑용 스칼라(capacity/SOH 등)
capVec = [57.49;57.57;54;52.22;53.45;51.28;57.91;56.51;42.14;57.27;57.18;58.4];

outDir = "G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\2RC_temp_plot_US06_SOC50";
capLabel = "Capacity (QC/40, Ah)";
titlePrefix = "US06 2RC @ SOC50";
filePrefix  = "US06_SOC50_";
doLegend = true;
xlimTemp = [-2 22];

%% ---------------- 로드 & 파라미터 수집 ----------------
assert(numel(paths)==numel(tempsC), "paths 개수와 tempsC 개수가 다릅니다.");

pNames = {'R0','R1','R2','tau1','tau2'};
nP = numel(pNames);

D = struct();  % D(ti).T, D(ti).keys, D(ti).P (nCell x 5)

for ti = 1:numel(tempsC)
    T = tempsC(ti);

    matPath = fullfile(paths(ti), "2RC_results_600s.mat");
    if ~isfile(matPath)
        error("mat 파일이 없습니다: %s", matPath);
    end

    S = load(matPath);

    % --- US06 테이블 가져오기 (두 형태 모두 지원) ---
    TblUS06 = [];
    if isfield(S,'Tbl_Load_ECM') && isstruct(S.Tbl_Load_ECM) && isfield(S.Tbl_Load_ECM, loadPick)
        TblUS06 = S.Tbl_Load_ECM.(loadPick);
    else
        varName = sprintf("Tbl_%s_ECM", loadPick);
        if isfield(S, varName)
            TblUS06 = S.(varName);
        end
    end

    if isempty(TblUS06) || ~istable(TblUS06)
        error("[%ddegC] US06 ECM 테이블을 찾지 못했습니다. (Tbl_Load_ECM.US06 또는 Tbl_US06_ECM 확인)", T);
    end

    rawNames = TblUS06.Properties.RowNames;
    if isstring(rawNames), rawNames = cellstr(rawNames); end
    rawNames = rawNames(:);

    nCell = numel(rawNames);
    keys = cell(nCell,1);
    for i = 1:nCell
        keys{i} = extractCellKey(rawNames{i});   % (A PATCH) 핵심: 키 추출
    end

    % --- SOC50 파라미터 추출 ---
    Pmat = nan(nCell, nP);
    for pi = 1:nP
        pname = pNames{pi};
        if pi <= 3
            vname = sprintf("SOC%d_%s_mOhm", socPick, pname);
        else
            vname = sprintf("SOC%d_%s", socPick, pname);
        end

        if ~ismember(vname, TblUS06.Properties.VariableNames)
            error("[%ddegC] 변수 %s 가 US06 테이블에 없습니다.", T, vname);
        end
        Pmat(:,pi) = TblUS06{:, vname};
    end

    D(ti).T    = T;
    D(ti).keys = keys;
    D(ti).P    = Pmat;
end

%% ---------------- 공통 셀 매칭 (키 기반) ----------------
commonKeys = D(1).keys;
for ti = 2:numel(D)
    commonKeys = intersect(commonKeys, D(ti).keys, 'stable');
end

if isempty(commonKeys)
    disp("=== 디버그: 온도별 키 샘플(상위 20개) ===");
    for ti = 1:numel(D)
        fprintf("[%ddegC] keys sample:\n", D(ti).T);
        disp(string(D(ti).keys(1:min(20,numel(D(ti).keys))))');
    end
    error("키 기반 교집합이 비었습니다. extractCellKey 규칙이 파일명 패턴과 맞는지 확인 필요.");
end

nCommon = numel(commonKeys);
fprintf(">> 공통 셀 수(키 기준): %d\n", nCommon);

% 온도별로 공통키 순서로 정렬된 P
P_byT = cell(numel(D),1);
for ti = 1:numel(D)
    [tf, loc] = ismember(commonKeys, D(ti).keys);
    if any(~tf)
        error("내부 오류: 공통키가 특정 온도에서 매칭되지 않았습니다.");
    end
    P_byT{ti} = D(ti).P(loc, :);   % [nCommon x 5]
end

% 레전드용 이름: 키 그대로 사용 (예: "01","02"...)
cellNames = commonKeys;

%% ---------------- capVec 길이 보정 ----------------
if numel(capVec) ~= nCommon
    warning("capVec 길이(%d) != 공통 셀 수(%d). min~max 범위로 자동 재생성합니다.", numel(capVec), nCommon);
    if isempty(capVec)
        capVec = linspace(48, 59, nCommon)';
    else
        capVec = linspace(min(capVec), max(capVec), nCommon)';
    end
else
    capVec = capVec(:);
end

%% ---------------- 컬러맵 (기존 스타일 유지) ----------------
Nmap = 256;
anchors = [0.88 0.16 0.24; 0.83 0.70 0.86; 0.16 0.38 0.92];
x  = [0 0.5 1];
xi = linspace(0,1,Nmap)';
cmap = [interp1(x,anchors(:,1),xi,'pchip'), ...
        interp1(x,anchors(:,2),xi,'pchip'), ...
        interp1(x,anchors(:,3),xi,'pchip')];
cmap = min(max(cmap,0),1);
hsvv = rgb2hsv(cmap);
hsvv(:,2) = max(0.35,hsvv(:,2));
hsvv(:,2) = min(1.0,hsvv(:,2)*1.2);
hsvv(:,3) = max(0.75,hsvv(:,3)*0.95);
cmap = hsv2rgb(hsvv);

capMin = min(capVec);
capMax = max(capVec);
mapColor = @(v) cmap( max(1, min(Nmap, 1 + round((v-capMin)/max(capMax-capMin,eps)*(Nmap-1)))), : );

%% ---------------- 플롯 ----------------
if ~exist(outDir,'dir'), mkdir(outDir); end

xT = tempsC(:)';  % [0 10 20]

for pi = 1:nP
    pname = pNames{pi};

    fig = figure('Color','w','Name',sprintf('%s vs Temp (SOC%d, %s)', pname, socPick, loadPick));
    hold on; grid on;

    for i = 1:nCommon
        col = mapColor(capVec(i));

        y = nan(1,numel(tempsC));
        for ti = 1:numel(tempsC)
            y(ti) = P_byT{ti}(i, pi);
        end

        plot(xT, y, '-o', ...
            'LineWidth', 1.8, ...
            'Color', col, ...
            'MarkerFaceColor', col, ...
            'DisplayName', cellNames{i});
    end

    xlabel('Temperature (°C)');
    if pi <= 3
        ylabel(sprintf('%s (mOhm)', pname));
    else
        ylabel(sprintf('%s (s)', pname));
    end

    title(sprintf('%s | %s | SOC%d | %s', titlePrefix, pname, socPick, loadPick));

    xlim(xlimTemp);
    xticks(tempsC);

    % y축 0부터 시작(원하면 제거 가능)
    allY = [];
    for ti = 1:numel(tempsC)
        allY = [allY; P_byT{ti}(:,pi)]; %#ok<AGROW>
    end
    ymax = max(allY, [], 'omitnan');
    if isfinite(ymax) && ymax > 0
        ylim([0, ymax*1.05]);
    end

    colormap(cmap);
    cb = colorbar('Location','eastoutside');
    cb.Label.String = capLabel;
    clim([capMin capMax]);

    if doLegend
        legend('Location','best', 'Interpreter','none');
    end

    savefig(fig, fullfile(outDir, sprintf('%s%s_vs_Temp.fig', filePrefix, pname)));
    exportgraphics(fig, fullfile(outDir, sprintf('%s%s_vs_Temp.png', filePrefix, pname)), 'Resolution', 200);
end

disp("완료: SOC50 고정, US06, 온도축(0/10/20) 플롯 저장 완료.");

%% ===================== 보조 함수 =====================
function key = extractCellKey(rowName)
% RowNames에서 온도에 무관한 "셀 키"를 뽑는다.
% 당신 파일명(캡처) 기준 최우선: "01_HNE_..." 처럼 시작하는 숫자
% 그 다음: "x05_..." 같은 패턴
% fallback: 원문 정리

    s = char(rowName);

    % 1) "01_..." 형태
    tok = regexp(s, '^(\d+)_', 'tokens', 'once');
    if ~isempty(tok)
        key = tok{1};
        return
    end

    % 2) "x05_..." 형태
    tok = regexp(s, '^x(\d+)', 'tokens', 'once');
    if ~isempty(tok)
        key = tok{1};
        return
    end

    % 3) fallback: 온도 문자열 제거 + 정리
    s = regexprep(s, '(0|10|20|30|35|45)degC', '');
    s = regexprep(s, '\s+', '');
    s = regexprep(s, '_{2,}', '_');
    key = s;
end
