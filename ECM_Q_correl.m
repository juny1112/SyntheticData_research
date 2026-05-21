%% ============================================================
% Capacity vs ECM parameters + capacitances
% - x: capacity (raw or log)
% - y: resistance / tau / capacitance parameters (raw or log)
% - same load -> same color family
% - same load, different param -> different shade
% - NO trendline
% - connect points in capacity order
% - marker unified to circle
%
% Layout of subplot:
%   [ R1   R2   R0 ]
%   [tau1 tau2  -- ]
%   [ C1   C2   -- ]
%
% NOTE:
%   R is in mOhm, tau is in s
%   C1 = tau1 / R1, C2 = tau2 / R2
%   -> unit is effectively kF when using tau[s] / R[mOhm]
%% ============================================================
clear; clc; close all;

%% -----------------------------
% USER OPTION
% ------------------------------
USE_LOG_CAP = false;   % true / false
USE_LOG_RES = true;    % true / false
USE_LOG_TAU = true;    % true / false
USE_LOG_C   = true;    % true / false

LOG_EPS = 1e-12;       % safety for log transform

%% -----------------------------
% Example data
% ------------------------------
cellID = [1 2 3 4 5 7 8 13 14 15 16 18]';

QC40 = [57.49; 57.57; 54.00; 52.22; 53.45; 51.28; 57.91; 56.51; 42.14; 57.27; 57.18; 58.40];

% -------- US06 --------
US06.R0   = [1.12;1.00;1.46;1.42;1.21;1.27;1.00;1.11;2.67;1.11;1.09;1.06];
US06.R1   = [0.18;0.14;0.28;0.23;0.19;0.21;0.15;0.15;0.61;0.15;0.15;0.15];
US06.R2   = [0.94;0.77;1.15;1.01;0.85;0.72;0.81;0.82;2.27;0.81;0.83;0.81];
US06.tau1 = [4.39;3.84;3.86;3.89;3.76;2.96;3.59;4.17;3.70;4.12;4.45;3.80];
US06.tau2 = [48.25;40.13;50.56;43.89;40.42;27.48;42.44;43.68;39.49;43.51;44.34;42.04];

% -------- CITY1 --------
CITY1.R0   = [1.16;1.04;1.54;1.49;1.26;1.34;1.02;1.16;2.78;1.15;1.14;1.10];
CITY1.R1   = [0.44;0.36;0.47;0.47;0.43;0.45;0.17;0.36;0.94;0.39;0.35;0.41];
CITY1.R2   = [1.31;0.88;2.05;2.24;1.64;1.74;0.90;1.15;6.47;1.05;1.32;1.15];
CITY1.tau1 = [11.74;11.24;7.10;8.63;9.01;7.33;4.60;10.79;7.77;11.57;10.73;12.35];
CITY1.tau2 = [154.76;106.02;124.12;147.57;147.35;115.27;49.52;121.91;190.36;140.21;133.47;164.65];

%% -----------------------------
% C1, C2 calculation
% ------------------------------
US06.C1  = US06.tau1 ./ US06.R1;    % kF
US06.C2  = US06.tau2 ./ US06.R2;    % kF

CITY1.C1 = CITY1.tau1 ./ CITY1.R1;  % kF
CITY1.C2 = CITY1.tau2 ./ CITY1.R2;  % kF

%% -----------------------------
% Put loads into struct
% ------------------------------
Loads.US06  = US06;
Loads.CITY1 = CITY1;

loadNames  = fieldnames(Loads);
resParams  = {'R0','R1','R2'};
tauParams  = {'tau1','tau2'};
cParams    = {'C1','C2'};

%% -----------------------------
% Base colors for each load
% ------------------------------
baseColors = [
    0.85 0.25 0.25   % US06 : red family
    0.20 0.45 0.85   % CITY1: blue family
    0.20 0.65 0.35   % spare
    0.55 0.35 0.80   % spare
    0.85 0.55 0.15   % spare
];

shadeRes = [1.00, 0.75, 0.50];   % R0, R1, R2용
shadeTau = [0.85, 0.45];         % tau1, tau2용
shadeC   = [0.85, 0.45];         % C1, C2용

%% -----------------------------
% Transform x data
% ------------------------------
xCap = applyTransform(QC40, USE_LOG_CAP, LOG_EPS);
xLabel = makeAxisLabel('Capacity, Q_{C/40} (Ah)', USE_LOG_CAP);

%% ============================================================
% Figure 1: Capacity vs Resistance parameters
%% ============================================================
figure('Color','w','Position',[100 100 980 680]);
hold on; grid on; box on;

legendHandles = [];
legendTexts   = {};

% 원하는 순서대로 저항 표시
resOrder = {'R1','R2','R0'};

for iL = 1:numel(loadNames)
    L = loadNames{iL};
    baseC = baseColors(iL,:);

    for ip = 1:numel(resOrder)
        p = resOrder{ip};

        yRaw = Loads.(L).(p);
        y = applyTransform(yRaw, USE_LOG_RES, LOG_EPS);

        idxShade = find(strcmp({'R0','R1','R2'}, p));
        c = mixWithWhite(baseC, shadeRes(idxShade));

        [xSorted, idxSort] = sort(xCap);
        ySorted = y(idxSort);

        plot(xSorted, ySorted, '-', 'Color', c, 'LineWidth', 1.2, ...
            'HandleVisibility','off');

        h = scatter(xCap, y, 72, ...
            'Marker', 'o', ...
            'MarkerFaceColor', c, ...
            'MarkerEdgeColor', max(c*0.65,0), ...
            'LineWidth', 1.0);

        legendHandles(end+1) = h; %#ok<SAGROW>
        legendTexts{end+1}   = sprintf('%s-%s', L, p); %#ok<SAGROW>
    end
end

xlabel(xLabel, 'FontWeight','bold');
ylabel(makeAxisLabel('Resistance parameter (m\Omega)', USE_LOG_RES), 'FontWeight','bold');
title(makeTitleText('Capacity vs Resistance parameters', USE_LOG_CAP, USE_LOG_RES, false, false), ...
    'FontWeight','bold');
legend(legendHandles, legendTexts, 'Location','eastoutside');
set(gca, 'FontSize', 11);

%% ============================================================
% Figure 2: Capacity vs Time constants
%% ============================================================
figure('Color','w','Position',[130 110 980 680]);
hold on; grid on; box on;

legendHandles = [];
legendTexts   = {};

for iL = 1:numel(loadNames)
    L = loadNames{iL};
    baseC = baseColors(iL,:);

    for ip = 1:numel(tauParams)
        p = tauParams{ip};

        yRaw = Loads.(L).(p);
        y = applyTransform(yRaw, USE_LOG_TAU, LOG_EPS);

        idxShade = find(strcmp(tauParams,p));
        c = mixWithWhite(baseC, shadeTau(idxShade));

        [xSorted, idxSort] = sort(xCap);
        ySorted = y(idxSort);

        plot(xSorted, ySorted, '-', 'Color', c, 'LineWidth', 1.2, ...
            'HandleVisibility','off');

        h = scatter(xCap, y, 72, ...
            'Marker', 'o', ...
            'MarkerFaceColor', c, ...
            'MarkerEdgeColor', max(c*0.65,0), ...
            'LineWidth', 1.0);

        legendHandles(end+1) = h; %#ok<SAGROW>
        legendTexts{end+1}   = sprintf('%s-%s', L, p); %#ok<SAGROW>
    end
end

xlabel(xLabel, 'FontWeight','bold');
ylabel(makeAxisLabel('Time constant (s)', USE_LOG_TAU), 'FontWeight','bold');
title(makeTitleText('Capacity vs Time constants', USE_LOG_CAP, false, USE_LOG_TAU, false), ...
    'FontWeight','bold');
legend(legendHandles, legendTexts, 'Location','eastoutside');
set(gca, 'FontSize', 11);

%% ============================================================
% Figure 3: Capacity vs Capacitances
%% ============================================================
figure('Color','w','Position',[160 120 980 680]);
hold on; grid on; box on;

legendHandles = [];
legendTexts   = {};

for iL = 1:numel(loadNames)
    L = loadNames{iL};
    baseC = baseColors(iL,:);

    for ip = 1:numel(cParams)
        p = cParams{ip};

        yRaw = Loads.(L).(p);   % kF
        y = applyTransform(yRaw, USE_LOG_C, LOG_EPS);

        idxShade = find(strcmp(cParams,p));
        c = mixWithWhite(baseC, shadeC(idxShade));

        [xSorted, idxSort] = sort(xCap);
        ySorted = y(idxSort);

        plot(xSorted, ySorted, '-', 'Color', c, 'LineWidth', 1.2, ...
            'HandleVisibility','off');

        h = scatter(xCap, y, 72, ...
            'Marker', 'o', ...
            'MarkerFaceColor', c, ...
            'MarkerEdgeColor', max(c*0.65,0), ...
            'LineWidth', 1.0);

        legendHandles(end+1) = h; %#ok<SAGROW>
        legendTexts{end+1}   = sprintf('%s-%s', L, p); %#ok<SAGROW>
    end
end

xlabel(xLabel, 'FontWeight','bold');
ylabel(makeAxisLabel('Capacitance (kF)', USE_LOG_C), 'FontWeight','bold');
title(makeTitleText('Capacity vs Capacitances', USE_LOG_CAP, false, false, USE_LOG_C), ...
    'FontWeight','bold');
legend(legendHandles, legendTexts, 'Location','eastoutside');
set(gca, 'FontSize', 11);

%% ============================================================
% Figure 4: tiled layout (custom fixed positions)
% Layout:
%   R1   R2   R0
%   tau1 tau2 --
%   C1   C2   --
%% ============================================================
figure('Color','w','Position',[150 100 1200 900]);
tl = tiledlayout(3,3,'TileSpacing','compact','Padding','compact');

plotOrder = {'R1','R2','R0','tau1','tau2','C1','C2'};
tilePos   = [  1,   2,   3,    4,    5,    7,   8  ];
% 3x3 tile numbering:
% [1 2 3
%  4 5 6
%  7 8 9]

for ip = 1:numel(plotOrder)
    p = plotOrder{ip};
    ax = nexttile(tilePos(ip)); hold(ax,'on'); grid(ax,'on'); box(ax,'on');

    isRes = ismember(p, resParams);
    isTau = ismember(p, tauParams);
    isC   = ismember(p, cParams);

    legendHandles = [];
    legendTexts   = {};

    for iL = 1:numel(loadNames)
        L = loadNames{iL};
        baseC = baseColors(iL,:);

        yRaw = Loads.(L).(p);

        if isRes
            y = applyTransform(yRaw, USE_LOG_RES, LOG_EPS);
            idxShade = find(strcmp({'R0','R1','R2'}, p));
            c = mixWithWhite(baseC, shadeRes(idxShade));
        elseif isTau
            y = applyTransform(yRaw, USE_LOG_TAU, LOG_EPS);
            idxShade = find(strcmp(tauParams,p));
            c = mixWithWhite(baseC, shadeTau(idxShade));
        else
            y = applyTransform(yRaw, USE_LOG_C, LOG_EPS);
            idxShade = find(strcmp(cParams,p));
            c = mixWithWhite(baseC, shadeC(idxShade));
        end

        [xSorted, idxSort] = sort(xCap);
        ySorted = y(idxSort);

        plot(ax, xSorted, ySorted, '-', 'Color', c, 'LineWidth', 1.2, ...
            'HandleVisibility','off');

        h = scatter(ax, xCap, y, 74, ...
            'Marker', 'o', ...
            'MarkerFaceColor', c, ...
            'MarkerEdgeColor', max(c*0.65,0), ...
            'LineWidth', 1.0);

        legendHandles(end+1) = h; %#ok<SAGROW>
        legendTexts{end+1}   = L; %#ok<SAGROW>
    end

    xlabel(ax, xLabel);

    if isRes
        ylabel(ax, makeAxisLabel(sprintf('%s (m\\Omega)', p), USE_LOG_RES), 'Interpreter','tex');
    elseif isTau
        ylabel(ax, makeAxisLabel(sprintf('%s (s)', p), USE_LOG_TAU), 'Interpreter','tex');
    else
        ylabel(ax, makeAxisLabel(sprintf('%s (kF)', p), USE_LOG_C), 'Interpreter','tex');
    end

    title(ax, sprintf('%s vs Capacity', p), 'Interpreter','none');

    if ip == 1
        legend(ax, legendHandles, legendTexts, 'Location','best');
    end
end

% 비워둘 타일 6, 9
axBlank1 = nexttile(6); axis(axBlank1,'off');
axBlank2 = nexttile(9); axis(axBlank2,'off');

sgtitle(tl, makeTitleText('ECM parameters vs Capacity', USE_LOG_CAP, USE_LOG_RES, USE_LOG_TAU, USE_LOG_C), ...
    'FontWeight','bold');

%% ============================================================
% Optional: print C1/C2 values
%% ============================================================
disp('--- US06 ---');
disp(table(cellID, QC40, US06.R1(:), US06.tau1(:), US06.C1(:), US06.R2(:), US06.tau2(:), US06.C2(:), ...
    'VariableNames', {'CellID','QC40','R1_mOhm','tau1_s','C1_kF','R2_mOhm','tau2_s','C2_kF'}));

disp('--- CITY1 ---');
disp(table(cellID, QC40, CITY1.R1(:), CITY1.tau1(:), CITY1.C1(:), CITY1.R2(:), CITY1.tau2(:), CITY1.C2(:), ...
    'VariableNames', {'CellID','QC40','R1_mOhm','tau1_s','C1_kF','R2_mOhm','tau2_s','C2_kF'}));

%% ============================================================
% Local functions
%% ============================================================
function y = applyTransform(x, useLog, log_eps)
    x = x(:);
    if useLog
        if any(x <= 0)
            error('Log transform requested, but data contains zero or negative values.');
        end
        y = log10(x + log_eps);
    else
        y = x;
    end
end

function c = mixWithWhite(baseColor, scaleVal)
    % scaleVal=1 -> original color
    % scaleVal small -> lighter color
    c = 1 - (1 - baseColor) * scaleVal;
end

function txt = makeAxisLabel(baseTxt, useLog)
    if useLog
        txt = ['log_{10}(' baseTxt ')'];
    else
        txt = baseTxt;
    end
end

function txt = makeTitleText(baseTxt, useLogCap, useLogRes, useLogTau, useLogC)
    capTxt = ternary(useLogCap, 'log-capacity', 'raw-capacity');
    resTxt = ternary(useLogRes, 'log-resistance', 'raw-resistance');
    tauTxt = ternary(useLogTau, 'log-tau', 'raw-tau');
    cTxt   = ternary(useLogC,   'log-C', 'raw-C');
    txt = sprintf('%s [%s | %s | %s | %s]', baseTxt, capTxt, resTxt, tauTxt, cTxt);
end

function out = ternary(cond, a, b)
    if cond
        out = a;
    else
        out = b;
    end
end