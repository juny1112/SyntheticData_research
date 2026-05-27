%% ========================================================================
%  Synthetic dataset visualization
%  x-axis : SOH_DRT
%  y-axis : fitted ECM parameters / RMSE
%  group  : driving load
% ========================================================================

% If Dataset is not already in workspace, load it.
load('G:\공유 드라이브\Battery Software Group (2025)\Members\김주은\PINN\합성데이터\Synthetic_DRT_SOH_Dataset_DRTLabel_CellRef.mat');

if ~exist('Dataset', 'var')
    error('Dataset variable does not exist. Load the generated MAT file first.');
end

% ------------------------------------------------------------------------
% Select variables to plot
% ------------------------------------------------------------------------
xVar = 'SOH_DRT';

plotVars = { ...
    'R1_hat',   'tau1_hat'; ...
    'R2_hat',   'tau2_hat'; ...
    'R0_hat',   'RMSE'};

plotTitles = { ...
    'R_1 vs SOH',      '\tau_1 vs SOH'; ...
    'R_2 vs SOH',      '\tau_2 vs SOH'; ...
    'R_0 vs SOH',      'RMSE vs SOH'};

yLabels = { ...
    'R_1 [\Omega]',    '\tau_1 [s]'; ...
    'R_2 [\Omega]',    '\tau_2 [s]'; ...
    'R_0 [\Omega]',    'RMSE [V]'};

% ------------------------------------------------------------------------
% Load grouping
% ------------------------------------------------------------------------
loadNames = string(Dataset.load_name);
uniqueLoads = unique(loadNames, 'stable');
nLoads = numel(uniqueLoads);

% ------------------------------------------------------------------------
% Plot
% ------------------------------------------------------------------------
figure('Color', 'w', 'Position', [100 100 1200 900]);

for r = 1:3
    for c = 1:2
        subplotIdx = (r-1)*2 + c;
        subplot(3, 2, subplotIdx);
        hold on; box on; grid on;

        yVar = plotVars{r,c};

        for i = 1:nLoads
            idx = loadNames == uniqueLoads(i);

            scatter(Dataset.(xVar)(idx), Dataset.(yVar)(idx), ...
                18, 'filled', ...   
                'DisplayName', uniqueLoads(i));
        end

        xlabel('SOH_{dummy} [%]');
        ylabel(yLabels{r,c});
        title(plotTitles{r,c});

        set(gca, 'FontSize', 11);
    end
end

% One shared legend
lgd = legend(uniqueLoads, 'Interpreter', 'none', ...
    'Location', 'eastoutside');
lgd.Layout.Tile = 'east';

sgtitle('Synthetic Dataset: SOH_{DRT} vs Fitted ECM Parameters', ...
    'FontSize', 15, 'FontWeight', 'bold');