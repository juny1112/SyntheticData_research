% ======================================================================
% Synthetic_DRT_SOH_SaveCSV_NoFitting.m
% ----------------------------------------------------------------------
% Purpose
%   1) Generate aging-dependent synthetic DRT parameters by degradation
%      index k in [0, 1].
%   2) Convert DRT to nRC truth, simulate voltage under driving profiles.
%   3) Add Markov noise to make synthetic voltage data.
%   4) Save each synthetic voltage data as a CSV file.
%
%
% Concept
%   - Synthetic CSV output contains only time, current, and synthetic voltage.
%   - No 2RC fitting is performed in this version.
%
% Required helper files in the same folder/path
%   - DRT_mu_sigma.m
%   - DRT_Rtau.m
%   - MarkovNoise_idx.m
%
% Notes
%   - cfg.fresh and cfg.aged are generated from existing fitted-cell reference
%     data only to define the synthetic aging trajectory.
%   - This script does not fit generated synthetic voltage data.
% ======================================================================

clear; clc; close all;

%% ------------------------ User configuration -------------------------
cfg = struct();

% Synthetic voltage CSV output. 2RC-generation version.
% One CSV is saved per generated cell and driving-load pair.
% File name format: CellID_LoadID.csv, e.g., 001_01.csv.
% Columns: time, current, voltage.
cfg.synthetic_csv_dir = 'G:\공유 드라이브\BSG_SYN\SYN_2RC';
cfg.save_synthetic_voltage_csv = true;
if cfg.save_synthetic_voltage_csv && ~exist(cfg.synthetic_csv_dir, 'dir')
    mkdir(cfg.synthetic_csv_dir);
end

% Synthetic sample count. Each sample has an independent k value.
cfg.nSyntheticSamples = 100;

% Random seed configuration
% markov : base seed for Markov voltage noise, applied per final voltage sample.
cfg.seed.markov = 1;

% 2RC voltage generation mode
% DRT-generated true ECM [R0,R1,R2,tau1,tau2] is used directly.
% No nRC DRT discretization is used for voltage generation in this version.

% ----------------------------------------------------------------------
% Fresh / aged reference from real-cell fitting results
% ----------------------------------------------------------------------
% Capacity list must correspond to the ascending cell-id order parsed from
% the fitting-result row names / file names. Example: 01,02,04,...,18.
cfg.capacity.Q_ref_100 = 58.50;
cfg.capacity.QC40_user = [57.49;57.57;54.00;52.22;53.45;51.28;57.91;56.51;42.14;57.27;57.18;58.40];

% Optional manual cell ID order. Leave [] to auto-detect IDs from file names
% and sort numerically. If auto-detection is ambiguous, fill this explicitly,
% e.g., cfg.capacity.cell_ids = [1 2 4 6 7 8 10 12 14 15 17 18];
cfg.capacity.cell_ids = [];

% Load references from HM_SIM8LOAD_600s_2RC_newprotocol_Customset.m output.
% Preferred: 2RC_results_600s.mat containing Tbl_ECM_mean.
% Alternative: True_ECM_Labels.csv.
cfg.reference.use_fitted_cell_references = true;
cfg.reference.results_mat = 'G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\20degC\2RC_fitting_600s\2RC_results_600s.mat';
cfg.reference.true_ecm_csv = fullfile(pwd, 'True_ECM_Labels.csv');

% Which SOC columns from Tbl_ECM_mean should define the reference ECM?
%   'all'   : mean across all SOC*_R0/R1/R2/tau1/tau2 columns.
%   numeric : use one SOC only, e.g., 70.
cfg.reference.soc_for_reference = 70;

% Driving profiles folder.
% Each file must have column 1 = time [s], column 2 = current [A].
cfg.driving_folder = 'G:\공유 드라이브\Battery Software Group (2025)\Protocols\2026 Driving_Loads_20\after_scaling';

% File extensions to search in the folder.
cfg.driving_exts = {'*.xlsx', '*.xls', '*.csv'};

% This will be filled automatically from cfg.driving_folder.
cfg.driving_files = {};

% Markov noise
cfg.noise.epsilon_percent_span = 1;
cfg.noise.initial_state = 51;
cfg.noise.sigma = 5;
cfg.noise.nSeeds = 1;                % seed 1..nSeeds. Seed 0 = non-noise.

cfg.noise.includeNonNoise = false; % True: non-noise 생성

% 2RC fitting is intentionally disabled in this CSV-export version.
% The script stops at synthetic voltage generation and CSV saving.

%% ------------------------ Initialization -----------------------------

% Fresh/aged reference must come from real fitted cell data.
% Dummy fallback references are not allowed in the final dataset generation.
if ~cfg.reference.use_fitted_cell_references
    error(['Final synthetic dataset generation requires real fitted cell references. ', ...
           'Set cfg.reference.use_fitted_cell_references = true and provide a valid results MAT file.']);
end

% Update cfg.fresh/cfg.aged/cfg.SOH_aged from measured-capacity cell IDs
% and fitted 2RC ECM results.
% The bimodal DRT reference is constructed by treating fitted
% [R0,R1,R2,tau1,tau2] as the DRT-generated true ECM:
%   DRT.R0       = R0
%   DRT.A_tot    = R1 + R2
%   DRT.w        = [R1; R2]
%   DRT.tau_mode = [tau1; tau2]
[cfg.fresh, cfg.aged, cfg.SOH_aged, cfg.reference_info] = ...
    loadFreshAgedDRTReferencesFromFitting(cfg);

% Collect driving profile files from the folder.
cfg.driving_files = collectDrivingFilesFromFolder( ...
    cfg.driving_folder, cfg.driving_exts);

if isempty(cfg.driving_files)
    error('No driving profile files were found in cfg.driving_folder.');
end

fprintf('Found %d driving profile files:\n', numel(cfg.driving_files));
for ii = 1:numel(cfg.driving_files)
    fprintf('  [%02d] %s\n', ii, cfg.driving_files{ii});
end

% Generate uniformly spaced degradation indices without exact endpoints.
% k = 0 and k = 1 are used only as fresh/aged anchors.
kList = ((1:cfg.nSyntheticSamples).' - 0.5) / cfg.nSyntheticSamples;

%% ------------------------ Main generation loop -----------------------
savedCsvCount = 0;

for sampleIdx = 1:cfg.nSyntheticSamples
    k = kList(sampleIdx);
    drtParam = interpolate_DRT_param(cfg.fresh, cfg.aged, k);

    % 2RC truth from aging-dependent DRT for voltage generation.
    % IMPORTANT: the DRT-generated TRUE ECM values are passed directly to
    % RC_model_2. Therefore the saved voltage CSVs are generated from the
    % same 2RC structure used later for 2RC fitting.
    X_true_2RC = drtParam_to_trueECM_2RC(drtParam);

    for fileIdx = 1:numel(cfg.driving_files)
        [t_vec, I_vec, base_name] = readDrivingProfile(cfg.driving_files{fileIdx}, fileIdx);
        OCV = zeros(size(t_vec));
        V_true = RC_model_2(X_true_2RC, t_vec, I_vec, OCV);

        % Seed 0 means no noise; seeds 1..nSeeds use Markov noise.
        if cfg.noise.includeNonNoise
            seedList = 0:cfg.noise.nSeeds;
        else
            seedList = 1:cfg.noise.nSeeds;
        end

        if cfg.save_synthetic_voltage_csv && numel(seedList) ~= 1
            error(['cfg.save_synthetic_voltage_csv uses CellID_LoadID.csv file names, ', ...
                   'so exactly one voltage version per cell-load pair is required. ', ...
                   'Set cfg.noise.nSeeds = 1 and cfg.noise.includeNonNoise = false, ', ...
                   'or extend the file name to include seed information.']);
        end

        for seed = seedList
            if seed == 0
                V_syn = V_true;
            else
                % Markov noise is assigned once per final voltage sample.
                % Example: nDRT = 50, nLoad = 20, nSeeds = 1
                %          -> markovSeed = cfg.seed.markov ... cfg.seed.markov+999.
                linearVoltageIdx = ((sampleIdx - 1) * numel(cfg.driving_files) + (fileIdx - 1)) ...
                    * cfg.noise.nSeeds + seed;
                rng(cfg.seed.markov + linearVoltageIdx - 1, 'twister');
                V_syn = MarkovNoise_idx(V_true, cfg.noise.epsilon_percent_span, ...
                    cfg.noise.initial_state, cfg.noise.sigma);
            end

            if cfg.save_synthetic_voltage_csv
                cellID = sprintf('%03d', sampleIdx);
                loadID = parseDrivingLoadID(base_name, fileIdx);
                synFileName = sprintf('%s_%s.csv', cellID, loadID);
                SynVoltageTable = table(t_vec(:), I_vec(:), V_syn(:), ...
                    'VariableNames', {'time','current','voltage'});
                writetable(SynVoltageTable, fullfile(cfg.synthetic_csv_dir, synFileName));
                savedCsvCount = savedCsvCount + 1;
            end
        end
    end

    if mod(sampleIdx, 10) == 0
        fprintf('Completed %d / %d synthetic DRT samples\n', sampleIdx, cfg.nSyntheticSamples);
    end
end

fprintf('\nDone. Saved synthetic CSV files = %d\n', savedCsvCount);
fprintf('Output folder: %s\n', cfg.synthetic_csv_dir);

%% =====================================================================
% Local functions
% =====================================================================
function [freshDRT, agedDRT, SOH_aged, info] = loadFreshAgedDRTReferencesFromFitting(cfg)
    Q = cfg.capacity.QC40_user(:);
    SOH_vec = 100 * Q / cfg.capacity.Q_ref_100;

    % Load fitted ECM table.
    [T, sourceName] = loadFittedECMTable(cfg.reference.results_mat, cfg.reference.true_ecm_csv);
    cellNames = getTableCellNames(T);
    cellIDs = parseCellIDs(cellNames);

    if isempty(cfg.capacity.cell_ids)
        if any(isnan(cellIDs))
            error(['Could not parse numeric cell IDs from fitting result row names/file names. ', ...
                   'Set cfg.capacity.cell_ids explicitly.']);
        end
        [cellIDs_sorted, order] = sort(cellIDs(:));
        T = T(order,:);
        cellNames = cellNames(order);
        cellIDs = cellIDs_sorted;
    else
        cellIDs_cfg = cfg.capacity.cell_ids(:);
        if numel(cellIDs_cfg) ~= numel(Q)
            error('cfg.capacity.cell_ids length must match cfg.capacity.QC40_user length.');
        end
        if any(isnan(cellIDs))
            % If row names cannot be parsed, assume table rows already follow cfg.capacity.cell_ids.
            if height(T) ~= numel(cellIDs_cfg)
                error('Table row count does not match cfg.capacity.cell_ids length.');
            end
            cellIDs = cellIDs_cfg;
        end
        [tf, loc] = ismember(cellIDs_cfg, cellIDs);
        if ~all(tf)
            error('Some cfg.capacity.cell_ids were not found in fitting results: %s', mat2str(cellIDs_cfg(~tf).'));
        end
        T = T(loc,:);
        cellNames = cellNames(loc);
        cellIDs = cellIDs_cfg;
    end

    if height(T) ~= numel(Q)
        error(['Capacity vector length (%d) and fitting result cell count (%d) differ. ', ...
               'Check cfg.capacity.cell_ids or the fitting result table.'], numel(Q), height(T));
    end

    [~, freshIdx] = max(Q);
    [~, agedIdx]  = min(Q);

    freshECM = extractReferenceECM(T(freshIdx,:), cfg.reference.soc_for_reference);
    agedECM  = extractReferenceECM(T(agedIdx,:),  cfg.reference.soc_for_reference);

    freshDRT = trueECM2RC_to_DRTParam(freshECM);
    agedDRT  = trueECM2RC_to_DRTParam(agedECM);
    SOH_aged = SOH_vec(agedIdx);

    info = struct();
    info.source = sourceName;
    info.cell_names = cellNames;
    info.cell_ids = cellIDs;
    info.QC40_user = Q;
    info.SOH_vec = SOH_vec;
    info.fresh_idx = freshIdx;
    info.aged_idx = agedIdx;
    info.fresh_cell_name = cellNames{freshIdx};
    info.aged_cell_name = cellNames{agedIdx};
    info.fresh_cell_id = cellIDs(freshIdx);
    info.aged_cell_id = cellIDs(agedIdx);
    info.fresh_capacity = Q(freshIdx);
    info.aged_capacity = Q(agedIdx);
    info.SOH_fresh_measured = SOH_vec(freshIdx);
    info.SOH_aged = SOH_aged;
    info.fresh_trueECM = freshECM;
    info.aged_trueECM = agedECM;

    fprintf('\nReference cells from %s\n', sourceName);
    fprintf('  Fresh ref: %s | ID=%g | Q=%.2f Ah | measured SOH=%.2f%% | label anchor=100%%\n', ...
        info.fresh_cell_name, info.fresh_cell_id, info.fresh_capacity, info.SOH_fresh_measured);
    fprintf('  Aged  ref: %s | ID=%g | Q=%.2f Ah | SOH=%.2f%%\n\n', ...
        info.aged_cell_name, info.aged_cell_id, info.aged_capacity, info.SOH_aged);
end

function [T, sourceName] = loadFittedECMTable(resultsMat, trueEcmCsv)
    if isfile(resultsMat)
        S = load(resultsMat);
        if isfield(S, 'Tbl_ECM_mean')
            T = S.Tbl_ECM_mean;
            sourceName = resultsMat;
            return
        elseif isfield(S, 'Tbl_Load_ECM')
            % Build an average table across load-specific ECM tables.
            flds = fieldnames(S.Tbl_Load_ECM);
            if isempty(flds)
                error('Tbl_Load_ECM exists but has no load tables.');
            end
            T = averageLoadECMTables(S.Tbl_Load_ECM);
            sourceName = [resultsMat '::Tbl_Load_ECM averaged'];
            return
        end
    end

    if isfile(trueEcmCsv)
        Tall = readtable(trueEcmCsv);
        T = averageTrueECMLabelsCsv(Tall);
        sourceName = trueEcmCsv;
        return
    end

    error(['Could not find fitted ECM references. Set cfg.reference.results_mat to 2RC_results_600s.mat ', ...
           'or cfg.reference.true_ecm_csv to True_ECM_Labels.csv.']);
end

function names = getTableCellNames(T)
    if ~isempty(T.Properties.RowNames)
        names = T.Properties.RowNames(:);
    elseif ismember('Cell_Name', T.Properties.VariableNames)
        names = cellstr(string(T.Cell_Name));
    else
        names = cellstr("cell_" + string((1:height(T)).'));
    end
end

function ids = parseCellIDs(names)
    ids = nan(numel(names),1);
    for i = 1:numel(names)
        s = char(names{i});
        tok = regexp(s, '^\D*(\d+)', 'tokens', 'once');
        if isempty(tok)
            tok = regexp(s, '(\d+)', 'tokens', 'once');
        end
        if ~isempty(tok)
            ids(i) = str2double(tok{1});
        end
    end
end

function p = trueECM2RC_to_DRTParam(ecm)
    ecm = ecm(:);
    if numel(ecm) ~= 5 || any(~isfinite(ecm))
        error('Reference ECM must be finite [R0 R1 R2 tau1 tau2].');
    end
    R0 = ecm(1); R = ecm(2:3); tau = ecm(4:5);
    if R0 < 0 || any(R < 0) || any(tau <= 0)
        error('Invalid reference ECM values: R0/R must be nonnegative and tau must be positive.');
    end
    [tauSorted, idx] = sort(tau, 'ascend');
    R = R(idx);
    p.R0 = R0;
    p.A_tot = sum(R);
    if p.A_tot <= 0
        error('Reference ECM has zero total polarization resistance.');
    end
    p.w = R(:);                 % DRT_Rtau normalizes w internally through area allocation.
    p.tau_mode = tauSorted(:);
end

function ecm = extractReferenceECM(Trow, socChoice)
    v = Trow.Properties.VariableNames;

    if isnumeric(socChoice)
        pref = sprintf('SOC%d_', socChoice);
        ecm = [getVarValue(Trow, [pref 'R0_mOhm'])/1000, ...
               getVarValue(Trow, [pref 'R1_mOhm'])/1000, ...
               getVarValue(Trow, [pref 'R2_mOhm'])/1000, ...
               getVarValue(Trow, [pref 'tau1']), ...
               getVarValue(Trow, [pref 'tau2'])];
    else
        % Average all SOC-specific columns for each parameter.
        R0cols = v(contains(v, 'R0_mOhm'));
        R1cols = v(contains(v, 'R1_mOhm'));
        R2cols = v(contains(v, 'R2_mOhm'));
        t1cols = v(contains(v, 'tau1'));
        t2cols = v(contains(v, 'tau2'));
        ecm = [meanCols(Trow, R0cols)/1000, meanCols(Trow, R1cols)/1000, meanCols(Trow, R2cols)/1000, ...
               meanCols(Trow, t1cols), meanCols(Trow, t2cols)];
    end

    ecm = ecm(:);
end

function y = getVarValue(Trow, name)
    if ~ismember(name, Trow.Properties.VariableNames)
        error('Missing column in fitting result table: %s', name);
    end
    y = Trow{1, name};
end

function y = meanCols(Trow, cols)
    if isempty(cols)
        error('No matching ECM columns found for averaging.');
    end
    vals = Trow{1, cols};
    y = mean(vals, 'omitnan');
end

function Tavg = averageLoadECMTables(Tbl_Load_ECM)
    flds = fieldnames(Tbl_Load_ECM);
    T0 = Tbl_Load_ECM.(flds{1});
    rowNames = T0.Properties.RowNames;
    varNames = T0.Properties.VariableNames;
    accum = zeros(height(T0), width(T0));
    count = zeros(height(T0), width(T0));

    for i = 1:numel(flds)
        Ti = Tbl_Load_ECM.(flds{i});
        Xi = Ti{:, varNames};
        mask = isfinite(Xi);
        Xi(~mask) = 0;
        accum = accum + Xi;
        count = count + mask;
    end
    Xavg = accum ./ max(count, 1);
    Xavg(count == 0) = NaN;
    Tavg = array2table(Xavg, 'VariableNames', varNames, 'RowNames', rowNames);
end

function Tavg = averageTrueECMLabelsCsv(Tall)
    if ~ismember('Cell_Name', Tall.Properties.VariableNames)
        error('True_ECM_Labels.csv must contain Cell_Name column.');
    end
    names = unique(string(Tall.Cell_Name), 'stable');
    rows = cell(numel(names), 1);
    mat = [];
    for i = 1:numel(names)
        idx = string(Tall.Cell_Name) == names(i);
        Ti = Tall(idx,:);
        % Average across load rows. Keep SOC-specific ECM columns.
        v = Ti.Properties.VariableNames;
        keep = v(contains(v, 'R0_mOhm') | contains(v, 'R1_mOhm') | contains(v, 'R2_mOhm') | ...
                 contains(v, 'tau1') | contains(v, 'tau2'));
        mat(i,:) = mean(Ti{:, keep}, 1, 'omitnan'); %#ok<AGROW>
        rows{i} = char(names(i));
    end
    Tavg = array2table(mat, 'VariableNames', keep, 'RowNames', rows);
end

function p = interpolate_DRT_param(fresh, aged, k)
    % Linear interpolation: R0, total area, mode area ratio.
    p.R0    = (1-k)*fresh.R0    + k*aged.R0;
    p.A_tot = (1-k)*fresh.A_tot + k*aged.A_tot;
    p.w     = (1-k)*fresh.w(:)  + k*aged.w(:);

    % Log interpolation: tau mode.
    logTau = (1-k)*log10(fresh.tau_mode(:)) + k*log10(aged.tau_mode(:));
    p.tau_mode = 10.^logTau;
end

function [X, meta] = make_nRC_from_DRT(param, sigma10, nRC, tau_min, tau_max)
    [mu10, ~, ~, ~, ~, ~, ~] = DRT_mu_sigma(param.tau_mode, sigma10);
    [theta, r_mode, r_theta, dth, R, tau, g_tau, A_modes, w_used] = ...
        DRT_Rtau(nRC, tau_min, tau_max, mu10, sigma10, param.A_tot, param.w(:));
    X = [param.R0; R(:); tau(:)];
    meta = struct('theta', theta, 'r_mode', r_mode, 'r_theta', r_theta, ...
                  'dth', dth, 'R', R, 'tau', tau, 'g_tau', g_tau, ...
                  'A_modes', A_modes, 'w_used', w_used);
end


function loadID = parseDrivingLoadID(base_name, fileIdx)
    % Use the leading number in the driving-load file name as load ID.
    % Examples: '01_...' -> '01', '20_...' -> '20'.
    tok = regexp(char(base_name), '^(\d+)', 'tokens', 'once');

    if isempty(tok)
        warning('Could not parse leading load ID from %s. Falling back to file index %02d.', ...
            char(base_name), fileIdx);
        loadID = sprintf('%02d', fileIdx);
    else
        loadNum = str2double(tok{1});
        if isnan(loadNum)
            warning('Could not convert leading load ID from %s. Falling back to file index %02d.', ...
                char(base_name), fileIdx);
            loadID = sprintf('%02d', fileIdx);
        else
            loadID = sprintf('%02d', loadNum);
        end
    end
end

function [t_vec, I_vec, base_name] = readDrivingProfile(item, fileIdx)
    if isstruct(item)
        t_vec = item.t(:);
        I_vec = item.I(:);

        if isfield(item, 'name')
            base_name = char(item.name);
        else
            base_name = sprintf('load%d', fileIdx);
        end
    else
        tbl = readtable(item, 'VariableNamingRule', 'preserve');
        t_vec = tbl{:,1};
        I_vec = tbl{:,2};
        [~, name, ~] = fileparts(item);
        base_name = char(name);
    end

    t_vec = t_vec(:);
    I_vec = I_vec(:);

    if numel(t_vec) ~= numel(I_vec)
        error('Time and current length mismatch in %s.', base_name);
    end

    if any(~isfinite(t_vec)) || any(~isfinite(I_vec))
        error('NaN or Inf exists in driving profile: %s.', base_name);
    end

    if any(diff(t_vec) <= 0)
        error('Time vector must be strictly increasing in %s.', base_name);
    end
end

function files = collectDrivingFilesFromFolder(folderPath, exts)
    if ~isfolder(folderPath)
        error('Driving profile folder does not exist: %s', folderPath);
    end

    files = {};

    for i = 1:numel(exts)
        listing = dir(fullfile(folderPath, exts{i}));

        for j = 1:numel(listing)
            if listing(j).isdir
                continue;
            end

            fname = listing(j).name;

            % Skip temporary Excel files.
            if startsWith(fname, '~$')
                continue;
            end

            files{end+1,1} = fullfile(listing(j).folder, fname); %#ok<AGROW>
        end
    end

    files = sort(files);
end
