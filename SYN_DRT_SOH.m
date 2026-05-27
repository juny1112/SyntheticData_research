% ======================================================================
% Synthetic_DRT_SOH_Dataset_DRTLabel_CellRef.m
% ----------------------------------------------------------------------
% Purpose
%   1) Generate aging-dependent synthetic DRT parameters by degradation
%      index k in [0, 1].
%   2) Convert DRT to nRC truth, simulate voltage under driving profiles.
%   3) Add Markov noise to make synthetic voltage data.
%   4) Fit each synthetic voltage with a 2RC ECM.
%   5) Generate SOH dummy label from the DRT-generated TRUE ECM state,
%      not from the fitted ECM parameters.
%
% Concept
%   - Label source: latent DRT / true ECM generated before voltage/noise/fit.
%   - ML/DL input: fitted 2RC ECM parameters from noisy synthetic voltage.
%   - This avoids leakage where the target SOH is a deterministic function of
%     the same fitted ECM parameters used as model input.
%
% Required helper files in the same folder/path
%   - DRT_mu_sigma.m
%   - DRT_Rtau.m
%   - RC_model_2.m
%   - MarkovNoise_idx.m
%
% Notes
%   - cfg.fresh and cfg.aged are automatically generated from the fitted
%     ECM parameters of the largest-capacity and smallest-capacity cells.
%   - SOH labels are synthetic/dummy labels for ML/DL data generation, not
%     directly measured SOH unless cfg.SOH_aged is assigned from measurement.
% ======================================================================

clear; clc; close all;

%% ------------------------ User configuration -------------------------
cfg = struct();

% Output
cfg.out_dir = 'G:\공유 드라이브\Battery Software Group (2025)\Members\김주은\PINN\합성데이터';
if ~exist(cfg.out_dir, 'dir'); mkdir(cfg.out_dir); end

% Synthetic sample count. Each sample has an independent k value.
cfg.nSyntheticSamples = 100;

% Random seed configuration
% soh    : base seed for SOH Gaussian noise, applied per DRT sample.
% markov : base seed for Markov voltage noise, applied per final voltage sample.
cfg.seed.soh    = 1;
cfg.seed.markov = 1;

% nRC DRT discretization
cfg.nRC = 50;
cfg.tau_min = 0.1;
cfg.tau_max = 400;

% DRT distribution. sigma10 is fixed by design.
cfg.sigma10 = [0.200; 0.190];

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

% If true, SOH label is clipped to [SOH_aged, 100] instead of a fixed [70,100].
cfg.reference.clip_to_real_reference_range = true;

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

% 2RC fitting options
cfg.fit.useParallel = true;
cfg.fit.nStartPoints = 40;
cfg.fit.para0 = [0.0012; 0.0008; 0.0012; 6; 70];
cfg.fit.lb    = [0; 0; 0; 0.001; 0.001];
cfg.fit.ub    = [0.05; 0.03; 0.03; 500; 2000];

% SOH dummy label configuration.
% These weights are now applied to DRT-generated TRUE ECM features,
% not fitted ECM features.
cfg.soh.weights = [0.30 0.25 0.20 0.15 0.10];
cfg.soh.noise_sigma = 1.0;            % epsilon ~ N(0, 1)
cfg.soh.clip_range = [70 100];

% k-based optional shape. Use 1 for no direct k nonlinearity.
% If cfg.soh.use_k_base = true, SOH uses k^gamma as the main trend and
% DRT score is saved for diagnostics. Default below uses only DRT true ECM
% score so the label is explicitly DRT-derived.
cfg.soh.use_k_base = false;
cfg.soh.k_gamma = 1.0;

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

if cfg.reference.clip_to_real_reference_range
    cfg.soh.clip_range = [cfg.SOH_aged 100];
end

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

ms = MultiStart('UseParallel', cfg.fit.useParallel, 'Display', 'off');
opt = optimset('display','off', 'MaxIter', 1e3, 'MaxFunEvals', 1e4, ...
               'TolFun', eps, 'TolX', eps);
startPts = CustomStartPointSet(makeStartPoints_LogTau(cfg.fit.nStartPoints, ...
                                                       cfg.fit.lb, cfg.fit.ub, ...
                                                       cfg.fit.para0));

% tau1 < tau2 constraint: [0 0 0 1 -1] * p <= 0
A_lin = [0 0 0 1 -1];
b_lin = 0;

% DRT-generated TRUE ECM references.
% For bimodal DRT, true ECM is [R0, R_mode1, R_mode2, tau_mode1, tau_mode2].
cfg.trueECM.fresh = drtParam_to_trueECM_2RC(cfg.fresh);
cfg.trueECM.aged  = drtParam_to_trueECM_2RC(cfg.aged);

S_aged = soh_score(cfg.trueECM.aged, cfg.trueECM.fresh, cfg.soh.weights);
if S_aged <= 0 || ~isfinite(S_aged)
    error('S_aged must be positive and finite. Check fresh/aged DRT true ECM references.');
end
cfg.soh.A = (100 - cfg.SOH_aged) / S_aged;

% Generate uniformly spaced degradation indices without exact endpoints.
% k = 0 and k = 1 are used only as fresh/aged anchors.
kList = ((1:cfg.nSyntheticSamples).' - 0.5) / cfg.nSyntheticSamples;

%% ------------------------ Main generation loop -----------------------
rows = {};
rowCounter = 0;

for sampleIdx = 1:cfg.nSyntheticSamples
    k = kList(sampleIdx);
    drtParam = interpolate_DRT_param(cfg.fresh, cfg.aged, k);

    % DRT-generated TRUE ECM for label. This is calculated before voltage,
    % noise, and fitting. It is the latent state used to assign SOH.
    trueECM = drtParam_to_trueECM_2RC(drtParam);

    % SOH Gaussian noise is assigned once per DRT sample.
    % Example: nDRT = 50 -> sohSeed = cfg.seed.soh ... cfg.seed.soh+49.
    sohSeed = cfg.seed.soh + sampleIdx - 1;
    rng(sohSeed, 'twister');
    sohEps = cfg.soh.noise_sigma * randn();

    [SOH_DRT, DRT_score, drtFeatureVec] = soh_label_from_true_drt_ecm( ...
        trueECM, cfg.trueECM.fresh, cfg.soh.weights, cfg.soh.A, ...
        sohEps, cfg.soh.clip_range, k, cfg.SOH_aged, ...
        cfg.soh.use_k_base, cfg.soh.k_gamma);

    % nRC truth from aging-dependent DRT for voltage generation.
    [X_true, drtMeta] = make_nRC_from_DRT(drtParam, cfg.sigma10, cfg.nRC, cfg.tau_min, cfg.tau_max);

    for fileIdx = 1:numel(cfg.driving_files)
        [t_vec, I_vec, base_name] = readDrivingProfile(cfg.driving_files{fileIdx}, fileIdx);
        OCV = zeros(size(t_vec));

        V_true = RC_model_n(X_true, t_vec, I_vec, cfg.nRC);

        % Seed 0 means no noise; seeds 1..nSeeds use Markov noise.
        if cfg.noise.includeNonNoise
            seedList = 0:cfg.noise.nSeeds;
        else
            seedList = 1:cfg.noise.nSeeds;
        end

        for seed = seedList
            markovSeed = NaN;

            if seed == 0
                V_syn = V_true;
                seedName = 'Non_noise';
            else
                % Markov noise is assigned once per final voltage sample.
                % Example: nDRT = 50, nLoad = 20, nSeeds = 1
                %          -> markovSeed = cfg.seed.markov ... cfg.seed.markov+999.
                linearVoltageIdx = ((sampleIdx - 1) * numel(cfg.driving_files) + (fileIdx - 1)) ...
                    * cfg.noise.nSeeds + seed;
                markovSeed = cfg.seed.markov + linearVoltageIdx - 1;

                rng(markovSeed, 'twister');
                V_syn = MarkovNoise_idx(V_true, cfg.noise.epsilon_percent_span, ...
                    cfg.noise.initial_state, cfg.noise.sigma);
                seedName = sprintf('V_SD%d', seed);
            end

            [bestP, rmse, exitflag, iter] = fit2RC_full(V_syn, t_vec, I_vec, OCV, ...
                cfg, ms, startPts, opt, A_lin, b_lin);

            rowCounter = rowCounter + 1;
            rows(rowCounter,:) = { ...
                sampleIdx, k, fileIdx, base_name, seed, seedName, sohSeed, markovSeed, ...
                drtParam.R0, drtParam.A_tot, drtParam.w(1), drtParam.w(2), ...
                drtParam.tau_mode(1), drtParam.tau_mode(2), ...
                trueECM(1), trueECM(2), trueECM(3), trueECM(4), trueECM(5), ...
                bestP(1), bestP(2), bestP(3), bestP(4), bestP(5), ...
                rmse, exitflag, iter, SOH_DRT, DRT_score, ...
                drtFeatureVec(1), drtFeatureVec(2), drtFeatureVec(3), drtFeatureVec(4), drtFeatureVec(5), ...
                sum(drtMeta.R), max(drtMeta.g_tau)};
        end
    end

    if mod(sampleIdx, 10) == 0
        fprintf('Completed %d / %d synthetic DRT samples\n', sampleIdx, cfg.nSyntheticSamples);
    end
end

varNames = {'sample_id','k','load_id','load_name','seed','seed_name','soh_seed','markov_seed', ...
            'true_R0','true_A_tot','true_w1','true_w2','true_tau_mode1','true_tau_mode2', ...
            'trueECM_R0','trueECM_R1','trueECM_R2','trueECM_tau1','trueECM_tau2', ...
            'R0_hat','R1_hat','R2_hat','tau1_hat','tau2_hat', ...
            'RMSE','exitflag','iterations','SOH_DRT','DRT_score', ...
            'z1_true_R0','z2_true_Rsum','z3_true_tau_ratio','z4_true_tau_max','z5_true_Rsplit', ...
            'DRT_sumR','DRT_g_tau_max'};

Dataset = cell2table(rows, 'VariableNames', varNames);

save(fullfile(cfg.out_dir, 'Synthetic_DRT_SOH_Dataset_DRTLabel_CellRef.mat'), 'Dataset', 'cfg');
writetable(Dataset, fullfile(cfg.out_dir, 'Synthetic_DRT_SOH_Dataset_DRTLabel_CellRef.csv'));

fprintf('\nDone. Rows = %d\n', height(Dataset));
fprintf('Saved MAT: %s\n', fullfile(cfg.out_dir, 'Synthetic_DRT_SOH_Dataset_DRTLabel_CellRef.mat'));
fprintf('Saved CSV: %s\n', fullfile(cfg.out_dir, 'Synthetic_DRT_SOH_Dataset_DRTLabel_CellRef.csv'));

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

function pTrue = drtParam_to_trueECM_2RC(param)
    % Converts bimodal DRT generator parameters into the latent TRUE 2RC ECM.
    % R_mode_i is the area of the i-th DRT mode: A_tot * w_i / sum(w).
    % tau_i is the DRT mode location. This true ECM is used only for labels
    % and diagnostics; ML/DL inputs should still use fitted ECM parameters.
    w = param.w(:);
    tau = param.tau_mode(:);
    if numel(w) ~= 2 || numel(tau) ~= 2
        error('drtParam_to_trueECM_2RC currently assumes bimodal DRT with two modes.');
    end
    if any(w < 0) || sum(w) <= 0
        error('DRT mode weights must be nonnegative and have positive sum.');
    end
    R_modes = param.A_tot * w / sum(w);
    [tauSorted, idx] = sort(tau, 'ascend');
    R_sorted = R_modes(idx);
    pTrue = [param.R0; R_sorted(:); tauSorted(:)];
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

function X0 = makeStartPoints_LogTau(nStart, lb, ub, para0)
    X0 = zeros(nStart, numel(para0));
    for ii = 1:nStart
        R0 = lb(1) + rand()*(ub(1)-lb(1));
        R1 = lb(2) + rand()*(ub(2)-lb(2));
        R2 = lb(3) + rand()*(ub(3)-lb(3));
        tauA = 10^(log10(lb(4)) + rand()*(log10(ub(4))-log10(lb(4))));
        tauB = 10^(log10(lb(5)) + rand()*(log10(ub(5))-log10(lb(5))));
        tauPair = sort([tauA tauB]);
        X0(ii,:) = [R0 R1 R2 tauPair(1) tauPair(2)];
    end
    X0 = [para0(:).'; X0];
end

function [bestP, bestFval, eflg, iter] = fit2RC_full(V, t, I, OCV, cfg, ms, startPts, opt, A_lin, b_lin)
    problem = createOptimProblem('fmincon', ...
        'objective', @(p)RMSE_2RC_local(V, p, t, I, OCV), ...
        'x0', cfg.fit.para0, 'lb', cfg.fit.lb, 'ub', cfg.fit.ub, ...
        'Aineq', A_lin, 'bineq', b_lin, 'options', opt);

    [bestP, bestFval, eflg, ~, sltns] = run(ms, problem, startPts);
    if isempty(sltns)
        iter = NaN;
    else
        idx = find([sltns.Fval] == bestFval, 1);
        iter = sltns(idx).Output.iterations;
    end
    bestP = bestP(:);
end

function cost = RMSE_2RC_local(data, para, t, I, OCV)
    model = RC_model_2(para, t, I, OCV);
    cost = sqrt(mean((data(:) - model(:)).^2));
end

function [SOH, scoreVal, z] = soh_label_from_true_drt_ecm(pTrue, pFreshTrue, weights, A, sohEps, clipRange, k, SOH_aged, useKBase, kGamma)
    % Main/default mode: SOH is generated from DRT true ECM score.
    % Optional mode: SOH follows k^gamma directly, while DRT score remains
    % saved as a diagnostic. Keep useKBase=false when you want the label to
    % come from DRT/true ECM features.
    scoreVal = soh_score(pTrue, pFreshTrue, weights);
    z = soh_features(pTrue, pFreshTrue);

    if useKBase
        SOH_raw = 100 - (100 - SOH_aged) * (k ^ kGamma) + sohEps;
    else
        SOH_raw = 100 - A * scoreVal + sohEps;
    end

    SOH = min(clipRange(2), max(clipRange(1), SOH_raw));
end

function scoreVal = soh_score(pHat, pFresh, weights)
    z = soh_features(pHat, pFresh);
    scoreVal = sum(weights(:).' .* z(:).');
end

function z = soh_features(pHat, pFresh)
    tiny = eps;
    R0h = max(pHat(1), tiny); R1h = max(pHat(2), tiny); R2h = max(pHat(3), tiny);
    t1h = max(pHat(4), tiny); t2h = max(pHat(5), tiny);
    R0f = max(pFresh(1), tiny); R1f = max(pFresh(2), tiny); R2f = max(pFresh(3), tiny);
    t1f = max(pFresh(4), tiny); t2f = max(pFresh(5), tiny);

    % z1: true ohmic resistance increase from DRT aging state
    z1 = max(0, log(R0h / R0f));

    % z2: true polarization resistance increase, i.e., DRT total area increase
    z2 = max(0, log((R1h + R2h) / (R1f + R2f)));

    % z3: true DRT mode separation/shape change
    z3 = max(0, log((t2h / t1h) / (t2f / t1f)));

    % z4: true slow time-constant increase
    z4 = max(0, log(max(t1h, t2h) / max(t1f, t2f)));

    % z5: true DRT branch/mode resistance redistribution penalty.
    % Penalizes only when a mode resistance deviates by more than 2x.
    rDev1 = abs(log(R1h / R1f));
    rDev2 = abs(log(R2h / R2f));
    z5 = softplus(max(rDev1, rDev2) - log(2));

    z = [z1 z2 z3 z4 z5];
end

function y = softplus(x)
    y = log1p(exp(-abs(x))) + max(x, 0);
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
