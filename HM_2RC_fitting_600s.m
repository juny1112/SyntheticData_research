function results = HM_2RC_fitting_600s(cfg)
% RUN_2RC_FITTING_600S
% Outputs (results struct):
%   results.cfg
%   results.all_para_hats.(cellKey) : [nSeg×8] = [R0 R1 R2 tau1 tau2 RMSE exitflag iter] (R in Ohm)
%   results.all_rmse.(cellKey)      : [nSeg×1] RMSE (V)
%   results.all_load_idx.(cellKey)  : [nSOC×nLoads] (SOC×부하별 seg index)
%   results.cellNameMap.(cellKey)   : original file stem (base_raw)
%   results.Tbl_Load_ECM.(LOAD)     : table (rows=cells, cols=SOC×params), R in mOhm
%   results.Tbl_Load_RMSE.(LOAD)    : table (rows=SOC, cols=cells), RMSE in mV
%
% Requirements:
%   - Optimization Toolbox (fmincon)
%   - Global Optimization Toolbox (MultiStart, RandomStartPointSet)
%   - (Optional) Parallel Computing Toolbox if cfg.useParallel=true

arguments
    cfg.folder_SIM (1,:) char
    cfg.fit_window_sec (1,1) double = 600

    cfg.SOC_list (1,:) double = [50 70]
    cfg.loadNames (1,:) cell = {'US06','UDDS','HWFET','WLTP','CITY1','CITY2','HW1','HW2'}

    cfg.use_block_mapping (1,1) logical = true
    cfg.blkSize (1,1) double = 8

    cfg.numStartPoints (1,1) double = 40
    cfg.useParallel (1,1) logical = true

    cfg.savePath (1,:) char = ''
    cfg.saveFigs (1,1) logical = true
    cfg.saveMat  (1,1) logical = true
end

%% ---- list files ----
sim_files = dir(fullfile(cfg.folder_SIM,"*_SIM.mat"));
if isempty(sim_files)
    error("SIM 파일을 찾지 못했습니다: %s", cfg.folder_SIM);
end

%% ---- save path ----
if isempty(cfg.savePath)
    save_root = fileparts(cfg.folder_SIM); % -> ...\SIM_parsed
    cfg.savePath = fullfile(save_root, sprintf('2RC_fitting_%ds', cfg.fit_window_sec));
end
if ~exist(cfg.savePath,'dir'); mkdir(cfg.savePath); end

%% ---- constants ----
SOC_list = cfg.SOC_list(:).';
nSOC     = numel(SOC_list);

loadNames = cfg.loadNames;
nLoads    = numel(loadNames);

if cfg.blkSize ~= nLoads
    warning('blkSize(%d) != nLoads(%d). 블록 매핑이 의도대로 동작하지 않을 수 있습니다.', cfg.blkSize, nLoads);
end

%% ---- optimizer config ----
ms       = MultiStart("UseParallel",cfg.useParallel,"Display","off");
startPts = RandomStartPointSet('NumStartPoints',cfg.numStartPoints);
opt      = optimset('display','off','MaxIter',1e3,'MaxFunEvals',1e4, ...
                    'TolFun',eps,'TolX',eps);

% 2-RC initial / bounds / linear constraint (tau1 < tau2)
para0 = [0.003 0.0005 0.0005 10 100];
lb    = [0       0       0      0.01  0.01];
ub    = [0.05 0.005 0.03 100 5000];
A_lin = [0 0 0 1 -1];  b_lin = 0;

%% ---- results containers ----
results = struct();
results.cfg           = cfg;
results.all_para_hats = struct();
results.all_rmse      = struct();
results.all_load_idx  = struct();
results.cellNameMap   = struct();   % key->base_raw (보기용)
results.Tbl_Load_ECM  = struct();
results.Tbl_Load_RMSE = struct();

%% ===================== main loop =====================
for f = 1:numel(sim_files)

    S = load(fullfile(cfg.folder_SIM,sim_files(f).name),"SIM_table");
    if ~isfield(S,"SIM_table")
        warning("SIM_table 없음: %s (skip)", sim_files(f).name);
        continue
    end
    SIM_table = S.SIM_table;

    base_raw   = erase(sim_files(f).name,"_SIM.mat");
    base_field = matlab.lang.makeValidName(base_raw);

    nSeg = height(SIM_table);
    if nSeg==0
        warning("No SIM rows: %s (skip)", base_raw);
        continue
    end

    results.cellNameMap.(base_field) = base_raw;

    % ---- SOC group code assignment (1..nSOC) ----
    grp_code = zeros(nSeg,1); % 0=unassigned

    % (A) block mapping (optional)
    if cfg.use_block_mapping && (nSeg >= cfg.blkSize*nSOC)
        for g = 1:nSOC
            ii = ((g-1)*cfg.blkSize + 1) : min(g*cfg.blkSize, nSeg);
            grp_code(ii) = g;
        end
    end

    % (B) fallback mapping by SOC_center
    if any(grp_code==0)
        SOC_center = mean([SIM_table.SOC1 SIM_table.SOC2], 2, 'omitnan');

        miss = isnan(SOC_center);
        if any(miss) && ismember('SOC_vec', SIM_table.Properties.VariableNames)
            try
                SOC_center(miss) = cellfun(@(v) mean(v,'omitnan'), SIM_table.SOC_vec(miss));
            catch
            end
        end

        valid = ~isnan(SOC_center);
        if any(valid)
            [~, gmin] = min(abs(SOC_center(valid) - SOC_list), [], 2);
            vidx = find(valid);
            grp_code(vidx(grp_code(vidx)==0)) = gmin(grp_code(vidx)==0);
        end
    end

    % ---- pick seg index per (SOC, Load) ----
    load_idx = nan(nSOC, nLoads);
    for g = 1:nSOC
        for l = 1:nLoads
            load_idx(g,l) = pickLoadSegIdx(g, l, grp_code, nSeg, cfg.use_block_mapping, cfg.blkSize, nLoads);
        end
    end
    results.all_load_idx.(base_field) = load_idx;

    % ---- fit all segments (cropped to window) ----
    para_hats = nan(nSeg, 8); % [R0 R1 R2 tau1 tau2 RMSE exitflag iter]
    RMSE_list = nan(nSeg, 1);

    seg_list = 1:nSeg;

    % optional plotting
    fig = [];
    cols = 8; rows = max(1, ceil(numel(seg_list)/cols));
    if cfg.saveFigs
        fig = figure('Name',[base_raw sprintf(' – 2RC fitting (%ds)', cfg.fit_window_sec)], ...
            'NumberTitle','off', 'Position',[100 100 1600 900], 'Color','w');
        try
            sgtitle(strrep(base_raw,'_','\_') + sprintf(" – 2RC fitting (%ds)", cfg.fit_window_sec), 'Interpreter','tex');
        catch
            suptitle(strrep(base_raw,'_','\_') + sprintf(" – 2RC fitting (%ds)", cfg.fit_window_sec));
        end
    end

    pcount = 0;
    for s = seg_list
        pcount = pcount + 1;

        try
            t = SIM_table.time{s};
            I = SIM_table.current{s};
            V = SIM_table.voltage{s};

            O = [];
            if ismember('OCV_vec', SIM_table.Properties.VariableNames)
                O = SIM_table.OCV_vec{s};
            end

            % normalize time to start at 0 (seconds)
            if isduration(t)
                t = seconds(t - t(1));
            else
                t = t - t(1);
            end

            % crop to window
            [t2, I2, V2, O2, okCrop] = cropToWindow(t, I, V, O, cfg.fit_window_sec);
            if ~okCrop
                warning("(%s) seg %d: crop(%ds) 후 데이터 부족 -> skip", base_raw, s, cfg.fit_window_sec);
                continue;
            end

            % (권장) OCV가 없으면 스킵하고 싶다면 아래 주석 해제
            % if isempty(O2)
            %     warning("(%s) seg %d: OCV 없음 -> skip", base_raw, s);
            %     continue;
            % end

            problem = createOptimProblem('fmincon', ...
              'objective',@(p)RMSE_2RC(V2,p,t2,I2,O2), ...
              'x0',para0,'lb',lb,'ub',ub, ...
              'Aineq',A_lin,'bineq',b_lin,'options',opt);

            [Pbest, Fval, exitflg, ~, sol] = run(ms,problem,startPts);

            it = NaN;
            if ~isempty(sol)
                it = sol(find([sol.Fval]==Fval,1)).Output.iterations;
            end

            para_hats(s,:) = [Pbest, Fval, exitflg, it];
            RMSE_list(s)   = Fval;

            % plot if enabled
            if cfg.saveFigs && isgraphics(fig,'figure')
                soc_txt = 'SOC ?';
                if grp_code(s) >= 1 && grp_code(s) <= nSOC
                    soc_txt = sprintf('SOC %d', SOC_list(grp_code(s)));
                end

                load_txt = 'LOAD ?';
                if grp_code(s) >= 1 && grp_code(s) <= nSOC
                    g = grp_code(s);
                    l = find(load_idx(g,:)==s, 1, 'first');
                    if ~isempty(l), load_txt = loadNames{l}; end
                end

                V_fit = RC_model_2(Pbest, t2, I2, O2);
                V_ini = RC_model_2(para0 , t2, I2, O2);

                subplot(rows, cols, pcount);
                plot(t2, V2, 'k', t2, V_fit, 'r', t2, V_ini, '--b', 'LineWidth', 1.1);
                grid on;
                xlabel('Time (s)'); ylabel('Voltage (V)');
                title(sprintf('SIM%d | %s | %s | RMSE=%.2f mV', s, load_txt, soc_txt, Fval*1e3), 'Interpreter','none');
                legend('True','Fitted','Initial','Location','northeast','Box','off');
            end

        catch ME
            warning("(%s) seg %d fitting failed: %s", base_raw, s, ME.message);
        end
    end

    % save fig per file
    if cfg.saveFigs && isgraphics(fig,'figure')
        savefig(fig, fullfile(cfg.savePath, sprintf('%s_2RC_fit_%ds.fig', base_raw, cfg.fit_window_sec)));
        close(fig);
    end

    % store per file
    results.all_para_hats.(base_field) = para_hats; % Ohm
    results.all_rmse.(base_field)      = RMSE_list; % V

    % log
    cntStr = "";
    for g=1:nSOC
        cntStr = cntStr + sprintf(" SOC%d=%d", SOC_list(g), nnz(grp_code==g));
    end
    fprintf('[done] %s | fitted %d segs (crop=%ds) | counts:%s\n', ...
        base_raw, numel(seg_list), cfg.fit_window_sec, cntStr);

end

fprintf("모든 파일 처리 완료!\n");

%% ============ build load-wise tables (only) ============
cellKeys = fieldnames(results.all_para_hats); % base_field keys
if isempty(cellKeys)
    warning('Load-wise 테이블 생성 실패: results.all_para_hats 가 비어 있습니다.');
else
    % columns (cells) will use valid names (base_field). 원본명은 results.cellNameMap에 보관.
    cellVarNames = cellfun(@matlab.lang.makeValidName, cellKeys, 'UniformOutput', false);
    nCells = numel(cellKeys);

    pNames = {'R0','R1','R2','tau1','tau2'};
    nP     = numel(pNames);

    Tbl_Load_ECM  = struct();
    Tbl_Load_RMSE = struct();

    for l = 1:nLoads
        loadName = loadNames{l};

        % ECM table (rows=cells, cols=SOC×params)
        ECM_mat = nan(nCells, numel(SOC_list)*nP);
        vnamesL = strings(1, numel(SOC_list)*nP);

        col = 0;
        for si = 1:numel(SOC_list)
            soc = SOC_list(si);
            for pi = 1:nP
                col = col + 1;
                p   = pNames{pi};

                if pi <= 3
                    vnamesL(col) = sprintf('SOC%d_%s_mOhm', soc, p);
                else
                    vnamesL(col) = sprintf('SOC%d_%s', soc, p);
                end

                for ci = 1:nCells
                    key_field   = cellKeys{ci};
                    load_idx_ci = results.all_load_idx.(key_field);
                    sIdx = load_idx_ci(si, l);

                    if ~isnan(sIdx) && sIdx>=1
                        Prow = results.all_para_hats.(key_field); % [nSeg×8], R in Ohm
                        if sIdx <= size(Prow,1)
                            val = Prow(sIdx, pi);
                            if pi <= 3
                                val = val * 1000; % Ohm -> mOhm
                            end
                            ECM_mat(ci, col) = val;
                        end
                    end
                end
            end
        end

        T_ECM = array2table(ECM_mat, ...
            'RowNames',      cellVarNames, ...
            'VariableNames', cellstr(vnamesL));
        T_ECM.Properties.Description = sprintf('rows=cells / cols=SOC-wise %s 2RC params (R mOhm, tau s) | fit=%ds', loadName, cfg.fit_window_sec);

        % RMSE table (rows=SOC, cols=cells, mV)
        RMSE_mat = nan(nSOC, nCells);
        rnames   = strings(nSOC,1);
        for si = 1:nSOC, rnames(si) = sprintf('SOC%d', SOC_list(si)); end

        for ci = 1:nCells
            key_field   = cellKeys{ci};
            load_idx_ci = results.all_load_idx.(key_field);
            Erow = results.all_rmse.(key_field); % V

            for si = 1:nSOC
                sIdx = load_idx_ci(si, l);
                if ~isnan(sIdx) && sIdx>=1 && sIdx <= numel(Erow)
                    RMSE_mat(si, ci) = Erow(sIdx) * 1e3; % mV
                end
            end
        end

        T_RMSE = array2table(RMSE_mat, ...
            'RowNames',      cellstr(rnames), ...
            'VariableNames', cellVarNames);
        T_RMSE.Properties.Description = sprintf('rows=SOC / cols=cells / value=%s RMSE (mV) | fit=%ds', loadName, cfg.fit_window_sec);

        Tbl_Load_ECM.(loadName)  = T_ECM;
        Tbl_Load_RMSE.(loadName) = T_RMSE;
    end

    results.Tbl_Load_ECM  = Tbl_Load_ECM;
    results.Tbl_Load_RMSE = Tbl_Load_RMSE;
end

%% ============ save mat ============
if cfg.saveMat
    results_file = fullfile(cfg.savePath, sprintf('2RC_results_%ds.mat', cfg.fit_window_sec));
    save(results_file, '-struct', 'results', '-v7.3');
    fprintf('2RC 결과 저장 완료: %s\n', results_file);
end

end % main function


%% ====================== local helper functions ======================
function cost = RMSE_2RC(V_true, para, t, I, OCV)
    V_est = RC_model_2(para,t,I,OCV);
    cost  = sqrt(mean((V_true - V_est).^2, 'omitnan'));
end

function [t2, I2, V2, O2, ok] = cropToWindow(t, I, V, O, winSec)
    ok = true;

    if isempty(t) || numel(t) < 5
        ok = false; t2=[]; I2=[]; V2=[]; O2=[]; return
    end

    m = (t <= winSec);
    if nnz(m) < 5
        ok = false; t2=[]; I2=[]; V2=[]; O2=[]; return
    end

    t2 = t(m); I2 = I(m); V2 = V(m);

    if isempty(O)
        O2 = [];
    else
        try
            O2 = O(m);
        catch
            O2 = [];
        end
    end
end

function sIdx = pickLoadSegIdx(g, loadIdx, grp_code, nSeg, use_block_mapping, blkSize, nLoads)
    sIdx = NaN;

    % 1) block mapping
    if use_block_mapping && (blkSize >= nLoads) && (nSeg >= blkSize*max(g,1))
        tmp = (g-1)*blkSize + loadIdx;
        if tmp >= 1 && tmp <= nSeg
            sIdx = tmp; return
        end
    end

    % 2) fallback: g-group에서 등장 순서대로 loadIdx번째
    idx = find(grp_code==g);
    if isempty(idx), return; end
    idx = sort(idx(:));
    if numel(idx) >= loadIdx
        sIdx = idx(loadIdx);
    end
end

function V_est = RC_model_2(X, t_vec, I_vec, OCV)
% 2RC model function (embedded)
% X: [R0, R1, R2, tau1, tau2]
% NOTE: OCV가 []일 수 있으므로 방어 로직 포함.

    R0   = X(1);
    R1   = X(2);
    R2   = X(3);
    tau1 = X(4);
    tau2 = X(5);

    t_vec = t_vec(:);
    I_vec = I_vec(:);
    N = length(t_vec);

    if length(I_vec) ~= N
        error('RC_model_2: length mismatch (t_vec=%d, I_vec=%d)', N, length(I_vec));
    end

    if isempty(OCV)
        OCV = zeros(N,1);
    else
        OCV = OCV(:);
        if length(OCV) ~= N
            error('RC_model_2: length mismatch (OCV=%d, expected=%d)', length(OCV), N);
        end
    end

    % dt 계산 (원본 로직 유지: 첫 dt=1)
    dt = [1; diff(t_vec)];

    V_est = zeros(N, 1);

    % RC states
    Vrc1 = 0;
    Vrc2 = 0;

    for k = 1:N
        IR0 = R0 * I_vec(k);

        alpha1 = exp(-dt(k)/tau1);
        alpha2 = exp(-dt(k)/tau2);

        if k > 1
            Vrc1 = Vrc1*alpha1 + R1*(1 - alpha1)*I_vec(k-1);
            Vrc2 = Vrc2*alpha2 + R2*(1 - alpha2)*I_vec(k-1);
        end

        V_est(k) = OCV(k) + IR0 + Vrc1 + Vrc2;
    end
end
