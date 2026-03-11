%% ========================================================================
%  CostSurface_2RC_fromCellData_MS_batch_optimset.m
%
%  (UPDATE)
%   - close(fig) 주석처리(그림 유지)
%   - 가독성 좋은 Summary Table 생성:
%       base_seg | surf_R0 surf_R1 surf_R2 surf_tau1 surf_tau2 surf_RMSE |
%                  MS_R0 MS_R1 MS_R2 MS_tau1 MS_tau2 MS_RMSE MS_exitflag
%
%  (IMPORTANT)
%   - cost-surface에서 R0/R1/R2 최적값도 저장하려면,
%     tau grid마다 fmincon의 argmin pBest를 같이 저장해야 함.
% ========================================================================

clear; clc; close all;

%% ========================= 사용자 설정 =========================

folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\20degC\이름정렬';

save_dir  = fullfile(folder_SIM, 'CostSurface_2RC_fromCellData');
if ~exist(save_dir,'dir'), mkdir(save_dir); end

fit_window_sec = 600;

cell_select.mode    = "index";      % "all" | "index" | "pattern"
cell_select.index   = 1:12;          % [1 2 5 6 8] 등
cell_select.pattern = "*_SIM.mat";

seg_select.mode  = "index";         % "all" | "index"
seg_select.index = 10 ;

allow_missing_ocv = true;

% --- τ grid (너가 현재 쓰는 값) ---
tau1_vec = 10.^(linspace(-2, 2, 51));     % 1 ~ 50 s
tau2_vec = 10.^(linspace(-2, 3.3, 201));    % 10 ~ 1000 s
[T1, T2] = meshgrid(tau1_vec, tau2_vec);

run_cost_surface = true;
run_multistart   = true;

if isempty(gcp('nocreate'))
    parpool;
end

ms       = MultiStart("UseParallel", true, "Display", "off");
startPts = RandomStartPointSet('NumStartPoints', 40);

optSurf = optimset('display','off', ...
    'MaxIter',1e3,'MaxFunEvals',1e4, ...
    'TolFun',eps,'TolX',eps);

optMS = optimset('display','off', ...
    'MaxIter',1e3,'MaxFunEvals',1e4, ...
    'TolFun',eps,'TolX',eps);

R0_init = 0.003; R1_init = 0.0005; R2_init = 0.0005;

tau1_init_MS = 10;
tau2_init_MS = 100;

R0_ub = 0.05;   % [ohm]
R1_ub = 0.005;
R2_ub = 0.03;

lb_MS = [0, 0, 0, tau1_vec(1),  tau2_vec(1)];
ub_MS = [R0_ub, R1_ub, R2_ub, tau1_vec(end), tau2_vec(end)];

%% ========================= 파일 리스트 구성 =========================
switch cell_select.mode
    case "all"
        sim_files = dir(fullfile(folder_SIM, "*_SIM.mat"));
        file_list = 1:numel(sim_files);

    case "index"
        sim_files = dir(fullfile(folder_SIM, "*_SIM.mat"));
        if isempty(sim_files), error("SIM 파일 없음: %s", folder_SIM); end
        file_list = cell_select.index;
        file_list = file_list(file_list>=1 & file_list<=numel(sim_files));
        if isempty(file_list), error("cell_select.index가 비었거나 범위 밖"); end

    case "pattern"
        sim_files = dir(fullfile(folder_SIM, cell_select.pattern));
        if isempty(sim_files), error("pattern 매칭 파일 없음: %s", cell_select.pattern); end
        file_list = 1:numel(sim_files);

    otherwise
        error("cell_select.mode 오류: %s", cell_select.mode);
end

if isempty(file_list)
    error("대상 SIM 파일이 0개입니다.");
end
fprintf("Targets: %d SIM files\n", numel(file_list));

%% ========================= 메인 루프: 셀 × 세그 =========================
for fi = file_list

    sim_path = fullfile(folder_SIM, sim_files(fi).name);
    S = load(sim_path, "SIM_table");
    if ~isfield(S, "SIM_table")
        warning("SIM_table 없음: %s (skip)", sim_files(fi).name);
        continue;
    end
    SIM_table = S.SIM_table;

    base_raw = erase(sim_files(fi).name, "_SIM.mat");
    nSeg = height(SIM_table);
    if nSeg == 0
        warning("No SIM rows: %s (skip)", base_raw);
        continue;
    end

    switch seg_select.mode
        case "all"
            seg_list = 1:nSeg;

        case "index"
            seg_list = seg_select.index;
            seg_list = seg_list(seg_list>=1 & seg_list<=nSeg);
            if isempty(seg_list)
                warning("(%s) 선택 seg 없음 → skip", base_raw);
                continue;
            end

        otherwise
            error("seg_select.mode 오류: %s", seg_select.mode);
    end

    fprintf("\n=== CELL: %s | nSeg=%d | run segs=%d ===\n", base_raw, nSeg, numel(seg_list));

    % ---- results_cell 누적 ----
    results_cell = struct();
    results_cell.sim_path        = sim_path;
    results_cell.base_raw        = base_raw;
    results_cell.fit_window_sec  = fit_window_sec;
    results_cell.tau1_vec        = tau1_vec;
    results_cell.tau2_vec        = tau2_vec;
    results_cell.segs = struct( ...
        'seg_idx',      {}, ...
        't_len',        {}, ...
        'surface_best', {}, ...
        'ms_best',      {}, ...
        'cost_surface', {}, ...
        'cost_surface_R0', {}, ...
        'cost_surface_R1', {}, ...
        'cost_surface_R2', {} );

    % ---- (NEW) Summary Table 누적용 ----
    Summary = table([],[],[],[],[],[],[],[],[],[],[],[],[],[], ...
        'VariableNames', { ...
        'base_seg', ...
        'surf_R0','surf_R1','surf_R2','surf_tau1','surf_tau2','surf_RMSE', ...
        'MS_R0','MS_R1','MS_R2','MS_tau1','MS_tau2','MS_RMSE','MS_exitflag'});

    for si = 1:numel(seg_list)
        s = seg_list(si);
        fprintf("  - seg %d (%d/%d)\n", s, si, numel(seg_list));

        try
            %% (1) 데이터 추출
            t = SIM_table.time{s};
            I = SIM_table.current{s};
            V = SIM_table.voltage{s};

            O = [];
            if ismember('OCV_vec', SIM_table.Properties.VariableNames)
                O = SIM_table.OCV_vec{s};
            end

            if isduration(t)
                t = seconds(t - t(1));
            else
                t = t - t(1);
            end

            [t2, I2, V2, O2, okCrop] = cropToWindow(t, I, V, O, fit_window_sec);
            if ~okCrop
                warning("(%s) seg %d: crop 후 데이터 부족 → skip", base_raw, s);
                continue;
            end

            t2 = t2(:); I2 = I2(:); V2 = V2(:);

            if isempty(O2)
                if allow_missing_ocv
                    O2 = zeros(size(t2));
                else
                    error("OCV_vec 없음 (allow_missing_ocv=false)");
                end
            else
                O2 = O2(:);
            end

            %% (2) Cost-surface (RMSE + R0/R1/R2 argmin 저장)
            cost_surface = [];
            cost_R0 = []; cost_R1 = []; cost_R2 = [];
            tau1_best = NaN; tau2_best = NaN; rmse_surf_best = NaN;
            surfR0_best = NaN; surfR1_best = NaN; surfR2_best = NaN;

            if run_cost_surface
                nT2 = numel(tau2_vec);
                nT1 = numel(tau1_vec);

                cost_surface = zeros(nT2, nT1);
                cost_R0 = nan(nT2, nT1);
                cost_R1 = nan(nT2, nT1);
                cost_R2 = nan(nT2, nT1);

                for ii = 1:nT1
                    tau1 = tau1_vec(ii);

                    for jj = 1:nT2
                        tau2 = tau2_vec(jj);

                        p0 = [R0_init, R1_init, R2_init, tau1, tau2];
                        lb = [0,       0,       0,       tau1, tau2];
                        ub = [R0_ub,   R1_ub,   R2_ub,   tau1, tau2];

                        [pBest, fval] = fmincon(@(p) RMSE_2RC(V2, p, t2, I2, O2), ...
                                                p0, [],[],[],[], lb, ub, [], optSurf);

                        cost_surface(jj, ii) = fval;
                        cost_R0(jj, ii) = pBest(1);
                        cost_R1(jj, ii) = pBest(2);
                        cost_R2(jj, ii) = pBest(3);
                    end
                end

                [rmse_surf_best, idxLin] = min(cost_surface(:));
                [r,c] = ind2sub(size(cost_surface), idxLin);

                tau1_best = tau1_vec(c);
                tau2_best = tau2_vec(r);

                surfR0_best = cost_R0(r,c);
                surfR1_best = cost_R1(r,c);
                surfR2_best = cost_R2(r,c);
            end

            %% (3) MultiStart (5변수)  [공정 평가: cost-surface 결과 미사용]
            xBest    = nan(1,5);
            fBest    = NaN;
            exitflag = NaN;

            if run_multistart

                x0 = [R0_init, R1_init, R2_init, tau1_init_MS, tau2_init_MS];

                % ---- MS bounds (R은 기존 ub, tau는 grid 범위) ----
                lb_MS = [0, 0, 0, tau1_vec(1),  tau2_vec(1)];
                ub_MS = [R0_ub, R1_ub, R2_ub, tau1_vec(end), tau2_vec(end)];

                % (권장) 물리 제약: tau1 <= tau2
                A_lin = [0 0 0 1 -1];
                b_lin = 0;

                problem = createOptimProblem('fmincon', ...
                    'objective', @(x) RMSE_2RC(V2, x, t2, I2, O2), ...
                    'x0', x0, ...
                    'lb', lb_MS, 'ub', ub_MS, ...
                    'Aineq', A_lin, 'bineq', b_lin, ...
                    'options', optMS);

                [xBest, fBest, exitflag] = run(ms, problem, startPts);
            end

            %% (4) Plot
            fig = [];
            if run_cost_surface && ~isempty(cost_surface)
                fig = figure('Name', sprintf('%s | seg %d', base_raw, s), 'NumberTitle', 'off', ...
                    'Color', 'w', 'Position', [100 100 1200 800]);

                surf(T1, T2, cost_surface, 'EdgeColor', 'none', 'FaceColor', 'interp', 'HandleVisibility','off');
                view(3); shading interp; colorbar; hold on;
                xlabel('\tau_1 [s]'); ylabel('\tau_2 [s]'); zlabel('RMSE [V]');
                title(sprintf('%s | seg %d | %ds window', base_raw, s, fit_window_sec), 'Interpreter','none');

                hStar = plot3(tau1_best, tau2_best, rmse_surf_best, 'r*', 'MarkerSize', 12, 'LineWidth', 2);

                if run_multistart && all(isfinite(xBest))
                    hMS = plot3(xBest(4), xBest(5), fBest, 'go', 'MarkerSize', 10, 'LineWidth', 2);
                    legend([hStar, hMS], ...
                        {sprintf('Surface best: (R0=%.4g R1=%.4g R2=%.4g) \\tau1=%.3f \\tau2=%.3f RMSE=%.2f mV', ...
                                  surfR0_best, surfR1_best, surfR2_best, tau1_best, tau2_best, rmse_surf_best*1e3), ...
                         sprintf('MultiStart best: (R0=%.4g R1=%.4g R2=%.4g) \\tau1=%.3f \\tau2=%.3f RMSE=%.2f mV', ...
                                  xBest(1), xBest(2), xBest(3), xBest(4), xBest(5), fBest*1e3)}, ...
                        'Location','best', 'AutoUpdate','off');
                else
                    legend(hStar, sprintf('Surface best: \\tau1=%.3f \\tau2=%.3f (RMSE=%.2f mV)', ...
                        tau1_best, tau2_best, rmse_surf_best*1e3), 'Location','best');
                end
            end

            %% (5) 저장
            segRes = struct();
            segRes.seg_idx = s;
            segRes.t_len = numel(t2);

            segRes.surface_best = struct( ...
                'R0',surfR0_best,'R1',surfR1_best,'R2',surfR2_best, ...
                'tau1',tau1_best,'tau2',tau2_best,'rmse',rmse_surf_best);

            segRes.ms_best      = struct('xBest',xBest,'rmse',fBest,'exitflag',exitflag);

            % 전체 surface 저장(무거우면 아래 4줄 주석처리 가능)
            segRes.cost_surface   = cost_surface;
            segRes.cost_surface_R0 = cost_R0;
            segRes.cost_surface_R1 = cost_R1;
            segRes.cost_surface_R2 = cost_R2;

            results_cell.segs(end+1) = segRes; %#ok<SAGROW>

            out_mat = fullfile(save_dir, sprintf('CostSurface_%s_seg%03d_%ds.mat', base_raw, s, fit_window_sec));
            save(out_mat, 'segRes', '-v7.3');

            % 그림 저장 (그림은 닫지 않음: close(fig) 없음)
            if isgraphics(fig,'figure')
                out_fig = fullfile(save_dir, sprintf('CostSurface_%s_seg%03d_%ds.fig', base_raw, s, fit_window_sec));
                out_png = fullfile(save_dir, sprintf('CostSurface_%s_seg%03d_%ds.png', base_raw, s, fit_window_sec));
                savefig(fig, out_fig);
                exportgraphics(fig, out_png, 'Resolution', 200);
                drawnow;
                % close(fig);  % ★ 주석(요청사항)
            end

            % ---- (NEW) Summary Table row append ----
            base_seg = sprintf('%s_seg%03d', base_raw, s);

            newRow = { ...
                string(base_seg), ...
                surfR0_best, surfR1_best, surfR2_best, tau1_best, tau2_best, rmse_surf_best, ...
                xBest(1), xBest(2), xBest(3), xBest(4), xBest(5), fBest, exitflag ...
                };

            Summary = [Summary; newRow]; %#ok<AGROW>

            fprintf("    saved: %s | surfRMSE=%.2f mV | msRMSE=%.2f mV\n", ...
                base_seg, rmse_surf_best*1e3, fBest*1e3);

        catch ME
            warning("(%s) seg %d 실패: %s", base_raw, s, ME.message);
        end
    end

    % ---- 셀 단위 저장 ----
    out_cell = fullfile(save_dir, sprintf('CostSurface_%s_ALLSEGS_%ds.mat', base_raw, fit_window_sec));
    save(out_cell, 'results_cell', 'Summary', '-v7.3');

    % (추가) csv도 저장(가독성)
    out_csv = fullfile(save_dir, sprintf('Summary_%s_%ds.csv', base_raw, fit_window_sec));
    try
        writetable(Summary, out_csv);
    catch
        warning("CSV 저장 실패: %s", out_csv);
    end

    disp(Summary);
    fprintf("=== CELL DONE: %s | saved %s\n", base_raw, out_cell);
end

fprintf("\n모든 대상 처리 완료!\n");

%% ========================================================================
%  RMSE (2RC)
% ========================================================================
function cost = RMSE_2RC(V_true, para, t, I, OCV)
    V_true = V_true(:);
    V_est  = RC_model_2(para, t(:), I(:), OCV(:));
    V_est  = V_est(:);

    assert(numel(V_true)==numel(V_est), ...
        'RMSE_2RC: length mismatch (true=%d, est=%d)', numel(V_true), numel(V_est));

    cost = sqrt(mean((V_true - V_est).^2, 'omitnan'));
end

%% ========================================================================
%  cropToWindow: winSec 이내로 트리밍
% ========================================================================
function [t2, I2, V2, O2, ok] = cropToWindow(t, I, V, O, winSec)
    ok = true;

    if isempty(t) || numel(t) < 5
        ok = false; t2=[]; I2=[]; V2=[]; O2=[];
        return
    end

    m = (t <= winSec);
    if nnz(m) < 5
        ok = false; t2=[]; I2=[]; V2=[]; O2=[];
        return
    end

    t2 = t(m);
    I2 = I(m);
    V2 = V(m);

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
