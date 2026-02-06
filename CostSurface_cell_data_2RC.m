%% ========================================================================
%  CostSurface_2RC_fromCellData_MS_batch_optimset.m
%
%  실제 셀 데이터(SIM_table: time/current/voltage/OCV_vec)로
%   1) (seg 선택) 600s crop
%   2) τ1–τ2 cost-surface 생성
%       - 각 (tau1,tau2) grid에서 R0,R1,R2만 fmincon으로 피팅 (tau는 고정)
%   3) MultiStart(5변수: [R0 R1 R2 tau1 tau2]) 최적화
%   4) 결과 저장 (mat + fig/png)
%
%  - fmincon 옵션: optimset 통일
%  - winner iteration trajectory: 없음
%  - (FIX) results_cell.segs 구조체 필드 mismatch 오류 해결:
%          results_cell.segs 를 "필드 포함 빈 struct"로 초기화
% ========================================================================

clear; clc; close all;

%% ========================= 사용자 설정 =========================

% --- SIM_parsed 폴더 ---
folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_fresh_integrated_7_Drivingprocessed\SIM_parsed';

% --- 결과 저장 폴더 ---
save_dir  = fullfile(folder_SIM, 'CostSurface_2RC_fromCellData');
if ~exist(save_dir,'dir'), mkdir(save_dir); end

% --- 600s 윈도우 ---
fit_window_sec = 600;

% --- (A) 어떤 셀(SIM 파일) 돌릴지 ---
% mode="all"     : folder_SIM의 *_SIM.mat 전부
% mode="index"   : index 리스트만 (예: [1 2 5 6 8])
% mode="pattern" : 패턴 매칭 (예: "*Cell15*_SIM.mat", "*fresh*_SIM.mat")
cell_select.mode    = "index";        % "all" | "index" | "pattern"
cell_select.index   = [1];            % mode="index"일 때만 사용 [1 2 5 6 8] 등
cell_select.pattern = "*_SIM.mat";    % mode="pattern"일 때 사용

% --- (B) 각 셀에서 어떤 seg 돌릴지 ---
% mode="all"   : 1:nSeg 전부
% mode="index" : seg index 리스트 (1:4, [1 2 5 6 8], 등)
seg_select.mode  = "index";    % "all" | "index"
seg_select.index = 9:16;       % 예) 1:4 / [1 2 5 6 8]

% --- (C) OCV 처리 ---
% SIM_table에 OCV_vec이 없을 경우:
allow_missing_ocv = true;   % true면 OCV=0으로 대체, false면 error

% --- (D) τ grid (너가 바꾼 값 반영) ---
tau1_vec = 10.^(linspace(0, 1.7, 61));     % 1 ~ 50 s
tau2_vec = 10.^(linspace(1, 3.0, 101));    % 10 ~ 1000 s
[T1, T2] = meshgrid(tau1_vec, tau2_vec);

% --- (E) cost-surface / MultiStart on/off ---
run_cost_surface = true;
run_multistart   = true;

% --- (F) 병렬 풀 / MultiStart 설정 ---
if isempty(gcp('nocreate'))
    parpool;
end

ms       = MultiStart("UseParallel", true, "Display", "off");
startPts = RandomStartPointSet('NumStartPoints', 40);

% --- (G) fmincon 옵션: optimset으로 통일 ---
optSurf = optimset('display','off', ...
    'MaxIter',1e3,'MaxFunEvals',1e4, ...
    'TolFun',eps,'TolX',eps);

optMS = optimset('display','off', ...
    'MaxIter',1e3,'MaxFunEvals',2e4, ...
    'TolFun',eps,'TolX',eps);

% --- (H) 파라미터 스케일(초기값/경계) ---
R0_init = 0.003; R1_init = 0.0005; R2_init = 0.0005;

% R 상한 (필요시 조절)
R0_ub = 0.05;   % [ohm]
R1_ub = 0.01;
R2_ub = 0.03;

% MultiStart 전체(5변수) 경계
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

    % seg_list 결정
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

    % ===== (FIX) 구조체 필드 mismatch 방지: 필드 포함 빈 struct로 초기화 =====
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
        'cost_surface', {} );

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

            % 시간축 정규화
            if isduration(t)
                t = seconds(t - t(1));
            else
                t = t - t(1);
            end

            % 600s crop
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

            %% (2) Cost-surface
            cost_surface = [];
            tau1_best = NaN; tau2_best = NaN; rmse_surf_best = NaN;

            if run_cost_surface
                cost_surface = zeros(numel(tau2_vec), numel(tau1_vec));

                for ii = 1:numel(tau1_vec)
                    tau1 = tau1_vec(ii);

                    for jj = 1:numel(tau2_vec)
                        tau2 = tau2_vec(jj);

                        % tau 고정: lb=ub로 묶기
                        p0 = [R0_init, R1_init, R2_init, tau1, tau2];
                        lb = [0,       0,       0,       tau1, tau2];
                        ub = [R0_ub,   R1_ub,   R2_ub,   tau1, tau2];

                        [~, fval] = fmincon(@(p) RMSE_2RC(V2, p, t2, I2, O2), ...
                                            p0, [],[],[],[], lb, ub, [], optSurf);

                        cost_surface(jj, ii) = fval;
                    end
                end

                % surface 최적점
                [rmse_surf_best, idxLin] = min(cost_surface(:));
                [r,c] = ind2sub(size(cost_surface), idxLin);
                tau1_best = tau1_vec(c);
                tau2_best = tau2_vec(r);
            end

            %% (3) MultiStart (5변수)
            xBest = nan(1,5);
            fBest = NaN;
            exitflag = NaN;

            if run_multistart
                % 초기값: surface best 있으면 그걸 사용, 아니면 중앙값
                if ~isnan(tau1_best) && ~isnan(tau2_best)
                    x0 = [R0_init, R1_init, R2_init, tau1_best, tau2_best];
                else
                    x0 = [R0_init, R1_init, R2_init, tau1_vec(round(end/2)), tau2_vec(round(end/2))];
                end

                problem = createOptimProblem('fmincon', ...
                    'objective', @(x) RMSE_2RC(V2, x, t2, I2, O2), ...
                    'x0', x0, 'lb', lb_MS, 'ub', ub_MS, 'options', optMS);

                [xBest, fBest, exitflag] = run(ms, problem, startPts);
            end

            %% (4) Plot (cost-surface가 있을 때만)
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
                        {sprintf('Surface best: \\tau1=%.3f \\tau2=%.3f (RMSE=%.2f mV)', tau1_best, tau2_best, rmse_surf_best*1e3), ...
                         sprintf('MultiStart best: \\tau1=%.3f \\tau2=%.3f (RMSE=%.2f mV)', xBest(4), xBest(5), fBest*1e3)}, ...
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
            segRes.surface_best = struct('tau1',tau1_best,'tau2',tau2_best,'rmse',rmse_surf_best);
            segRes.ms_best      = struct('xBest',xBest,'rmse',fBest,'exitflag',exitflag);

            % cost_surface 저장(무거우면 주석처리 가능)
            segRes.cost_surface = cost_surface;

            % ===== (FIX) 이제 필드 동일하므로 오류 없이 append 됨 =====
            results_cell.segs(end+1) = segRes;

            out_mat = fullfile(save_dir, sprintf('CostSurface_%s_seg%03d_%ds.mat', base_raw, s, fit_window_sec));
            save(out_mat, 'segRes', '-v7.3');

            if isgraphics(fig,'figure')
                out_fig = fullfile(save_dir, sprintf('CostSurface_%s_seg%03d_%ds.fig', base_raw, s, fit_window_sec));
                out_png = fullfile(save_dir, sprintf('CostSurface_%s_seg%03d_%ds.png', base_raw, s, fit_window_sec));
                savefig(fig, out_fig);
                exportgraphics(fig, out_png, 'Resolution', 200);
                close(fig);
            end

            fprintf("    saved: seg %d | surfRMSE=%.2f mV | msRMSE=%.2f mV\n", ...
                s, rmse_surf_best*1e3, fBest*1e3);

        catch ME
            warning("(%s) seg %d 실패: %s", base_raw, s, ME.message);
        end
    end

    out_cell = fullfile(save_dir, sprintf('CostSurface_%s_ALLSEGS_%ds.mat', base_raw, fit_window_sec));
    save(out_cell, 'results_cell', '-v7.3');
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
